"""Snap a clicked image point to the nearest stem found by segment_stems.

The click is matched in the image: every cluster's skeleton is projected into the
camera image and the nearest skeleton segment within max_px wins. Segments on a
cluster's main path (the peduncle) are preferred over side branches (pedicels, sepals)
when both are within reach, so a click near a junction lands on the peduncle. The 3D
grasp point is the matching point on that segment. The stem direction is the main axis
(PCA) of the main-path skeleton within +-pca_radius along the stem, or, on a side
branch, of the cluster's stem points within pca_radius.

No ROS dependency: works on numpy arrays, so it can be tested offline.
"""
from dataclasses import dataclass

import numpy as np

from segment_stems import Params, segment

DEFAULT_MAX_PX = 25        # clicks further than this from any stem are not snapped
DEFAULT_PCA_RADIUS = 0.012 # m, local neighbourhood for the stem direction
MIN_PCA_POINTS = 10
AHEAD_M = 0.02             # length of the direction vector projected into the image
# The skeleton point is the mean of the stem points around it, which sits below the
# stem's top surface: the camera sees only the top half, and on thin stems the edge
# pixels take their depth from whatever lies behind the stem. The grasp depth is
# therefore taken from the near surface instead: this percentile of the depths of the
# stem points within SURFACE_RADIUS_M of the line of sight through the snapped point.
SURFACE_PERCENTILE = 10
SURFACE_RADIUS_M = 0.004
MIN_SURFACE_POINTS = 5


@dataclass
class Snap:
    point_cam: np.ndarray    # (3,) grasp point in the camera frame
    axis_cam: np.ndarray     # (3,) unit stem direction in the camera frame
    uv: tuple                # snapped pixel (x, y)
    uv_ahead: tuple          # pixel AHEAD_M further along the stem, sets the yaw
    cluster: int
    distance_px: float       # click-to-stem distance
    on_main_path: bool       # snapped to the peduncle rather than a side branch
    pca_points: int          # points used for the direction (0 = segment fallback)


def depth_to_cloud(depth_m, rgb, K):
    """Deproject a depth image (metres, 0 = invalid) aligned with rgb (uint8, HxWx3)
    into camera-frame points and 0..1 colours, with the pinhole model of K."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = (depth_m > 0) & np.isfinite(depth_m)
    z = depth_m[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1), rgb[ok].astype(np.float64) / 255.0


def _apply(m, pts):
    return pts @ m[:3, :3].T + m[:3, 3]


class StemSnapper:
    """Segment once per image, then snap any number of clicks against the result."""

    def __init__(self, xyz_cam, rgb, cam_to_base, K, params=Params()):
        self.K = K
        self.cam_to_base = cam_to_base
        self.base_to_cam = np.linalg.inv(cam_to_base)
        # Segment in the planning frame, where the crate-floor crop applies.
        self.result = segment(_apply(cam_to_base, xyz_cam), rgb, params)
        self.xyz_cam = _apply(self.base_to_cam, self.result.xyz)
        self.nodes_cam, self.nodes_uv, self._main_edges = [], [], []
        for c in self.result.clusters:
            on_path = {tuple(sorted(e)) for e in zip(c.main_path[:-1], c.main_path[1:])}
            self._main_edges.append(np.array([tuple(sorted(e)) in on_path for e in c.edges.tolist()], bool))
            n = _apply(self.base_to_cam, c.nodes)
            self.nodes_cam.append(n)
            self.nodes_uv.append(self.project(n))

    def project(self, pts_cam):
        z = np.where(pts_cam[:, 2] > 1e-6, pts_cam[:, 2], np.nan)
        return np.stack([self.K[0, 0] * pts_cam[:, 0] / z + self.K[0, 2],
                         self.K[1, 1] * pts_cam[:, 1] / z + self.K[1, 2]], axis=1)

    @property
    def n_clusters(self):
        return len(self.result.clusters)

    def snap(self, click, max_px=DEFAULT_MAX_PX, pca_radius=DEFAULT_PCA_RADIUS):
        """Return a Snap for click (x, y), or None if no stem is within max_px."""
        click = np.asarray(click, dtype=float)
        best = None
        for ci, (c, uv) in enumerate(zip(self.result.clusters, self.nodes_uv)):
            if len(c.edges) == 0:
                continue
            main = self._main_edges[ci]
            a, b = uv[c.edges[:, 0]], uv[c.edges[:, 1]]
            ab = b - a
            t = np.clip(np.einsum('ij,ij->i', click - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0, 1)
            d = np.linalg.norm(a + t[:, None] * ab - click, axis=1)
            d[~np.isfinite(d)] = np.inf
            for k in np.flatnonzero(d <= max_px):
                # Rank: main-path segments first, then pixel distance.
                key = (not main[k], d[k])
                if best is None or key < best[0]:
                    best = (key, ci, k, t[k])
        if best is None:
            return None
        (off_main, dist), ci, k, t = best
        c = self.result.clusters[ci]
        na, nb = self.nodes_cam[ci][c.edges[k]]
        point = self._on_surface(ci, na + t * (nb - na))
        if off_main:
            axis, used = self._local_axis(ci, point, pca_radius, fallback=nb - na)
        else:
            axis, used = self._path_axis(ci, point, pca_radius)
        uv = self.project(point[None])[0]
        uv_ahead = self.project((point + AHEAD_M * axis)[None])[0]
        return Snap(point, axis, tuple(int(v) for v in np.round(uv)), tuple(int(v) for v in np.round(uv_ahead)),
                    ci, float(dist), not off_main, used)

    def _on_surface(self, ci, point):
        """Move point along its line of sight to the stem's near (top) surface."""
        pts = self.xyz_cam[self.result.clusters[ci].idx]
        ray = point / np.linalg.norm(point)
        along = pts @ ray
        off_ray = np.linalg.norm(pts - along[:, None] * ray, axis=1)
        near = along[off_ray < SURFACE_RADIUS_M]
        if len(near) < MIN_SURFACE_POINTS:
            return point
        return ray * min(np.percentile(near, SURFACE_PERCENTILE), np.linalg.norm(point))

    def _path_axis(self, ci, point, radius):
        """PCA of the main-path skeleton nodes within +-radius (along the path) of point."""
        path = self.nodes_cam[ci][self.result.clusters[ci].main_path]
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        here = arc[np.argmin(np.linalg.norm(path - point, axis=1))]
        near = path[np.abs(arc - here) <= radius]
        if len(near) < 2:
            near = path[max(0, np.searchsorted(arc, here) - 1):][:2]
        axis = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)[2][0]
        axis = axis / np.linalg.norm(axis)
        return (axis if axis[0] >= 0 else -axis), len(near)

    def _local_axis(self, ci, point, radius, fallback):
        pts = self.xyz_cam[self.result.clusters[ci].idx]
        near = pts[np.linalg.norm(pts - point, axis=1) < radius]
        if len(near) >= MIN_PCA_POINTS:
            centred = near - near.mean(axis=0)
            axis = np.linalg.svd(centred, full_matrices=False)[2][0]
            used = len(near)
        else:
            axis, used = fallback, 0
        axis = axis / np.linalg.norm(axis)
        # Sign is arbitrary for a stem; keep it pointing right in the image for stable drawing.
        return (axis if axis[0] >= 0 else -axis), used
