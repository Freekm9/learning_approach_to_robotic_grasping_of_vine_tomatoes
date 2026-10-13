#!/usr/bin/env python3
"""Non-learned stem segmentation of a coloured point cloud (step 0 baseline).

Pipeline, per cloud:
  1. Voxel downsample (colours averaged per voxel).
  2. Stem colour mask: the same rules as Babuska's 2D detector (find_tomato_grasp.py),
     a strict green test plus a looser olive test that is only kept where it is
     3D-connected to strict stem points.
  3. Radius outlier removal on the stem points.
  4. Euclidean clustering: connected components of the radius graph. Each
     cluster is one candidate stem system (ideally one truss's peduncle).
  5. Per cluster, a geodesic level-set (Reeb graph) skeleton: geodesic distance
     along the stem from one extremity, binned; each connected piece of a bin
     becomes a node at its centroid, and touching pieces of adjacent bins are
     linked. Short spurs are pruned. Degree-1 nodes are candidate stem ends, the
     longest path is the main stem axis.

No ROS dependency, so it runs offline on saved .pcd files.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree


@dataclass
class Params:
    voxel: float = 0.0015            # m, working resolution
    crop_z_min: float = -0.02        # m, drop points below the crate floor (base frame)
    outlier_radius: float = 0.004    # m
    outlier_min_neighbors: int = 6
    cluster_radius: float = 0.004    # m, max gap bridged inside one stem cluster
    min_cluster_points: int = 150    # at working resolution; ~3 cm of 5 mm stem
    skeleton_bin: float = 0.006      # m, geodesic bin width = skeleton node spacing
    min_branch_length: float = 0.015 # m, shorter end branches are pruned as spurs


@dataclass
class Cluster:
    idx: np.ndarray                  # indices into the downsampled cloud
    nodes: np.ndarray                # (K,3) skeleton nodes
    edges: np.ndarray                # (E,2) skeleton edges (a tree)
    ends: np.ndarray                 # node indices of degree 1
    junctions: np.ndarray            # node indices of degree >= 3
    main_path: np.ndarray            # node indices along the longest path
    main_length: float               # m
    extent: float                    # m, largest bounding-box side


@dataclass
class Result:
    xyz: np.ndarray                  # downsampled cloud
    rgb: np.ndarray
    stem_mask: np.ndarray            # bool over downsampled cloud, after outlier removal
    labels: np.ndarray               # cluster id per point, -1 = not stem / too small
    clusters: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def voxel_downsample(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    # One int64 key per voxel: a 1D unique is several times faster than unique(axis=0).
    dims = keys.max(axis=0) + 1
    _, inv, counts = np.unique((keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2],
                               return_inverse=True, return_counts=True)
    out_xyz = np.stack([np.bincount(inv, xyz[:, d]) for d in range(3)], axis=1)
    out_rgb = np.stack([np.bincount(inv, rgb[:, d]) for d in range(3)], axis=1)
    return out_xyz / counts[:, None], out_rgb / counts[:, None]


def stem_color_masks(rgb):
    """Babuska's 2D rules (find_tomato_grasp._analyze_scene), applied per point."""
    R, G, B = rgb.T
    V = rgb.max(axis=1)
    S = np.divide(V - rgb.min(axis=1), V, out=np.zeros_like(V), where=V > 0)
    color = (G > 1.12 * B) & (G - B > 0.035) & (S > 0.16) & (V > 0.12)
    strict = color & (G > 0.95 * R)
    loose = color & (G > 0.8 * R) & ((G > 0.95 * R) | (S > 0.3))
    return strict, loose


def _components(points, radius):
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type='ndarray')
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def grow_loose(xyz, strict, loose, radius):
    """Keep loose (olive) points only in components of the loose set that contain a
    strict point: the 3D version of Babuska's morphological reconstruction."""
    cand = np.flatnonzero(loose | strict)
    labels = _components(xyz[cand], radius)
    seeded = np.unique(labels[strict[cand]])
    keep = np.zeros(len(xyz), bool)
    keep[cand[np.isin(labels, seeded)]] = True
    return keep


def remove_outliers(xyz, mask, radius, min_neighbors):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    counts = np.array([len(n) for n in cKDTree(xyz[idx]).query_ball_point(xyz[idx], radius)]) - 1
    out = np.zeros_like(mask)
    out[idx[counts >= min_neighbors]] = True
    return out


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _radius_graph(points, radius):
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    w = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    n = len(points)
    return pairs, coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()


def skeleton(points, p):
    """Geodesic level-set skeleton of one (connected) cluster, then spur pruning."""
    pairs, graph = _radius_graph(points, p.cluster_radius)
    # Root at a geodesic extremity: the point farthest from an arbitrary start.
    d0 = shortest_path(graph, directed=False, indices=0)
    root = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    dist = shortest_path(graph, directed=False, indices=root)
    dist[~np.isfinite(dist)] = 0
    bins = np.floor(dist / p.skeleton_bin).astype(int)

    # Split every bin into pieces connected within the bin: one node per piece.
    same = bins[pairs[:, 0]] == bins[pairs[:, 1]]
    n_pts = len(points)
    g_same = coo_matrix((np.ones(same.sum()), (pairs[same, 0], pairs[same, 1])), shape=(n_pts, n_pts))
    _, piece = connected_components(g_same, directed=False)
    _, node_of = np.unique(piece, return_inverse=True)
    node_of = node_of.ravel()
    n = node_of.max() + 1
    nodes = np.zeros((n, 3))
    np.add.at(nodes, node_of, points)
    nodes /= np.bincount(node_of, minlength=n)[:, None]

    # Link pieces that touch across a bin boundary; the MST removes the rare cycles.
    cross = node_of[pairs[~same]]
    cross = np.unique(np.sort(cross[cross[:, 0] != cross[:, 1]], axis=1), axis=0)
    if len(cross) == 0:
        return nodes, np.zeros((0, 2), int)
    w = np.linalg.norm(nodes[cross[:, 0]] - nodes[cross[:, 1]], axis=1)
    mst = minimum_spanning_tree(coo_matrix((w, (cross[:, 0], cross[:, 1])), shape=(n, n))).tocoo()
    edges = np.stack([mst.row, mst.col], axis=1)

    # Prune: repeatedly remove end branches (end node up to the first junction)
    # shorter than min_branch_length. Removing a spur can turn a junction into a
    # plain path node, so iterate until nothing changes.
    alive = np.ones(n, bool)
    edge_set = {tuple(sorted(e)) for e in edges}
    changed = True
    while changed:
        changed = False
        adj = _adjacency(n, edge_set)
        deg = np.array([len(a) for a in adj])
        for end in np.flatnonzero((deg == 1) & alive):
            # Walk from the end through degree-2 nodes to the first other node.
            path, prev, cur, length = [end], -1, end, 0.0
            while True:
                nxt = next(q for q in adj[cur] if q != prev)
                length += np.linalg.norm(nodes[nxt] - nodes[cur])
                prev, cur = cur, nxt
                if deg[cur] != 2:
                    break
                path.append(cur)
            # cur is now a junction (deg >= 3) or the other end of an isolated path.
            if deg[cur] >= 3 and length < p.min_branch_length:
                for a, b in zip(path, path[1:] + [cur]):
                    edge_set.discard(tuple(sorted((a, b))))
                alive[path] = False
                changed = True
                break  # degrees changed; recompute
    edges = np.array(sorted(edge_set), dtype=int).reshape(-1, 2)
    # Re-index to the surviving nodes.
    used = np.unique(edges) if len(edges) else np.array([0])
    remap = -np.ones(n, int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[edges]


def analyse_tree(nodes, edges):
    n = len(nodes)
    deg = np.bincount(edges.ravel(), minlength=n) if len(edges) else np.zeros(n, int)
    ends = np.flatnonzero(deg == 1)
    junctions = np.flatnonzero(deg >= 3)
    if len(edges) == 0:
        return ends, junctions, np.array([0]), 0.0
    w = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    g = coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    # Tree diameter by double sweep.
    d0 = shortest_path(g, directed=False, indices=ends[0] if len(ends) else 0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = shortest_path(g, directed=False, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return ends, junctions, np.array(path[::-1]), float(da[b])


def segment(xyz, rgb, p=Params()):
    t = {}
    t0 = time.time()
    keep = xyz[:, 2] > p.crop_z_min
    xyz, rgb = voxel_downsample(xyz[keep], rgb[keep], p.voxel)
    t['downsample'] = time.time() - t0

    t0 = time.time()
    strict, loose = stem_color_masks(rgb)
    stem = grow_loose(xyz, strict, loose, 1.5 * p.voxel)
    stem = remove_outliers(xyz, stem, p.outlier_radius, p.outlier_min_neighbors)
    t['color_mask'] = time.time() - t0

    t0 = time.time()
    labels = -np.ones(len(xyz), int)
    idx = np.flatnonzero(stem)
    clusters = []
    if len(idx):
        comp = _components(xyz[idx], p.cluster_radius)
        sizes = np.bincount(comp)
        order = np.argsort(-sizes)
        for c in order:
            if sizes[c] < p.min_cluster_points:
                break
            members = idx[comp == c]
            labels[members] = len(clusters)
            clusters.append(members)
    t['clustering'] = time.time() - t0

    t0 = time.time()
    out = []
    for members in clusters:
        pts = xyz[members]
        nodes, edges = skeleton(pts, p)
        ends, junctions, main_path, main_length = analyse_tree(nodes, edges)
        out.append(Cluster(members, nodes, edges, ends, junctions, main_path, main_length,
                           float(np.ptp(pts, axis=0).max())))
    t['skeleton'] = time.time() - t0
    return Result(xyz, rgb, stem, labels, out, t)


def main():
    """Usage: segment_stems.py cloud.pcd [more.pcd ...] -- prints a per-cluster summary."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pcd_io import load_pcd
    for path in sys.argv[1:]:
        r = segment(*load_pcd(path))
        print(f"{path}: {len(r.xyz)} points, {int(r.stem_mask.sum())} stem points, "
              f"{len(r.clusters)} clusters ({sum(r.timings.values()):.2f} s)")
        for i, c in enumerate(r.clusters):
            print(f"  cluster {i}: {len(c.idx)} pts, extent {100 * c.extent:.1f} cm, "
                  f"main path {100 * c.main_length:.1f} cm, {len(c.ends)} ends, "
                  f"{len(c.junctions)} junctions")


if __name__ == '__main__':
    main()
