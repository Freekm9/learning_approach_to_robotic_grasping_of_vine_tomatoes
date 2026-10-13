#!/usr/bin/env python3
"""Live stem segmentation: runs segment_stems on the latest camera cloud and publishes
the result, so its consistency can be checked while the arm moves.

    rosrun grasp stem_segmentation_node.py      (next to vine_grasping.launch)

Publishes (all in the /stem_segmentation namespace):
    image     sensor_msgs/Image        colour image with clusters, skeletons and stem
                                       ends drawn on it (rqt_image_view)
    clusters  sensor_msgs/PointCloud2  stem points coloured by cluster (RViz)
    skeleton  visualization_msgs/MarkerArray  skeleton lines and stem ends (RViz)
Each processed frame is also appended to
    ~/Crate_truss_pick-and-place/experiments/stem_segmentation/live_log.csv

Only the newest cloud is processed (no backlog), at most ~rate times per second.
"""
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_stems import Params, segment  # noqa: E402

# Same categorical palette as the results page, RGB 0-255.
PALETTE = [(47, 125, 209), (232, 134, 42), (47, 163, 90), (209, 66, 63), (139, 92, 199),
           (138, 90, 68), (212, 98, 168), (127, 140, 42), (24, 163, 181), (201, 162, 39)]
UNCLUSTERED = (150, 150, 150)
END_COLOR = (230, 40, 30)
LOG_PATH = os.path.expanduser('~/Crate_truss_pick-and-place/experiments/stem_segmentation/live_log.csv')


def cloud_to_numpy(msg):
    """xyz (N,3) float64 and rgb (N,3) in 0..1 from a PointCloud2 with x, y, z, rgb fields."""
    offsets = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([buf[:, offsets[c]:offsets[c] + 4].copy().view(np.float32).ravel()
                    for c in 'xyz'], axis=1).astype(np.float64)
    packed = buf[:, offsets['rgb']:offsets['rgb'] + 4].copy().view(np.uint32).ravel()
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1) / 255.0
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok], rgb[ok]


def transform_matrix(tf_msg):
    t, q = tf_msg.transform.translation, tf_msg.transform.rotation
    m = quaternion_matrix([q.x, q.y, q.z, q.w])
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def apply(m, pts):
    return pts @ m[:3, :3].T + m[:3, 3]


class StemSegmentationNode:
    def __init__(self):
        self.planning_frame = rospy.get_param('/planning_frame', 'panda_link0')
        self.rate = rospy.get_param('~rate', 3.0)
        self.max_range = rospy.get_param('~max_range', 0.9)  # m from the camera
        self.log = rospy.get_param('~log', True)
        self.params = Params()

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)
        self.lock = threading.Lock()
        self.cloud, self.image, self.K = None, None, None
        self.frame = 0

        rospy.Subscriber(rospy.get_param('~cloud_topic', '/panda/camera/depth/color/points'),
                         PointCloud2, self._store('cloud'), queue_size=1, buff_size=2 ** 25)
        rospy.Subscriber(rospy.get_param('~image_topic', '/panda/camera/color/image_raw'),
                         Image, self._store('image'), queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(rospy.get_param('~info_topic', '/panda/camera/color/camera_info'),
                         CameraInfo, self._info_cb, queue_size=1)
        self.image_pub = rospy.Publisher('/stem_segmentation/image', Image, queue_size=1)
        self.cloud_pub = rospy.Publisher('/stem_segmentation/clusters', PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher('/stem_segmentation/skeleton', MarkerArray, queue_size=1)

        if self.log:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            new = not os.path.exists(LOG_PATH)
            self.log_file = open(LOG_PATH, 'a', newline='')
            self.writer = csv.writer(self.log_file)
            if new:
                self.writer.writerow(['stamp', 'cam_x', 'cam_y', 'cam_z', 'stem_points', 'clusters',
                                      'sizes', 'main_path_cm', 'ends', 'seconds'])
            rospy.on_shutdown(self.log_file.close)

    def _store(self, name):
        def cb(msg):
            with self.lock:
                setattr(self, name, msg)
        return cb

    def _info_cb(self, msg):
        self.K = np.array(msg.K).reshape(3, 3)

    def spin(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            with self.lock:
                cloud, self.cloud = self.cloud, None
                image = self.image
            if cloud is not None:
                try:
                    self.process(cloud, image)
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                        tf2_ros.ConnectivityException) as e:
                    rospy.logwarn_throttle(5, f'stem_segmentation: no transform yet ({e})')
            rate.sleep()

    def process(self, cloud, image):
        t0 = time.time()
        xyz_cam, rgb = cloud_to_numpy(cloud)
        near = xyz_cam[:, 2] < self.max_range
        xyz_cam, rgb = xyz_cam[near], rgb[near]
        tf_msg = self.tf_buffer.lookup_transform(self.planning_frame, cloud.header.frame_id,
                                                 rospy.Time(0), rospy.Duration(0.5))
        cam_to_base = transform_matrix(tf_msg)
        result = segment(apply(cam_to_base, xyz_cam), rgb, self.params)
        seconds = time.time() - t0
        self.frame += 1

        header = Header(stamp=cloud.header.stamp, frame_id=self.planning_frame)
        self.cloud_pub.publish(self._cluster_cloud(result, header))
        self.marker_pub.publish(self._markers(result, header))
        if image is not None and self.K is not None:
            base_to_cam = np.linalg.inv(cam_to_base)
            self.image_pub.publish(self._overlay(result, image, base_to_cam, seconds))
        if self.log:
            big = result.clusters[0] if result.clusters else None
            self.writer.writerow([f'{cloud.header.stamp.to_sec():.3f}',
                                  *np.round(cam_to_base[:3, 3], 4),
                                  int(result.stem_mask.sum()), len(result.clusters),
                                  ' '.join(str(len(c.idx)) for c in result.clusters),
                                  round(100 * big.main_length, 1) if big else '',
                                  len(big.ends) if big else '', round(seconds, 3)])
            self.log_file.flush()
        rospy.loginfo_throttle(5, f'stem_segmentation: {len(result.clusters)} clusters, '
                                  f'{int(result.stem_mask.sum())} stem points, {seconds:.2f} s')

    def _cluster_cloud(self, r, header):
        idx = np.flatnonzero(r.stem_mask)
        cols = np.array([PALETTE[l % len(PALETTE)] if l >= 0 else UNCLUSTERED
                         for l in r.labels[idx]], dtype=np.uint32).reshape(-1, 3)
        packed = ((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]).astype(np.uint32)
        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), PointField('rgb', 12, PointField.UINT32, 1)]
        pts = [(x, y, z, c) for (x, y, z), c in zip(r.xyz[idx], packed)]
        return point_cloud2.create_cloud(header, fields, pts)

    def _markers(self, r, header):
        arr = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
        for i, c in enumerate(r.clusters):
            col = PALETTE[i % len(PALETTE)]
            lines = Marker(header=header, ns='skeleton', id=i, type=Marker.LINE_LIST, action=Marker.ADD)
            lines.scale.x = 0.002
            lines.color = ColorRGBA(col[0] / 255, col[1] / 255, col[2] / 255, 1.0)
            lines.pose.orientation.w = 1.0
            lines.points = [Point(*c.nodes[k]) for e in c.edges for k in e]
            ends = Marker(header=header, ns='ends', id=i, type=Marker.SPHERE_LIST, action=Marker.ADD)
            ends.scale.x = ends.scale.y = ends.scale.z = 0.006
            ends.color = ColorRGBA(END_COLOR[0] / 255, END_COLOR[1] / 255, END_COLOR[2] / 255, 1.0)
            ends.pose.orientation.w = 1.0
            ends.points = [Point(*c.nodes[k]) for k in c.ends]
            arr.markers += [lines, ends]
        return arr

    def _project(self, pts_base, base_to_cam, shape):
        p = apply(base_to_cam, pts_base)
        front = p[:, 2] > 0.01
        uv = (p[:, :2] / np.where(front, p[:, 2], 1)[:, None]) @ self.K[:2, :2].T + self.K[:2, 2]
        uv = np.round(uv).astype(int)
        ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
        return uv, ok

    def _overlay(self, r, image_msg, base_to_cam, seconds):
        img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        # Fade the photo towards grey so the cluster colours stand out.
        out = cv2.addWeighted(img, 0.55, np.full_like(img, 128), 0.45, 0)
        bgr = lambda c: (int(c[2]), int(c[1]), int(c[0]))

        idx = np.flatnonzero(r.stem_mask)
        uv, ok = self._project(r.xyz[idx], base_to_cam, out.shape)
        for (u, v), l in zip(uv[ok], r.labels[idx][ok]):
            cv2.circle(out, (u, v), 2, bgr(PALETTE[l % len(PALETTE)] if l >= 0 else UNCLUSTERED), -1)
        for i, c in enumerate(r.clusters):
            nuv, nok = self._project(c.nodes, base_to_cam, out.shape)
            for a, b in c.edges:
                if nok[a] and nok[b]:
                    cv2.line(out, tuple(nuv[a]), tuple(nuv[b]), (20, 20, 20), 4, cv2.LINE_AA)
                    cv2.line(out, tuple(nuv[a]), tuple(nuv[b]), bgr(PALETTE[i % len(PALETTE)]), 2, cv2.LINE_AA)
            for k in c.ends:
                if nok[k]:
                    cv2.circle(out, tuple(nuv[k]), 6, bgr(END_COLOR), 2, cv2.LINE_AA)
            mid = c.main_path[len(c.main_path) // 2]
            if nok[mid]:
                u, v = nuv[mid]
                label = f'{i}: {100 * c.main_length:.0f} cm'
                cv2.putText(out, label, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, label, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        hud = (f'frame {self.frame} | {len(r.clusters)} clusters | '
               f'{int(r.stem_mask.sum())} stem pts | {seconds:.2f} s')
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(out, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        msg.header = image_msg.header
        return msg


if __name__ == '__main__':
    rospy.init_node('stem_segmentation')
    StemSegmentationNode().spin()
