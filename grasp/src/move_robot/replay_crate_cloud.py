#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish a saved crate point cloud (capture_full_cloud.py output, already in the
planning frame) as if it came from the gripper camera, so plan_movement.py's
crate-contents obstacle (common/crate_contents.py) can be tested in Gazebo, which has
no simulated camera. See launch/crate_contents_sim_test.launch.

Usage:
    rosrun grasp replay_crate_cloud.py _pcd:=/path/to/cloud.pcd _topic:=/panda/camera/depth/color/points
"""
import glob
import os
import sys

import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import PointCloud2, PointField

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stem_segmentation'))
from pcd_io import load_pcd  # noqa: E402  (stem_segmentation isn't a catkin python package)


def latest_saved_cloud():
    ws = os.path.dirname(os.path.dirname(os.path.dirname(rospkg.RosPack().get_path('grasp'))))
    clouds = sorted(glob.glob(os.path.join(ws, 'experiments', 'full_crate_clouds', '*.pcd')))
    if not clouds:
        raise RuntimeError("No saved clouds in experiments/full_crate_clouds -- pass _pcd:=...")
    return clouds[-1]


def main():
    rospy.init_node('replay_crate_cloud')
    path = rospy.get_param('~pcd', '') or latest_saved_cloud()
    topic = rospy.get_param('~topic', '/panda/camera/depth/color/points')
    frame = rospy.get_param('~frame_id', rospy.get_param('/planning_frame', 'panda_link0'))
    z_offset = rospy.get_param('~z_offset', 0.0)  # e.g. to fit a cloud from another crate's floor height

    xyz, _ = load_pcd(path)
    xyz = (xyz + [0, 0, z_offset]).astype(np.float32)
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    msg.is_bigendian = False
    msg.point_step, msg.row_step = 12, 12 * len(xyz)
    msg.is_dense = True
    msg.data = xyz.tobytes()

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
    rospy.loginfo(f"Replaying {len(xyz)} points from {path} on {topic} (frame {frame}) at 2 Hz")
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    main()
