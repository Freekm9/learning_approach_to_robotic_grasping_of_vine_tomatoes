#!/usr/bin/env python3
"""Save one full camera point cloud (the whole crate, not the per-truss map) as an
ASCII PCD in the planning frame, for offline stem segmentation, and the camera's colour
image of the same scan as a PNG with the same name.

Move the arm to the overview pose first, then:
    rosrun grasp capture_full_cloud.py [output_dir] [_topic:=/other/points] [_image_topic:=/other/image]
Default output_dir: ~/Crate_truss_pick-and-place/experiments/full_crate_clouds
"""
import os
import sys
import time

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

# The camera runs in the /panda namespace (vine_grasping.launch); create_map_from_pointclouds
# subscribes to the relative name from inside it. Override with _topic:=... if needed.
DEFAULT_TOPIC = '/panda/camera/depth/color/points'
DEFAULT_IMAGE_TOPIC = '/panda/camera/color/image_raw'
DEFAULT_DIR = os.path.expanduser('~/Crate_truss_pick-and-place/experiments/full_crate_clouds')


def write_pcd(path, xyz, rgb_packed):
    with open(path, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\n'
                'SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n'
                f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                f'POINTS {len(xyz)}\nDATA ascii\n')
        for (x, y, z), c in zip(xyz, rgb_packed):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {c}\n')


def main():
    argv = rospy.myargv()  # drops ROS remappings such as _topic:=...
    out_dir = argv[1] if len(argv) > 1 else DEFAULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rospy.init_node('capture_full_cloud', anonymous=True)
    planning_frame = rospy.get_param('/planning_frame', 'panda_link0')
    topic = rospy.get_param('~topic', DEFAULT_TOPIC)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    print(f'Waiting for a cloud on {topic}...')
    cloud = rospy.wait_for_message(topic, PointCloud2, timeout=10)
    transform = tf_buffer.lookup_transform(planning_frame, cloud.header.frame_id,
                                           rospy.Time(0), rospy.Duration(2))
    cloud = do_transform_cloud(cloud, transform)

    pts = np.array(list(point_cloud2.read_points(cloud, field_names=('x', 'y', 'z', 'rgb'),
                                                 skip_nans=True)), dtype=np.float32)
    rgb_packed = pts[:, 3].copy().view(np.uint32)  # rgb is a float32 holding packed bytes
    path = os.path.join(out_dir, time.strftime('%Y%m%d_%H%M%S') + '.pcd')
    write_pcd(path, pts[:, :3], rgb_packed)
    print(f'Saved {len(pts)} points in {planning_frame} to {path}')

    # The camera's own view of this scan, same name as the cloud. A missing image must not cost the scan.
    image_topic = rospy.get_param('~image_topic', DEFAULT_IMAGE_TOPIC)
    try:
        image = rospy.wait_for_message(image_topic, Image, timeout=5)
        image_path = os.path.splitext(path)[0] + '.png'
        if not cv2.imwrite(image_path, CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')):
            raise IOError(f'could not write {image_path}')
        print(f'Saved the camera view to {image_path}')
    except Exception as e:
        print(f'WARNING: no camera view saved ({e}); the cloud is saved. Check {image_topic} and scan again if you need the image.')


if __name__ == '__main__':
    main()
