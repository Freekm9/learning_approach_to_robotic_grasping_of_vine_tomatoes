#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive plan_movement.py's move to pre-grasp over the replayed crate cloud, to watch
the crate-contents obstacle work (see launch/crate_contents_sim_test.launch).

For each grasp point (x, y; z is taken from the cloud: 3 cm below the top of the
tomatoes there), it sets a straight-down grasp pose and calls move_robot
'go_to_pre_grasp' -- the same planned move grasp() makes, with the crate contents
added while planning. By default it hops between opposite corners of the crate
without returning home, the case where the arm used to cut through the tomatoes.
--home returns to the saved pose between points instead.

Watch it in RViz: add a PlanningScene display (topic /panda/move_group/monitored_planning_scene)
to see the 'crate_contents' columns appear with a hole at the grasp point, and a
PointCloud2 display on /panda/camera/depth/color/points for the raw cloud.

Usage:
    rosrun grasp test_crate_contents_sim.py
    rosrun grasp test_crate_contents_sim.py --points 0.40 0.00 0.52 0.20 --home
    rosrun grasp test_crate_contents_sim.py --compare   # each point without, then with the obstacle
"""
import argparse

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from grasp.srv import pipeline_command, set_grasp_pose_command
from common.crate_contents import cloud_to_xyz
from move_robot.test_crate_reachability import build_straight_down_pose

DEFAULT_POINTS = [0.38, -0.04, 0.54, 0.24, 0.54, -0.04, 0.38, 0.24]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--points', type=float, nargs='+', default=DEFAULT_POINTS, help="x1 y1 x2 y2 ... (m)")
    parser.add_argument('--yaw-deg', type=float, default=0.0)
    parser.add_argument('--home', action='store_true', help="go_to_saved_pose between points")
    parser.add_argument('--compare', action='store_true',
                        help="move to each point twice: obstacle off, back to the previous point, obstacle on")
    args = parser.parse_args(rospy.myargv()[1:])
    if len(args.points) % 2:
        parser.error("--points needs x y pairs")

    rospy.init_node('test_crate_contents_sim')
    ns = f'/{args.robot_name}'
    rospy.wait_for_service(f'{ns}/move_robot')
    move_robot = rospy.ServiceProxy(f'{ns}/move_robot', pipeline_command)
    set_grasp_pose = rospy.ServiceProxy(f'{ns}/set_grasp_pose', set_grasp_pose_command)
    param = f'{ns}/plan_movement/avoid_crate_contents'
    frame = rospy.get_param('/planning_frame', f'{args.robot_name}_link0')

    cloud = rospy.wait_for_message(f'{ns}/camera/depth/color/points', PointCloud2, timeout=10)
    xyz = cloud_to_xyz(cloud)  # replayed in the planning frame already

    def go(x, y, avoid):
        near = xyz[(np.abs(xyz[:, 0] - x) < 0.015) & (np.abs(xyz[:, 1] - y) < 0.015) & (xyz[:, 2] < 0.25)]
        z = (np.percentile(near[:, 2], 95) - 0.03) if len(near) else 0.03
        rospy.set_param(param, avoid)
        set_grasp_pose(build_straight_down_pose(x, y, z, args.yaw_deg, frame))
        result = move_robot('go_to_pre_grasp').success
        print(f"  pre-grasp over ({x:.2f}, {y:.2f}, grasp z {z:.3f}) with obstacle {'ON ' if avoid else 'OFF'}: {result}")
        return result

    points = list(zip(args.points[::2], args.points[1::2]))
    try:
        for i, (x, y) in enumerate(points):
            print(f"[{i + 1}/{len(points)}]")
            if args.compare:
                go(x, y, False)
                input("  Watched it? Enter to go back to the previous point and repeat with the obstacle ON...")
                if i > 0:
                    go(*points[i - 1], True)
                elif args.home:
                    move_robot('go_to_saved_pose')
            go(x, y, True)
            if args.home:
                move_robot('go_to_saved_pose')
    finally:
        rospy.set_param(param, True)


if __name__ == '__main__':
    main()
