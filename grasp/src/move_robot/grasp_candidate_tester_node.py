#!/usr/bin/env python3
"""
Interactive grasp candidate tester.
Moves the robot through each IK-filtered candidate pose one by one so you can
visually verify which approach angles are physically valid in the real setup.

Controls (type in terminal and press Enter):
  n  – next candidate
  p  – previous candidate
  g  – execute grasp at current candidate and exit
  q  – quit without grasping

Launch alongside the normal pipeline. The tester re-uses the existing
move_robot, set_grasp_pose, and find_grasp_candidates_* services.
"""
import sys
import rospy
import moveit_commander
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, PoseArray
from sensor_msgs.msg import PointCloud2
from grasp.srv import (pipeline_command, set_grasp_pose_command,
                       find_grasp_candidates_command, create_map_command)


def _euler_from_pose(pose):
    o = pose.orientation
    rot = R.from_quat([o.x, o.y, o.z, o.w])
    return np.degrees(rot.as_euler('xyz'))


class GraspCandidateTester:
    def __init__(self):
        rospy.init_node('grasp_candidate_tester')
        robot_name = rospy.get_param('/robot_name', 'panda')
        planning_frame = rospy.get_param('/planning_frame', robot_name + '_link0')

        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander(robot_name + "_manipulator")
        self.move_group.set_end_effector_link(robot_name + '_hand_tcp')
        self.move_group.set_planning_pipeline_id("ompl")
        self.move_group.set_planner_id("RRTConnect")
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.2)
        self.planning_frame = planning_frame

        rospy.wait_for_service('move_robot', timeout=30)
        self.move_robot_srv = rospy.ServiceProxy('move_robot', pipeline_command)

        rospy.wait_for_service('set_grasp_pose', timeout=10)
        self.set_grasp_pose_srv = rospy.ServiceProxy('set_grasp_pose', set_grasp_pose_command)

        self.candidates = []
        self.index = 0

    def load_from_pose_array(self, pose_array):
        self.candidates = []
        for pose in pose_array.poses:
            ps = PoseStamped()
            ps.header = pose_array.header
            ps.pose = pose
            self.candidates.append(ps)
        print(f"Loaded {len(self.candidates)} candidates")

    def _move_to(self, index):
        pose = self.candidates[index]
        self.move_group.set_pose_target(pose)
        success, plan, _, _ = self.move_group.plan()
        euler = _euler_from_pose(pose.pose)
        pos = pose.pose.position
        print(f"\n--- Candidate {index + 1}/{len(self.candidates)} ---")
        print(f"  Position : x={pos.x:.3f}  y={pos.y:.3f}  z={pos.z:.3f}")
        print(f"  Euler XYZ: {euler[0]:.1f}°  {euler[1]:.1f}°  {euler[2]:.1f}°")
        print(f"  Reachable: {success}")
        if success:
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def run(self):
        if not self.candidates:
            print("No candidates loaded. Exiting.")
            return

        while not rospy.is_shutdown():
            self._move_to(self.index)
            try:
                key = input("  Controls – n:next  p:prev  g:grasp  q:quit  > ").strip().lower()
            except EOFError:
                break

            if key == 'n':
                self.index = min(self.index + 1, len(self.candidates) - 1)
            elif key == 'p':
                self.index = max(self.index - 1, 0)
            elif key == 'g':
                print("Executing grasp on current candidate...")
                self.set_grasp_pose_srv(self.candidates[self.index])
                result = self.move_robot_srv('grasp')
                print(f"Grasp result: {result.success}")
                break
            elif key == 'q':
                print("Quitting without grasp.")
                break
            else:
                print("Unknown key. Use n/p/g/q.")


def main():
    tester = GraspCandidateTester()

    # Wait for the grasp candidate topic (published by the pipeline after detection)
    robot_name = rospy.get_param('/robot_name', 'panda')
    topic = f"/{robot_name}/grasp_candidates"
    print(f"Waiting for {topic} (trigger candidate detection in the pipeline first)...")
    try:
        pose_array = rospy.wait_for_message(topic, PoseArray, timeout=120)
        tester.load_from_pose_array(pose_array)
    except rospy.ROSException:
        print("Timed out waiting for grasp candidates. Exiting.")
        return

    tester.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
