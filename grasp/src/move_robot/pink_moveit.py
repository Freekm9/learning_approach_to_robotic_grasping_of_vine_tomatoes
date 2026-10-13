#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoveIt glue for pink_planner.PinkPlanner: builds the Pink model from the
robot_description and the planning scene's crate boxes, plans from a move group's
current state to a PoseStamped goal, and checks the result against MoveIt's full
planning scene. Used by plan_movement.py (`~pipeline_id: pink`) and
test_pink_planner.py, so both run exactly the same planner."""

import os
import time

import numpy as np
import rospy
import rospkg
import moveit_msgs.msg
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from trajectory_msgs.msg import JointTrajectoryPoint

from common.transforms import transform_pose
from common.crate_contents import CRATE_CONTENTS_ID
from move_robot.pink_planner import PinkPlanner, pin
from move_robot.test_crate_reachability import box_world_pose_and_dims


class PinkMoveItPlanner(object):
    """robot/scene: moveit_commander RobotCommander / PlanningSceneInterface.
    tf_buffer: only needed for goals not already in the URDF root frame. ns: prefix
    for robot_description/check_state_validity, e.g. '/panda/' for a node that isn't
    itself started in the robot's namespace (default: resolve relative to the node)."""

    def __init__(self, robot, scene, tf_buffer=None, ns=''):
        self.robot = robot
        self.ns = ns
        self.scene = scene
        self.tf_buffer = tf_buffer
        self._planner = None
        self._key = None
        self._validity_service = None
        self.last_result = None  # PinkPlanResult of the last plan() call, for logging

    def get_planner(self, move_group):
        """Pink model of the robot plus the planning scene's boxes, built on first use
        and rebuilt whenever the scene's boxes or the finger opening change (the fingers
        are locked in the Pink model, at their current opening)."""
        boxes = []
        for name, obj in sorted(self.scene.get_objects().items()):
            if name == CRATE_CONTENTS_ID:
                # Hundreds of boxes would make every Pink step slow; the MoveIt check of
                # the finished path (first_invalid_waypoint) still rejects paths through them.
                continue
            box = box_world_pose_and_dims(obj)
            if box is None:
                rospy.logwarn_once(f"Pink: skipping non-box planning-scene object '{name}'")
                continue
            center, rot, dims = box
            boxes.append((name, center, rot.as_matrix(), dims))

        state = self.robot.get_current_state().joint_state
        fingers = {n: p for n, p in zip(state.name, state.position) if 'finger_joint' in n}
        arm_joints = move_group.get_active_joints()
        key = (tuple(arm_joints),
               tuple((n, tuple(np.round(c, 3)), tuple(np.round(d, 3))) for n, c, _, d in boxes),
               tuple(sorted((n, round(p, 3)) for n, p in fingers.items())))
        if self._key != key:
            urdf = rospy.get_param(self.ns + 'robot_description')
            franka_description_dir = rospkg.RosPack().get_path('franka_description')
            build_start = time.time()
            self._planner = PinkPlanner(
                urdf, [os.path.dirname(franka_description_dir)], arm_joints, boxes,
                locked_joint_values=fingers,
                collision_geometry=rospy.get_param('~pink_collision_geometry', 'capsules'))
            self._key = key
            rospy.loginfo(f"Pink model built in {time.time() - build_start:.2f}s with "
                          f"{len(self._planner.geom_model.collisionPairs)} robot-vs-scene pairs "
                          f"({[b[0] for b in boxes]})")
        return self._planner

    def plan(self, goal_pose, move_group, start_joint_values=None):
        """Pink counterpart of move_group.plan(): differential IK from the current (or
        given) joint state to goal_pose (PoseStamped for move_group's end-effector link),
        then checked waypoint-by-waypoint against MoveIt's full planning scene
        (self-collisions included, which Pink doesn't model). Returns (success, plan),
        plan an untimed moveit_msgs RobotTrajectory (retime it before executing), or
        None on failure."""
        planner = self.get_planner(move_group)
        root_frame = planner.model.frames[1].name  # first frame after "universe" is the URDF root
        if goal_pose.header.frame_id.lstrip('/') != root_frame:
            goal_pose = transform_pose(goal_pose, root_frame, self.tf_buffer)
        p, o = goal_pose.pose.position, goal_pose.pose.orientation
        goal_se3 = pin.SE3(pin.Quaternion(o.w, o.x, o.y, o.z).normalized().toRotationMatrix(),
                           np.array([p.x, p.y, p.z]))

        if start_joint_values is None:
            start_joint_values = move_group.get_current_joint_values()
        result = planner.plan(start_joint_values, move_group.get_end_effector_link(), goal_se3,
                              d_min=rospy.get_param('~pink_d_min', 0.015),
                              max_time_s=rospy.get_param('~pink_max_time_s', 20.0))
        self.last_result = result
        print(f"PINK: {result.reason} after {result.iterations} steps / {result.planning_time_s:.2f}s, "
              f"error {result.position_error_m * 1000:.1f} mm / {np.rad2deg(result.orientation_error_rad):.1f} deg, "
              f"min clearance to scene {result.min_clearance_m * 1000:.1f} mm")
        if not result.success:
            if result.reason == "stalled":
                link, obstacle, distance = planner.describe_closest_pair(result.waypoints[-1])
                print(f"PINK stalled with {link} {distance * 1000:.1f} mm from {obstacle}")
            return False, None

        joint_names = move_group.get_active_joints()
        invalid = self.first_invalid_waypoint(move_group, joint_names, result.waypoints)
        if invalid is not None:
            result.success = False
            result.reason = "moveit_collision"
            print(f"PINK path rejected by MoveIt collision check: {invalid}")
            return False, None

        plan = moveit_msgs.msg.RobotTrajectory()
        plan.joint_trajectory.joint_names = joint_names
        plan.joint_trajectory.points = [JointTrajectoryPoint(positions=q) for q in result.waypoints]
        return True, plan

    def first_invalid_waypoint(self, move_group, joint_names, waypoints):
        """None if every waypoint is collision-free in MoveIt's planning scene, else a
        description of the contacts at the first one that isn't."""
        if self._validity_service is None:
            rospy.wait_for_service(self.ns + 'check_state_validity', timeout=30)
            self._validity_service = rospy.ServiceProxy(self.ns + 'check_state_validity', GetStateValidity)
        req = GetStateValidityRequest()
        req.group_name = move_group.get_name()
        req.robot_state = RobotState()
        req.robot_state.is_diff = True  # fingers etc. stay as they are now
        req.robot_state.joint_state.name = joint_names
        for q in waypoints:
            req.robot_state.joint_state.position = q
            res = self._validity_service(req)
            if not res.valid:
                contacts = sorted({f"{c.contact_body_1}-{c.contact_body_2}" for c in res.contacts})
                return contacts or "invalid state"
        return None
