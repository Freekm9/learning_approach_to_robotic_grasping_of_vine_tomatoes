#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning-only comparison of Pink (pink_planner.py) against OMPL planners for
moves into the crate: from one fixed start configuration, plan to straight-down
pre-grasp poses on a grid over the crate interior, at several yaws, with every
planner, and log success / planning time / joint-space path length.

Nothing is executed (OMPL plans from a set start state, Pink integrates from the
given start joints), so this runs against any bringup with move_group and the crate
scene: MoveIt fake execution, Gazebo, or the real robot. Like plan_movement.py's
go_to_pose(allow_flip=True), a failed goal is retried once with the gripper rolled
180 degrees; both attempts are logged.

Usage:
    rosrun grasp test_pink_planner.py --robot-name panda --planners pink RRTConnect BiTRRT

    # plan all the way down to the grasp pose instead of pre-grasp (stress test
    # of Pink's local nature next to the walls), coarser grid:
    rosrun grasp test_pink_planner.py --target grasp --spacing-m 0.08

Writes experiments/pink_comparison/planning_only_<timestamp>.csv and prints a
per-planner summary."""
import argparse
import csv
import os
import sys
import time

import numpy as np
import rospy
import rospkg
import moveit_commander
from moveit_msgs.msg import RobotState

from common.util import flip_z_rotation_stamped_pose
from move_robot.pink_moveit import PinkMoveItPlanner
from move_robot.test_crate_reachability import (compute_crate_interior, build_straight_down_pose,
                                                PRE_GRASP_OFFSET_M)

# franka_example_controllers' "ready" pose; close to the SavePose overview pose above the crate
READY_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

FIELDS = ["planner", "x", "y", "z", "yaw_deg", "attempt", "success", "planning_time_s",
          "path_length", "num_waypoints", "pink_reason", "pink_min_clearance_mm"]


def joint_path_length(points):
    """Same metric as plan_movement.py's _joint_path_length."""
    return float(sum(np.linalg.norm(np.array(b) - np.array(a)) for a, b in zip(points[:-1], points[1:])))


def plan_once(planner_name, pink, move_group, goal, start_joints, args):
    row = {"planner": planner_name, "pink_reason": "", "pink_min_clearance_mm": ""}
    t0 = time.time()
    if planner_name == "pink":
        success, plan = pink.plan(goal, move_group, start_joint_values=start_joints)
        res = pink.last_result
        row["pink_reason"] = res.reason
        if np.isfinite(res.min_clearance_m):
            row["pink_min_clearance_mm"] = round(res.min_clearance_m * 1000, 1)
    else:
        start = RobotState()
        start.joint_state.name = move_group.get_active_joints()
        start.joint_state.position = start_joints
        start.is_diff = True
        move_group.set_start_state(start)
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id(planner_name)
        move_group.set_planning_time(args.planning_time)
        move_group.set_num_planning_attempts(args.num_planning_attempts)
        move_group.set_pose_target(goal)
        success, plan, _, _ = move_group.plan()
        move_group.clear_pose_targets()
    row["planning_time_s"] = round(time.time() - t0, 3)
    row["success"] = bool(success)
    if success:
        points = [p.positions for p in plan.joint_trajectory.points]
        row["path_length"] = round(joint_path_length(points), 4)
        row["num_waypoints"] = len(points)
    return row


def summarize(rows, planners):
    print(f"\n{'planner':<12} {'1st try':>8} {'w/ flip':>8} {'median t [s]':>13} {'median len [rad]':>17}")
    for planner in planners:
        goals = {}
        for r in rows:
            if r["planner"] == planner:
                goals.setdefault((r["x"], r["y"], r["yaw_deg"]), []).append(r)
        if not goals:
            continue
        first = np.mean([g[0]["success"] for g in goals.values()])
        any_ok = np.mean([any(r["success"] for r in g) for g in goals.values()])
        ok = [r for r in rows if r["planner"] == planner and r["success"]]
        t = np.median([r["planning_time_s"] for r in ok]) if ok else float('nan')
        length = np.median([r["path_length"] for r in ok]) if ok else float('nan')
        print(f"{planner:<12} {first:>8.0%} {any_ok:>8.0%} {t:>13.2f} {length:>17.2f}")
    reasons = {}
    for r in rows:
        if r["planner"] == "pink" and not r["success"]:
            reasons[r["pink_reason"]] = reasons.get(r["pink_reason"], 0) + 1
    if reasons:
        print(f"Pink failure reasons: {reasons}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--planners', nargs='+', default=['pink', 'RRTConnect'],
                        help="'pink' and/or ompl planner names")
    parser.add_argument('--target', choices=['pre_grasp', 'grasp'], default='pre_grasp',
                        help="Plan to the pre-grasp pose (what go_to_pose plans to in grasp(); the "
                             "descent is a Cartesian move) or all the way to the grasp pose")
    parser.add_argument('--floor-clearance-m', type=float, default=0.03,
                        help="Grasp height above the bottom of the crate walls (m)")
    parser.add_argument('--edge-margin-m', type=float, default=0.03,
                        help="Inset of the goal grid from the crate's interior walls (m)")
    parser.add_argument('--spacing-m', type=float, default=0.05, help="Goal grid spacing (m)")
    parser.add_argument('--yaw-step-deg', type=float, default=45.0,
                        help="Yaws 0, step, ... < 180 (the flip retry covers the other half)")
    parser.add_argument('--start-joints', type=float, nargs=7, default=READY_JOINTS)
    parser.add_argument('--planning-time', type=float, default=5.0, help="OMPL budget, as plan_movement.py")
    parser.add_argument('--num-planning-attempts', type=int, default=100, help="as plan_movement.py")
    parser.add_argument('--limit-goals', type=int, default=None, help="Only the first N goals (smoke test)")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('test_pink_planner')
    moveit_commander.roscpp_initialize(sys.argv)
    ns = f'/{args.robot_name}'
    robot = moveit_commander.RobotCommander(robot_description=f'{ns}/robot_description', ns=ns)
    scene = moveit_commander.PlanningSceneInterface(ns=ns)
    move_group = moveit_commander.MoveGroupCommander(f'{args.robot_name}_manipulator',
                                                     robot_description=f'{ns}/robot_description', ns=ns)
    move_group.set_end_effector_link(f'{args.robot_name}_hand_tcp')
    move_group.set_max_velocity_scaling_factor(0.3)
    move_group.set_max_acceleration_scaling_factor(0.3)
    pink = PinkMoveItPlanner(robot, scene, ns=ns + '/')
    planning_frame = move_group.get_planning_frame()

    x_lo, x_hi, y_lo, y_hi, grasp_z, walls = compute_crate_interior(
        args.robot_name, grid_res_m=0.005, occupancy_margin_m=0.10, floor_clearance_m=args.floor_clearance_m)
    goal_z = grasp_z + (PRE_GRASP_OFFSET_M if args.target == 'pre_grasp' else 0.0)
    xs = np.arange(x_lo + args.edge_margin_m, x_hi - args.edge_margin_m + 1e-9, args.spacing_m)
    ys = np.arange(y_lo + args.edge_margin_m, y_hi - args.edge_margin_m + 1e-9, args.spacing_m)
    yaws = np.arange(0, 180, args.yaw_step_deg)
    goals = [(x, y, yaw) for x in xs for y in ys for yaw in yaws][:args.limit_goals]
    print(f"Crate interior x=[{x_lo:.3f}, {x_hi:.3f}] y=[{y_lo:.3f}, {y_hi:.3f}]; {len(goals)} {args.target} "
          f"goals at z={goal_z:.3f} x {args.planners}")

    rospack = rospkg.RosPack()
    ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(rospack.get_path('grasp'))))
    out_dir = os.path.join(ws_dir, "experiments", "pink_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"planning_only_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    rows = []
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (x, y, yaw) in enumerate(goals):
            goal = build_straight_down_pose(x, y, goal_z, yaw, planning_frame)
            for planner_name in args.planners:
                for attempt, target in (("original", goal), ("flipped", flip_z_rotation_stamped_pose(goal))):
                    if rospy.is_shutdown():
                        return
                    row = plan_once(planner_name, pink, move_group, target, args.start_joints, args)
                    row.update(x=round(x, 3), y=round(y, 3), z=round(goal_z, 3), yaw_deg=yaw, attempt=attempt)
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    if row["success"]:
                        break
            print(f"[{i + 1}/{len(goals)}] ({x:.3f}, {y:.3f}) yaw {yaw:.0f}: " +
                  ", ".join(f"{p}={'ok' if any(r['success'] for r in rows if r['planner'] == p and r['x'] == round(x, 3) and r['y'] == round(y, 3) and r['yaw_deg'] == yaw) else 'FAIL'}"
                            for p in args.planners))

    summarize(rows, args.planners)
    print(f"\nLog: {out_path}")


if __name__ == '__main__':
    main()
