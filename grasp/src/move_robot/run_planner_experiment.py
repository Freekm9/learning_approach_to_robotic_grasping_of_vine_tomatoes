#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated runner for the planner-comparison experiment (see
experiments/planner_comparison_protocol.md for the full procedure).

Switches /<robot_name>/plan_movement/planner_id via rosparam between trials
-- no node restart needed, since Planner.planner_id re-reads it on every
access -- and triggers one grasp attempt per trial through the existing
simple_pick_point service. The human still clicks the grasp point in the
popped-up window each trial and confirms via Enter before each attempt
runs (repositioning the truss, checking clearance, etc.); everything else
(planner switching, trial sequencing, resetting the arm between attempts,
logging which planner produced which run) is automatic.

Per-attempt metrics (planning success/time, path length, execution
success, goal_reached, side + gripper video files) are already logged by
plan_movement.py to experiments/planning_metrics/metrics.csv, one row per
grasp() call.
This script writes a companion trial log (block/trial/planner/timestamp)
into experiments/planner_comparison/trial_log.csv so trials can be joined
back to their metrics.csv row by timestamp, and shuffles planner order
within each block (rather than running all trials of one planner, then
the next) so time-based drift -- truss wilting, lighting changes -- doesn't
confound the comparison between planners.

Usage:
    rosrun grasp run_planner_experiment.py --robot-name panda \\
        --planners RRTConnect BiTRRT RRTstar --trials-per-planner 15

    # chomp and pink are whole pipelines, not ompl planner names -- mix them in freely:
    rosrun grasp run_planner_experiment.py --robot-name panda \\
        --planners RRTConnect chomp pink --trials-per-planner 15

    # 'luuk' is Luuk van den Bent's original grasp/place motion (plan_movement's
    # motion_method 'luuk', position controller only).
    # Our method vs his, with the truss also placed at the SavePosePlace pose:
    rosrun grasp run_planner_experiment.py --robot-name panda \\
        --planners BiTRRT luuk --trials-per-planner 15 --drop-off --log luuk_comparison.csv

    # all three motion styles plus STOMP (launch_moveit.launch stomp:=true):
    rosrun grasp run_planner_experiment.py --robot-name panda \\
        --planners BiTRRT stomp waypoints luuk --trials-per-planner 15 --drop-off --log methods.csv

The arm is always reset with our motion, so every trial starts the same way.
"""
import argparse
import csv
import os
import random
import time

import rospy
import rospkg

from grasp.srv import pipeline_command


def get_log_path(name="trial_log.csv"):
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    log_dir = os.path.join(catkin_ws_dir, "experiments", "planner_comparison")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, name)


def build_schedule(planners, trials_per_planner, rng):
    """One block per trial index: each block is a shuffled full pass over
    `planners`, so trials interleave planners instead of clustering them."""
    schedule = []
    for block in range(trials_per_planner):
        block_planners = list(planners)
        rng.shuffle(block_planners)
        for planner in block_planners:
            schedule.append((block, planner))
    return schedule


def reset_arm(move_robot, robot_name, planner):
    # Same configuration for every reset, whatever the last trial used
    apply_planning_config(robot_name, planner, planner)
    move_robot("open_gripper")
    move_robot("go_to_saved_pose")


# Planning pipelines that have no ompl-style named planner_id (see
# plan_movement.py's pipeline_id/planner_id properties) -- entries here select
# the whole pipeline instead of an ompl planner.
PIPELINE_ONLY_CONFIGS = {'chomp', 'pink', 'stomp'}
# Whole motion methods instead of a planner (see plan_movement.py's motion_method).
MOTION_METHOD_CONFIGS = {'luuk', 'waypoints'}


def apply_planning_config(robot_name, config, fallback_planner):
    """Point plan_movement.py at `config` for the next trial. `config` is either
    an ompl planner name (e.g. 'RRTConnect'), a pipeline name from
    PIPELINE_ONLY_CONFIGS (e.g. 'chomp') or a motion method from MOTION_METHOD_CONFIGS.
    Motion methods get ompl with fallback_planner for any move they plan ('waypoints'
    plans a blocked transit or approach; 'luuk' plans nothing with it)."""
    pipeline_param = f'/{robot_name}/plan_movement/pipeline_id'
    planner_param = f'/{robot_name}/plan_movement/planner_id'
    rospy.set_param(f'/{robot_name}/plan_movement/motion_method', config if config in MOTION_METHOD_CONFIGS else 'ours')
    if config in MOTION_METHOD_CONFIGS:
        rospy.set_param(pipeline_param, 'ompl')
        rospy.set_param(planner_param, fallback_planner)
        return
    if config in PIPELINE_ONLY_CONFIGS:
        rospy.set_param(pipeline_param, config)
    else:
        rospy.set_param(pipeline_param, 'ompl')
        rospy.set_param(planner_param, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--planners', nargs='+', default=['RRTConnect', 'BiTRRT', 'RRTstar'],
                         help="ompl planner names, pipeline names from PIPELINE_ONLY_CONFIGS (e.g. chomp) "
                              "and/or motion methods 'luuk' (Luuk van den Bent's motion) and 'waypoints' "
                              "(straight transit + approach); 'stomp' needs launch_moveit.launch stomp:=true")
    parser.add_argument('--fallback-planner', default='BiTRRT',
                        help="ompl planner for the arm resets between trials and for moves that 'waypoints' "
                             "has to plan (default BiTRRT)")
    parser.add_argument('--drop-off', action='store_true',
                        help="After a successful grasp, also place the truss at the SavePosePlace pose (drop_off) "
                             "with the same method; its result is logged as drop_off_result")
    parser.add_argument('--trials-per-planner', type=int, default=15)
    parser.add_argument('--log', default='trial_log.csv',
                        help="Trial log file name in experiments/planner_comparison/")
    parser.add_argument('--seed', type=int, default=None, help="Fixed seed for a reproducible trial order")
    args = parser.parse_args(rospy.myargv()[1:])

    # disable_signals: Ctrl-C raises KeyboardInterrupt, so it also aborts a blocking service call
    rospy.init_node('run_planner_experiment', disable_signals=True)

    move_robot_srv_name = f'/{args.robot_name}/move_robot'
    pick_srv_name = f'/{args.robot_name}/simple_pick_point'
    rospy.loginfo(f"Waiting for services {move_robot_srv_name}, {pick_srv_name} ...")
    rospy.wait_for_service(move_robot_srv_name)
    rospy.wait_for_service(pick_srv_name)
    move_robot = rospy.ServiceProxy(move_robot_srv_name, pipeline_command)
    simple_pick_point = rospy.ServiceProxy(pick_srv_name, pipeline_command)

    rng = random.Random(args.seed)
    schedule = build_schedule(args.planners, args.trials_per_planner, rng)

    total = len(schedule)
    print(f"\n{total} trials queued: {args.planners} x {args.trials_per_planner} each, "
          f"shuffled in {args.trials_per_planner} blocks.")
    if input("Start? [y/N] ").strip().lower() != 'y':
        print("Aborted.")
        return

    log_path = get_log_path(args.log)
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        fields = ["timestamp", "block", "trial_in_run", "planner_id", "service_result", "drop_off_result"]
        if not write_header:
            with open(log_path, newline='') as existing:
                header = next(csv.reader(existing), [])
            if "drop_off_result" not in header:
                fields.remove("drop_off_result")  # older log: keep its columns
                if args.drop_off:
                    raise SystemExit(f"{log_path} predates drop_off_result; pass --log <new name> for --drop-off runs")
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        rospy.loginfo("Resetting arm to saved pose before first trial...")
        reset_arm(move_robot, args.robot_name, args.fallback_planner)

        for i, (block, planner) in enumerate(schedule):
            print(f"\n--- Trial {i + 1}/{total} (block {block + 1}/{args.trials_per_planner}) -- planner: {planner} ---")
            response = input("Ready (reposition truss / check clearance now if needed) -- press Enter to run, 's' to skip, 'q' to stop: ").strip().lower()
            if response == 'q':
                print("Stopped early.")
                break
            if response == 's':
                continue

            apply_planning_config(args.robot_name, planner, args.fallback_planner)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = simple_pick_point("grasp")
            row = {
                "timestamp": timestamp,
                "block": block,
                "trial_in_run": i,
                "planner_id": planner,
                "service_result": result.success,
            }
            if args.drop_off:
                row["drop_off_result"] = move_robot("drop_off").success if result.success == "success" else ""
            writer.writerow(row)
            f.flush()
            print(f"Result: {result.success}" + (f", drop-off: {row['drop_off_result']}" if args.drop_off else ""))

            rospy.loginfo("Resetting arm for next trial...")
            reset_arm(move_robot, args.robot_name, args.fallback_planner)

    print(f"\nDone. Trial log: {log_path}")
    print("Per-attempt planning/execution metrics: experiments/planning_metrics/metrics.csv")


if __name__ == '__main__':
    main()
