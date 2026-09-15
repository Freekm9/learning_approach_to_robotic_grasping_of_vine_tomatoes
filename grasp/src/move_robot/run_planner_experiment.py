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

    # chomp is a whole pipeline, not an ompl planner name -- mix it in freely:
    rosrun grasp run_planner_experiment.py --robot-name panda \\
        --planners RRTConnect chomp --trials-per-planner 15
"""
import argparse
import csv
import os
import random
import time

import rospy
import rospkg

from grasp.srv import pipeline_command


def get_log_path():
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    log_dir = os.path.join(catkin_ws_dir, "experiments", "planner_comparison")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "trial_log.csv")


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


def reset_arm(move_robot):
    move_robot("open_gripper")
    move_robot("go_to_saved_pose")


# Planning pipelines that have no ompl-style named planner_id (see
# plan_movement.py's pipeline_id/planner_id properties) -- entries here select
# the whole pipeline instead of an ompl planner.
PIPELINE_ONLY_CONFIGS = {'chomp'}


def apply_planning_config(robot_name, config):
    """Point plan_movement.py at `config` for the next trial. `config` is either
    an ompl planner name (e.g. 'RRTConnect') or a pipeline name from
    PIPELINE_ONLY_CONFIGS (e.g. 'chomp')."""
    pipeline_param = f'/{robot_name}/plan_movement/pipeline_id'
    planner_param = f'/{robot_name}/plan_movement/planner_id'
    if config in PIPELINE_ONLY_CONFIGS:
        rospy.set_param(pipeline_param, config)
    else:
        rospy.set_param(pipeline_param, 'ompl')
        rospy.set_param(planner_param, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--planners', nargs='+', default=['RRTConnect', 'BiTRRT', 'RRTstar'],
                         help="ompl planner names and/or pipeline names from PIPELINE_ONLY_CONFIGS (e.g. chomp)")
    parser.add_argument('--trials-per-planner', type=int, default=15)
    parser.add_argument('--seed', type=int, default=None, help="Fixed seed for a reproducible trial order")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('run_planner_experiment')

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

    log_path = get_log_path()
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "block", "trial_in_run", "planner_id", "service_result"])
        if write_header:
            writer.writeheader()

        rospy.loginfo("Resetting arm to saved pose before first trial...")
        reset_arm(move_robot)

        for i, (block, planner) in enumerate(schedule):
            print(f"\n--- Trial {i + 1}/{total} (block {block + 1}/{args.trials_per_planner}) -- planner: {planner} ---")
            response = input("Ready (reposition truss / check clearance now if needed) -- press Enter to run, 's' to skip, 'q' to stop: ").strip().lower()
            if response == 'q':
                print("Stopped early.")
                break
            if response == 's':
                continue

            apply_planning_config(args.robot_name, planner)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            result = simple_pick_point("grasp")
            writer.writerow({
                "timestamp": timestamp,
                "block": block,
                "trial_in_run": i,
                "planner_id": planner,
                "service_result": result.success,
            })
            f.flush()
            print(f"Result: {result.success}")

            rospy.loginfo("Resetting arm for next trial...")
            reset_arm(move_robot)

    print(f"\nDone. Trial log: {log_path}")
    print("Per-attempt planning/execution metrics: experiments/planning_metrics/metrics.csv")


if __name__ == '__main__':
    main()
