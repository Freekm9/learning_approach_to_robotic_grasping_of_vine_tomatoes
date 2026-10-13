#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize a planner-comparison experiment run by run_planner_experiment.py.

Joins the two logs it produces:
  - experiments/planner_comparison/trial_log.csv: one row per trial, with the
    simple_pick_point service's overall success/failure for that pick.
  - experiments/planning_metrics/metrics.csv: one row per go_to_pose() call
    (each trial makes two: "grasp_pre_grasp" then "grasp_post_grasp"), with
    per-movement planning/execution detail.

Trial-level success/failure answers "did the pick work"; the per-movement
metrics answer "why" (planning time, path length, waypoint count) broken
down by planner and by which half of the grasp motion it was.

Usage:
    rosrun grasp summarize_planner_experiment.py
    rosrun grasp summarize_planner_experiment.py --since "2026-08-21 00:00:00"
    rosrun grasp summarize_planner_experiment.py --log luuk_comparison.csv
"""
import argparse
import os

import pandas as pd
import rospkg


def get_experiment_dirs(log_name="trial_log.csv"):
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    experiments_dir = os.path.join(catkin_ws_dir, "experiments")
    return (
        os.path.join(experiments_dir, "planner_comparison", log_name),
        os.path.join(experiments_dir, "planning_metrics", "metrics.csv"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', default=None, help="Only include rows at/after this timestamp, e.g. '2026-08-21 00:00:00'")
    parser.add_argument('--log', default='trial_log.csv', help="Trial log file name in experiments/planner_comparison/")
    args = parser.parse_args()

    trial_log_path, metrics_path = get_experiment_dirs(args.log)
    trials = pd.read_csv(trial_log_path, parse_dates=["timestamp"])
    metrics = pd.read_csv(metrics_path, parse_dates=["timestamp"])

    if args.since:
        since = pd.Timestamp(args.since)
        trials = trials[trials["timestamp"] >= since]
        metrics = metrics[metrics["timestamp"] >= since]

    print(f"Trial log: {trial_log_path} ({len(trials)} trials)")
    print(f"Metrics log: {metrics_path} ({len(metrics)} rows)\n")

    print("=== Trial-level pick success rate, by planner ===")
    trial_summary = trials.groupby("planner_id").agg(
        n_trials=("service_result", "count"),
        n_success=("service_result", lambda s: (s == "success").sum()),
    )
    trial_summary["success_rate"] = (trial_summary["n_success"] / trial_summary["n_trials"]).round(3)
    if "drop_off_result" in trials:
        placed = trials[trials["drop_off_result"].notna()]
        if len(placed):
            trial_summary["n_placed"] = placed.groupby("planner_id")["drop_off_result"].apply(lambda s: (s == "success").sum())
            trial_summary["n_placed"] = trial_summary["n_placed"].fillna(0).astype(int)
    print(trial_summary.to_string())

    print("\n=== Per-movement planning/execution detail, by planner ===")
    # grasp_descent: only 'luuk' logs it (impedance move); ours is a straight line, not logged
    # grasp_pre_grasp_transit/_approach: the two straight segments of 'waypoints'
    grasp_moves = metrics[metrics["movement"].isin(["grasp_pre_grasp", "grasp_pre_grasp_transit", "grasp_pre_grasp_approach",
                                                    "grasp_descent", "grasp_post_grasp",
                                                    "go_to_place_above", "go_to_place", "go_to_place_retreat"])]
    move_summary = grasp_moves.groupby(["planner_id", "movement"]).agg(
        n=("planning_success", "count"),
        planning_success_rate=("planning_success", "mean"),
        mean_planning_duration_s=("planning_duration_s", "mean"),
        mean_path_length=("path_length", "mean"),
        mean_num_waypoints=("num_waypoints", "mean"),
        mean_execution_duration_s=("execution_duration_s", "mean"),
        goal_reached_rate=("goal_reached", "mean"),
    ).round(3)
    print(move_summary.to_string())


if __name__ == '__main__':
    main()
