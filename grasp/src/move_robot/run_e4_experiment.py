#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 runner: top-down (Van den Bent's 4-DOF) vs tilted (our 6/7-DOF) grasp generator on
the real robot, in three conditions (structured crate / unstructured crate / no crate).

The unit of replication is a crate run: fill the crate as the condition prescribes, pick
until the stopping rule (--max-consecutive-failures failed attempts in a row, or the
operator says the crate is empty), record how much was cleared. Both methods use the same
hardware, planner (--planner, RRTConnect from E2), motion ('ours') and crate-contents carve
mode (--carve-mode); only simple_pick_point's ~grasp_generator differs.

  schedule  Writes experiments/e4/schedule.csv: --replicates runs per method x condition,
            conditions in blocks (rebuilding a setup is slow), methods alternating within a
            block in counterbalanced ABBA order, so drift over the day (truss wilting,
            lighting) doesn't favour one method.
  run       Runs the next unfinished run of the schedule (or --run-id). Per attempt the
            operator clicks the grasp point as usual (simple_pick_point), the truss is
            placed (drop_off), and the operator answers: was the truss removed from the
            crate, and did neighbouring trusses move. Logs to experiments/e4/attempts.csv
            (one row per attempt, incl. simple_pick_point's ~last_attempt: whether the
            generator found a reachable orientation = the E3 prediction, and its tilt)
            and experiments/e4/runs.csv (one row per run, clearance fraction).
            Planning/execution metrics are in experiments/planning_metrics/metrics.csv
            (join on timestamp).

Usage:
    rosrun grasp run_e4_experiment.py schedule --replicates 3 --seed 0
    rosrun grasp run_e4_experiment.py run                    # next run of the schedule
    rosrun grasp run_e4_experiment.py run --run-id 4         # redo/choose a specific run
"""
import argparse
import csv
import os
import random
import time

import rospy
import rospkg

METHODS = ("topdown", "tilted")
CONDITIONS = ("structured", "unstructured", "no_crate")
SCHEDULE_FIELDS = ["run_id", "condition", "method", "replicate"]
ATTEMPT_FIELDS = ["timestamp", "run_id", "condition", "method", "replicate", "attempt", "predicted_reachable",
                  "tilt_x_deg", "tilt_y_deg", "flip", "x", "y", "z", "yaw_deg", "pick_result", "drop_off_result",
                  "removed", "disturbance", "consecutive_failures", "notes"]
RUN_FIELDS = ["run_id", "condition", "method", "replicate", "start", "end", "trusses_start", "removed",
              "clearance_fraction", "attempts", "stop_reason", "planner", "carve_mode", "finger_extension_m"]
DISTURBANCE = {"0": "none", "1": "moved", "2": "knocked_over_or_out"}
STOCK_TCP_Z_M = 0.1034  # hand -> TCP without extension (franka_robot.xacro)


def e4_dir():
    ws = os.path.dirname(os.path.dirname(os.path.dirname(rospkg.RosPack().get_path('grasp'))))
    d = os.path.join(ws, "experiments", "e4")
    os.makedirs(d, exist_ok=True)
    return d


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def ask(prompt, options):
    while True:
        answer = input(prompt).strip().lower()
        if answer in options:
            return answer
        print(f"  answer one of: {', '.join(options)}")


def cmd_schedule(args):
    path = os.path.join(e4_dir(), "schedule.csv")
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite it")
    rng = random.Random(args.seed)
    conditions = list(CONDITIONS)
    rng.shuffle(conditions)
    rows, run_id = [], 0
    for condition in conditions:
        first = rng.choice(METHODS)
        second = METHODS[1 - METHODS.index(first)]
        # ABBA ABBA ...: each method gets the early and the late slot equally often
        order = [first if (i % 4) in (0, 3) else second for i in range(2 * args.replicates)]
        count = {m: 0 for m in METHODS}
        for method in order:
            rows.append(dict(run_id=run_id, condition=condition, method=method, replicate=count[method]))
            count[method] += 1
            run_id += 1
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"  run {r['run_id']:>2}: {r['condition']:<12} {r['method']:<8} (replicate {r['replicate']})")
    print(f"Schedule: {path}")


def configure(robot, method, args):
    """Everything both methods share, plus the generator: set before every attempt, so a
    parameter changed by hand in between can't leak into the run."""
    ns = f"/{robot}"
    rospy.set_param(f"{ns}/simple_pick_point/grasp_generator", method)
    rospy.set_param(f"{ns}/plan_movement/motion_method", "ours")
    rospy.set_param(f"{ns}/plan_movement/pipeline_id", "ompl")
    rospy.set_param(f"{ns}/plan_movement/planner_id", args.planner)
    rospy.set_param(f"{ns}/plan_movement/avoid_crate_contents", args.condition_uses_crate)
    rospy.set_param(f"{ns}/plan_movement/crate_contents_carve_mode", args.carve_mode)


def loaded_finger_extension(robot):
    """Finger extension of the running robot_description: hand -> hand_tcp distance from TF
    minus the stock offset. finger_extension_m is only a launch argument, not a parameter."""
    import tf2_ros
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)
    t = buffer.lookup_transform(f"{robot}_hand", f"{robot}_hand_tcp", rospy.Time(0), rospy.Duration(5.0))
    v = t.transform.translation
    return (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5 - STOCK_TCP_Z_M


def cmd_run(args):
    from grasp.srv import pipeline_command
    d = e4_dir()
    schedule = read_csv(os.path.join(d, "schedule.csv"))
    if not schedule:
        raise SystemExit("No schedule yet: rosrun grasp run_e4_experiment.py schedule")
    done = {r["run_id"] for r in read_csv(os.path.join(d, "runs.csv"))}
    if args.run_id is not None:
        run = next((r for r in schedule if int(r["run_id"]) == args.run_id), None)
        if run is None:
            raise SystemExit(f"No run {args.run_id} in the schedule")
    else:
        run = next((r for r in schedule if r["run_id"] not in done), None)
        if run is None:
            raise SystemExit("All scheduled runs are done")
    method, condition = run["method"], run["condition"]
    args.condition_uses_crate = condition != "no_crate"

    rospy.init_node('run_e4_experiment', disable_signals=True)
    ns = f"/{args.robot_name}"
    rospy.wait_for_service(f"{ns}/move_robot")
    rospy.wait_for_service(f"{ns}/simple_pick_point")
    move_robot = rospy.ServiceProxy(f"{ns}/move_robot", pipeline_command)
    pick = rospy.ServiceProxy(f"{ns}/simple_pick_point", pipeline_command)
    finger = loaded_finger_extension(args.robot_name)
    if abs(finger - args.finger_extension_m) > 0.002:
        raise SystemExit(f"The loaded robot has a {finger * 100:.1f} cm finger extension (hand -> TCP from TF), "
                         f"not {args.finger_extension_m * 100:.1f} cm: relaunch launch_moveit.launch with "
                         f"finger_extension_m:={args.finger_extension_m} "
                         f"tcp_xyz:=\"0 0 {STOCK_TCP_Z_M + args.finger_extension_m:.4f}\"")

    print(f"\n=== Run {run['run_id']}: condition {condition}, method {method} (replicate {run['replicate']})"
          f"{'  [ALREADY DONE: this adds a second record]' if run['run_id'] in done else ''} ===")
    print(f"planner {args.planner}, carve mode {args.carve_mode}, stop after {args.max_consecutive_failures} "
          f"consecutive failures or when the crate is empty")
    print(f"Set up the {condition.replace('_', ' ')} scene as the protocol describes (photo it).")
    trusses = int(input("Number of trusses in the scene: ").strip())

    start = time.strftime("%Y-%m-%d %H:%M:%S")
    configure(args.robot_name, method, args)
    move_robot("go_to_saved_pose")
    attempt, removed, consecutive, stop_reason = 0, 0, 0, "operator"
    try:
        while True:
            if removed >= trusses:
                stop_reason = "empty"
                break
            if consecutive >= args.max_consecutive_failures:
                stop_reason = "consecutive_failures"
                break
            key = input(f"\n--- Attempt {attempt + 1} ({removed}/{trusses} removed, {consecutive} failures in a row)"
                        f" -- Enter to pick, 'e' if the operator judges the crate empty/unpickable, 'q' to abort: ")
            key = key.strip().lower()
            if key == 'q':
                stop_reason = "aborted"
                break
            if key == 'e':
                stop_reason = "operator_empty"
                break

            configure(args.robot_name, method, args)
            # Labels every move plan_movement logs to experiments/trajectories (replay_trajectory.py)
            rospy.set_param(f"{ns}/plan_movement/trajectory_log_tag",
                            f"e4 run {run['run_id']} attempt {attempt + 1}")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            pick_result = pick("grasp").success
            info = rospy.get_param(f"{ns}/simple_pick_point/last_attempt", {}) or {}
            drop = move_robot("drop_off").success if pick_result == "success" else ""
            if not info:
                print("No grasp point was chosen (window closed?): not counted as an attempt.")
                move_robot("go_to_saved_pose")
                continue

            ok = ask("Truss removed from the crate and placed? [y/n]: ", ("y", "n")) == "y"
            disturbance = DISTURBANCE[ask("Neighbouring trusses: 0 none, 1 moved, 2 knocked over/out: ",
                                          tuple(DISTURBANCE))]
            notes = input("Notes (failure cause etc., Enter for none): ").strip()
            attempt += 1
            removed += ok
            consecutive = 0 if ok else consecutive + 1
            append_csv(os.path.join(d, "attempts.csv"), ATTEMPT_FIELDS, dict(
                timestamp=timestamp, run_id=run["run_id"], condition=condition, method=method,
                replicate=run["replicate"], attempt=attempt, predicted_reachable=info.get("reachable", ""),
                tilt_x_deg=info.get("tilt_x_deg", ""), tilt_y_deg=info.get("tilt_y_deg", ""),
                flip=info.get("flip", ""), x=info.get("x", ""), y=info.get("y", ""), z=info.get("z", ""),
                yaw_deg=info.get("yaw_deg", ""), pick_result=pick_result, drop_off_result=drop, removed=ok,
                disturbance=disturbance, consecutive_failures=consecutive, notes=notes))
            move_robot("go_to_saved_pose")
    except KeyboardInterrupt:
        stop_reason = "aborted"
    rospy.set_param(f"{ns}/plan_movement/trajectory_log_tag", "")  # later moves aren't part of this run
    if stop_reason == "aborted" and input("Run aborted: still record it in runs.csv? [y/N] ").strip().lower() != 'y':
        print("Not recorded; its attempts stay in attempts.csv. Rerun it with --run-id.")
        return
    append_csv(os.path.join(d, "runs.csv"), RUN_FIELDS, dict(
        run_id=run["run_id"], condition=condition, method=method, replicate=run["replicate"], start=start,
        end=time.strftime("%Y-%m-%d %H:%M:%S"), trusses_start=trusses, removed=removed,
        clearance_fraction=round(removed / trusses, 3) if trusses else "", attempts=attempt,
        stop_reason=stop_reason, planner=args.planner, carve_mode=args.carve_mode,
        finger_extension_m=round(finger, 4)))
    print(f"\nRun {run['run_id']} done: {removed}/{trusses} removed in {attempt} attempts ({stop_reason}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('schedule', help="Write the counterbalanced run order")
    s.add_argument('--replicates', type=int, default=3, help="Crate runs per method x condition")
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--force', action='store_true')
    r = sub.add_parser('run', help="Run one crate run")
    r.add_argument('--robot-name', default='panda')
    r.add_argument('--run-id', type=int, default=None)
    r.add_argument('--planner', default='RRTConnect', help="E2's choice")
    r.add_argument('--carve-mode', default='gripper', choices=['cylinder', 'gripper'])
    r.add_argument('--max-consecutive-failures', type=int, default=3)
    r.add_argument('--finger-extension-m', type=float, default=0.10,
                   help="Expected finger extension beyond the stock fingertip; the run refuses to start otherwise")
    args = parser.parse_args(rospy.myargv()[1:])
    cmd_schedule(args) if args.cmd == 'schedule' else cmd_run(args)


if __name__ == '__main__':
    main()
