#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List, show and replay trajectories logged by plan_movement.py (common/trajectory_log.py:
experiments/trajectories/trajectories_<YYYYMMDD>.jsonl.gz).

  list    one line per logged move: index, time, movement, kind, duration, result, tag
  show    publish one move to RViz (DisplayTrajectory on /replay_trajectory/display_path):
          the planned trajectory, or with --source executed the measured joint states
  replay  execute one move again on the robot (or a sim):
            - the robot must be at the move's start configuration (max joint difference
              --start-tolerance); --go-to-start plans there first (shown and confirmed)
            - --with-scene adds the move's logged crate contents to the planning scene
            - every waypoint is collision-checked against the current planning scene; a
              collision aborts unless --force
            - runs at --speed (default 0.5: half the logged speed; times stretched, velocities
              and accelerations scaled to match) and asks for confirmation (--yes skips it)
            - the replay itself is logged too (kind 'replay', with the source record)
          --source planned (default) replays the trajectory that was sent to the controller;
          --source executed replays the measured joint states (retimed), also for 'go'
          moves that have no planned trajectory.

Usage (same ROS master as the robot / sim):
    rosrun grasp replay_trajectory.py list [--file F] [--movement grasp_pre_grasp]
    rosrun grasp replay_trajectory.py show 12
    rosrun grasp replay_trajectory.py replay 12 --speed 0.3 --with-scene
Default file: the newest log in experiments/trajectories.
"""
import argparse
import glob
import os
import sys

import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def ws_dir():
    import rospkg
    return os.path.dirname(os.path.dirname(os.path.dirname(rospkg.RosPack().get_path('grasp'))))


def log_file(args):
    if args.file:
        return args.file
    files = sorted(glob.glob(os.path.join(ws_dir(), 'experiments', 'trajectories', 'trajectories_*.jsonl.gz')))
    if not files:
        raise SystemExit("No trajectory logs in experiments/trajectories")
    return files[-1]


def duration(rec):
    if rec.get("trajectory") and rec["trajectory"]["time_from_start"]:
        return rec["trajectory"]["time_from_start"][-1]
    t = rec.get("executed", {}).get("t") or [0.0]
    return t[-1]


def arm_path(rec, source, robot_name):
    """(joint names, positions) of a record's arm motion: the planned trajectory or the
    measured joint states (arm joints only, in the arm's joint order)."""
    if source == "planned":
        if not rec.get("trajectory"):
            raise SystemExit(f"Move is a '{rec['kind']}' move without a planned trajectory: use --source executed")
        return rec["trajectory"]["joint_names"], np.array(rec["trajectory"]["positions"])
    ex = rec.get("executed") or {}
    if not ex.get("positions"):
        raise SystemExit("No executed joint states in this record")
    names = [f"{robot_name}_{j}" for j in ARM_JOINTS]
    idx = [ex["names"].index(n) for n in names]
    return names, np.array(ex["positions"])[:, idx]


def cmd_list(args):
    from common.trajectory_log import read_log
    path = log_file(args)
    records = read_log(path)
    print(f"{path}: {len(records)} moves")
    print(f"{'#':>4}  {'time':19}  {'movement':24} {'kind':9} {'planner':11} {'dur s':>6} {'pts':>4} {'exec':>5}  "
          f"{'ok':5} {'scene':5} tag")
    for i, r in enumerate(records):
        if args.movement and r.get("movement") != args.movement:
            continue
        n = len(r["trajectory"]["positions"]) if r.get("trajectory") else 0
        print(f"{i:4d}  {r.get('time', ''):19}  {r.get('movement', '')[:24]:24} {r.get('kind', ''):9} "
              f"{str(r.get('planner', ''))[:11]:11} {duration(r):6.2f} {n:4d} {len(r.get('executed', {}).get('t', [])):5d}  "
              f"{str(r.get('success')):5} {'yes' if r.get('scene') else '':5} {r.get('tag', '')}")


class Robot(object):
    def __init__(self, robot_name):
        import rospy
        import moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)
        self.ns = f'/{robot_name}'
        self.robot_name = robot_name
        self.robot = moveit_commander.RobotCommander(robot_description=f'{self.ns}/robot_description', ns=self.ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns=self.ns)
        self.mg = moveit_commander.MoveGroupCommander(f'{robot_name}_manipulator',
                                                      robot_description=f'{self.ns}/robot_description', ns=self.ns)
        self.frame = self.mg.get_planning_frame()
        rospy.sleep(1.0)


def build_trajectory(robot, rec, args):
    """RobotTrajectory to execute: the logged one slowed down to --speed, or the measured
    joint states retimed at --speed x the pipeline's 0.3 scaling."""
    from common.trajectory_log import dict_to_trajectory
    from moveit_msgs.msg import RobotState, RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    names, q = arm_path(rec, args.source, robot.robot_name)
    if args.source == "planned":
        return names, q, dict_to_trajectory(rec["trajectory"], speed=args.speed)
    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(names)
    keep = [0] + [i for i in range(1, len(q)) if np.max(np.abs(q[i] - q[i - 1])) > 1e-4]
    traj.joint_trajectory.points = [JointTrajectoryPoint(positions=list(q[i])) for i in keep]
    ref = RobotState()
    ref.joint_state.name, ref.joint_state.position = list(names), list(q[0])
    traj = robot.mg.retime_trajectory(ref, traj, velocity_scaling_factor=0.3 * args.speed,
                                      acceleration_scaling_factor=0.3 * args.speed)
    return names, q[keep], traj


def check_validity(robot, names, q, step=0.01):
    """Index of the first colliding configuration along the path (linearly interpolated
    in steps of at most `step` rad), or None."""
    import rospy
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
    srv = rospy.ServiceProxy(f'{robot.ns}/check_state_validity', GetStateValidity)
    srv.wait_for_service(5.0)
    req = GetStateValidityRequest()
    req.group_name = robot.mg.get_name()
    current = robot.robot.get_current_state()
    for k in range(len(q)):
        segment = [q[k]] if k == 0 else [q[k - 1] + (q[k] - q[k - 1]) * s / n
                                         for n in [max(int(np.ceil(np.max(np.abs(q[k] - q[k - 1])) / step)), 1)]
                                         for s in range(1, n + 1)]
        for c in segment:
            state = RobotState()
            state.joint_state.name = list(current.joint_state.name)
            pos = dict(zip(current.joint_state.name, current.joint_state.position))
            pos.update(dict(zip(names, c)))
            state.joint_state.position = [pos[n] for n in state.joint_state.name]
            req.robot_state = state
            res = srv(req)
            if not res.valid:
                contacts = sorted({f"{c.contact_body_1}-{c.contact_body_2}" for c in res.contacts})
                return k, contacts
    return None


def confirm(prompt, args):
    return args.yes or input(f"{prompt} [y/N] ").strip().lower() == 'y'


def cmd_show(args):
    import rospy
    from moveit_msgs.msg import DisplayTrajectory, RobotState
    from common.trajectory_log import read_log
    rospy.init_node('replay_trajectory_show')
    rec = read_log(log_file(args))[args.index]
    robot = Robot(args.robot_name)
    names, q, traj = build_trajectory(robot, rec, args)
    start = RobotState()
    start.joint_state.name, start.joint_state.position = list(names), list(q[0])
    pub = rospy.Publisher('/replay_trajectory/display_path', DisplayTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.5)
    pub.publish(DisplayTrajectory(trajectory_start=start, trajectory=[traj]))
    print(f"#{args.index} {rec.get('movement')} ({rec.get('kind')}, {args.source}): published on "
          f"/replay_trajectory/display_path (RViz: Trajectory display). Ctrl-C to quit.")
    rospy.spin()


def cmd_replay(args):
    import rospy
    from common.crate_contents import columns_to_collision_object, CRATE_CONTENTS_ID
    from common.trajectory_log import TrajectoryLog, read_log
    rospy.init_node('replay_trajectory')
    path = log_file(args)
    rec = read_log(path)[args.index]
    if not 0 < args.speed <= 1:
        raise SystemExit("--speed must be in (0, 1]")
    robot = Robot(args.robot_name)
    names, q, traj = build_trajectory(robot, rec, args)
    print(f"#{args.index} {rec.get('time')} {rec.get('movement')} ({rec.get('kind')}, planner {rec.get('planner')}), "
          f"source {args.source}: {len(q)} points, {traj.joint_trajectory.points[-1].time_from_start.to_sec():.1f} s "
          f"at speed {args.speed}")

    current = dict(zip(robot.mg.get_active_joints(), robot.mg.get_current_joint_values()))
    offset = max(abs(current[n] - v) for n, v in zip(names, q[0]))
    if offset > args.start_tolerance:
        print(f"Robot is {offset:.3f} rad (max joint) from the move's start configuration.")
        if not args.go_to_start:
            raise SystemExit("Move it there first, or pass --go-to-start")
        robot.mg.set_max_velocity_scaling_factor(0.3 * args.speed)
        robot.mg.set_max_acceleration_scaling_factor(0.3 * args.speed)
        robot.mg.set_joint_value_target(dict(zip(names, q[0])))
        ok, plan, _, _ = robot.mg.plan()
        if not ok:
            raise SystemExit("No plan to the start configuration")
        if not confirm(f"Move to the start configuration ({plan.joint_trajectory.points[-1].time_from_start.to_sec():.1f} s)?",
                       args):
            raise SystemExit("Aborted")
        robot.mg.execute(plan, wait=True)
        robot.mg.stop()

    added = False
    try:
        if args.with_scene:
            if not rec.get("scene"):
                print("This move has no logged crate contents; replaying against the current scene only")
            else:
                s = rec["scene"]
                robot.scene.add_object(columns_to_collision_object(s["columns"], s["floor_z"], s["cell_m"], robot.frame))
                rospy.sleep(1.0)
                added = True
        bad = check_validity(robot, names, q)
        if bad is not None:
            print(f"Collision at waypoint {bad[0]} of {len(q)} against the current planning scene: {bad[1]}")
            if not args.force:
                raise SystemExit("Not replaying (use --force to replay anyway)")
        else:
            print("Collision check against the current planning scene: OK")
        if not confirm(f"Execute the move at speed {args.speed}?", args):
            raise SystemExit("Aborted")
        log = TrajectoryLog(os.path.join(ws_dir(), 'experiments', 'trajectories'),
                            joint_states_topic=f'{robot.ns}/joint_states')
        rospy.sleep(0.3)
        with log.recording(rec.get("movement"), "replay", plan=traj, planner=rec.get("planner"),
                           source=dict(file=os.path.basename(path), index=args.index, of=args.source,
                                       speed=args.speed)) as lrec:
            lrec["success"] = robot.mg.execute(traj, wait=True)
        robot.mg.stop()
        end = dict(zip(robot.mg.get_active_joints(), robot.mg.get_current_joint_values()))
        err = max(abs(end[n] - v) for n, v in zip(names, q[-1]))
        print(f"Executed: {lrec['success']}; final max joint error {err:.4f} rad")
    finally:
        if added and CRATE_CONTENTS_ID in robot.scene.get_known_object_names():
            robot.scene.remove_world_object(CRATE_CONTENTS_ID)


def main():
    import rospy
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('list', 'show', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('--file', default=None, help="Trajectory log (default: newest)")
        p.add_argument('--robot-name', default='panda')
        if name == 'list':
            p.add_argument('--movement', default=None, help="Only this movement")
        else:
            p.add_argument('index', type=int, help="Record number from 'list'")
            p.add_argument('--source', choices=['planned', 'executed'], default='planned')
            p.add_argument('--speed', type=float, default=0.5, help="Fraction of the logged speed (replay)")
        if name == 'replay':
            p.add_argument('--start-tolerance', type=float, default=0.01, help="rad, max joint difference")
            p.add_argument('--go-to-start', action='store_true', help="Plan to the start configuration first")
            p.add_argument('--with-scene', action='store_true', help="Add the move's logged crate contents")
            p.add_argument('--force', action='store_true', help="Replay despite a collision in the check")
            p.add_argument('--yes', action='store_true', help="Don't ask for confirmation")
    args = parser.parse_args(rospy.myargv()[1:])
    {'list': cmd_list, 'show': cmd_show, 'replay': cmd_replay}[args.cmd](args)


if __name__ == '__main__':
    main()
