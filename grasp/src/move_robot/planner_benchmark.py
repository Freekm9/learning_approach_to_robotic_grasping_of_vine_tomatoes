#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning-only planner benchmark with fixed joint goals (stage A of the planner
selection experiment). Nothing is executed, so it runs against any bringup with
move_group and the crate scene (Gazebo, MoveIt fake execution); use an isolated ROS
master for it, never the one the real robot runs on.

Choosing the goal is separated from planning, so every planner (and every repeat)
plans the same start -> goal query:

  build      Grasp points from a saved crate cloud (capture_full_cloud.py output, in
             the planning frame): a grid over the crate, z just below the top of the
             trusses there, at several yaws. For each point and start configuration,
             both grasp generators pick an orientation exactly as the pipeline does
             (test_crate_reachability.py checks the whole pick sequence):
               topdown  straight down, else its 180-degree roll (Van den Bent's 4-DOF)
               tilted   straight down, roll, then the tilt table (our 6/7-DOF)
             The pre-grasp pose of that orientation is then resolved to ONE joint goal:
             the nearest of --ik-seeds collision-free IK solutions (crate contents in
             the scene, as when grasp() plans to pre-grasp; --no-contents leaves them
             out). Unreachable points are kept in the file with the reason
             (no_orientation: no feasible pick sequence; goal_in_crate_contents: the
             pre-grasp only collides with the crate contents; no_goal_ik); they count
             for reachability, not for a planner.
             Start poses: --start NAME j1..j7 (repeatable), e.g. the robot's capture
             pose, which every pre-grasp move starts from (go_to_saved_pose resets the
             joints there); 'ready' is added unless --no-ready. With --place-joints
             (the joints at the place pose, SavePosePlace), every reachable grasp also
             gets a 'transport' query: from the lift pose after the grasp (nearest IK to
             the pre-grasp goal, as the Cartesian descent/back-out/lift stay near it) to
             go_to_place_above's goal (the place pose at the lift height, nearest IK to
             the lift joints), without the crate contents, as in drop_off().
  run        Plans every reachable query with every planner, --repeats times, to that
             fixed joint goal (OMPL planner names, 'chomp', 'stomp' or 'pink'; Pink is
             differential IK, so it gets the pre-grasp pose and its end configuration
             is compared with the fixed goal). The crate-contents height map from the
             query file's cloud, carved for each query's approach, is in the scene
             (unless the queries were built with --no-contents). Queries with the
             same start and goal (both generators chose the same orientation) are
             planned once; the result is written for each of them (same plan_key).
  summarize  Per move (pre_grasp/transport), goal set (topdown/tilted) and planner:
             success rate (Wilson 95% CI), planning time, joint path length relative to
             the direct joint distance, end-effector path length and minimum clearance
             to the crate walls. --decide applies the selection rule (see decide())
             to the unique plans and tests the others against the chosen planner.

Clearance is measured with Pink's collision model (Franka's *_sc capsules + hand,
fingers and camera vs. the crate boxes), so it is a consistent relative measure, not
the exact distance of the collision meshes; the crate contents are not included.

Usage (sim on its own ROS master, with the gripper the queries should be built for):
    rosrun grasp planner_benchmark.py build                 # newest cloud, ready start
    rosrun grasp planner_benchmark.py build --no-ready --start capture j1 ... j7 --place-joints j1 ... j7
    rosrun grasp planner_benchmark.py build --spacing-m 0.08 --limit-points 5   # smoke test
    rosrun grasp planner_benchmark.py run --queries <dir>/queries.json \\
        --planners RRTConnect BiTRRT RRTstar chomp pink --repeats 5
    rosrun grasp planner_benchmark.py summarize --decide <dir>/results_*.csv
    rosrun grasp planner_benchmark.py show --queries <dir>/queries.json --plan-key pre_grasp:... --planner RRTConnect

build writes experiments/planner_benchmark/<timestamp>/queries.json; run writes
results_<timestamp>.csv next to the query file it used.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stem_segmentation'))

# franka_example_controllers' "ready" pose; close to the SavePose overview pose above the crate
READY_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
JOINT_GOAL_PIPELINES = ("chomp", "stomp")
# Corridor kept free of crate contents along the approach; = plan_movement.py's
# WAYPOINT_ENTRY_OFFSET_M (not imported: plan_movement pulls in the camera/Pilz stack)
CONTENTS_CORRIDOR_M = 0.30

RESULT_FIELDS = ["query_id", "plan_key", "move", "method", "start_name", "x", "y", "z", "yaw_deg", "tilt_x_deg", "tilt_y_deg", "flip",
                 "planner", "repeat", "success", "error_code", "planning_time_s", "num_waypoints",
                 "joint_path_length", "direct_joint_distance", "path_ratio", "ee_path_length_m",
                 "ee_straight_m", "ee_ratio", "min_clearance_mm", "goal_deviation_rad", "pink_reason",
                 "contents_columns"]


def ws_dir():
    import rospkg
    return os.path.dirname(os.path.dirname(os.path.dirname(rospkg.RosPack().get_path('grasp'))))


def joint_path_length(points):
    """Same metric as plan_movement.py's _joint_path_length."""
    return float(sum(np.linalg.norm(np.array(b) - np.array(a)) for a, b in zip(points[:-1], points[1:])))


def pose_to_list(pose_stamped):
    p, o = pose_stamped.pose.position, pose_stamped.pose.orientation
    return [p.x, p.y, p.z, o.x, o.y, o.z, o.w]


def list_to_pose(values, frame_id):
    from geometry_msgs.msg import PoseStamped
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    p, o = pose.pose.position, pose.pose.orientation
    p.x, p.y, p.z, o.x, o.y, o.z, o.w = values
    return pose


class Bench(object):
    """MoveIt connections shared by build and run."""

    def __init__(self, robot_name):
        import rospy
        import moveit_commander
        from move_robot.pink_moveit import PinkMoveItPlanner
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot_name = robot_name
        self.ns = f'/{robot_name}'
        self.robot = moveit_commander.RobotCommander(robot_description=f'{self.ns}/robot_description', ns=self.ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns=self.ns)
        self.move_group = moveit_commander.MoveGroupCommander(
            f'{robot_name}_manipulator', robot_description=f'{self.ns}/robot_description', ns=self.ns)
        self.move_group.set_end_effector_link(f'{robot_name}_hand_tcp')
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.frame = self.move_group.get_planning_frame()
        self.joint_names = self.move_group.get_active_joints()
        self.pink = PinkMoveItPlanner(self.robot, self.scene, ns=self.ns + '/')
        rospy.sleep(1.0)  # let the planning scene interface receive the scene

    def scene_boxes(self):
        from common.crate_contents import CRATE_CONTENTS_ID
        from move_robot.test_crate_reachability import box_world_pose_and_dims
        boxes = []
        for name, obj in self.scene.get_objects().items():
            box = box_world_pose_and_dims(obj) if name != CRATE_CONTENTS_ID else None
            if box is not None:
                boxes.append((name,) + box)
        return boxes

    def contents_obstacle(self):
        """CrateContentsObstacle with plan_movement.py's default parameters."""
        from common.crate_contents import CrateContentsObstacle, crate_interior_from_boxes
        return CrateContentsObstacle(self.scene, None, self.frame, crate_interior_from_boxes(self.scene_boxes()),
                                     cell_m=0.02, carve_radius_m=0.04, corridor_length_m=CONTENTS_CORRIDOR_M)

    def tcp_offset_m(self):
        """hand -> hand_tcp distance of the loaded robot_description (0.1034 + finger extension)."""
        planner = self.pink.get_planner(self.move_group)
        q = READY_JOINTS
        hand = planner.frame_pose(q, f'{self.robot_name}_hand')
        tcp = planner.frame_pose(q, f'{self.robot_name}_hand_tcp')
        return float(np.linalg.norm(tcp.translation - hand.translation))

    def pipeline_loaded(self, pipeline):
        import rospy
        return any(n.startswith(self.ns + '/') and n.endswith(f'/{pipeline}/planning_plugin')
                   for n in rospy.get_param_names())


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def grasp_points(xyz, interior, spacing_m, edge_margin_m, top_crop_z, min_points=20):
    """(x, y, z) grid over the crate interior where the cloud has trusses: z is 3 cm
    below the top of the trusses there, as in test_crate_contents_sim.py."""
    x_lo, x_hi, y_lo, y_hi, floor_z, _ = interior
    points = []
    for x in np.arange(x_lo + edge_margin_m, x_hi - edge_margin_m + 1e-9, spacing_m):
        for y in np.arange(y_lo + edge_margin_m, y_hi - edge_margin_m + 1e-9, spacing_m):
            near = xyz[(np.abs(xyz[:, 0] - x) < 0.015) & (np.abs(xyz[:, 1] - y) < 0.015) & (xyz[:, 2] < top_crop_z)]
            if len(near) < min_points:
                continue
            z = max(np.percentile(near[:, 2], 95) - 0.03, floor_z + 0.02)
            points.append((float(x), float(y), float(z)))
    return points


def orientation_label(orientation):
    if orientation is None:
        return "unreachable"
    dx, dy, flip = orientation
    return "down+flip" if flip else f"tilt {dx}/{dy}" if (dx or dy) else "down"


def nearest_ik(bench, ik_service, pose, start_joints, seeds):
    """Nearest (joint distance to start_joints) of `seeds` collision-free IK solutions
    for pose, with the fingers at the pre-grasp opening. Returns (joints, distance, n)."""
    import rospy
    from moveit_msgs.srv import GetPositionIKRequest
    from common.grasp_motion import PRE_GRASP_GRIPPER_WIDTH_M
    req = GetPositionIKRequest()
    req.ik_request.group_name = bench.move_group.get_name()
    req.ik_request.ik_link_name = bench.move_group.get_end_effector_link()
    req.ik_request.pose_stamped = pose
    req.ik_request.avoid_collisions = True
    req.ik_request.robot_state.is_diff = True
    fingers = [f'{bench.robot_name}_finger_joint1', f'{bench.robot_name}_finger_joint2']
    req.ik_request.robot_state.joint_state.name = list(bench.joint_names) + fingers
    req.ik_request.robot_state.joint_state.position = list(start_joints) + [PRE_GRASP_GRIPPER_WIDTH_M / 2] * 2
    req.ik_request.timeout = rospy.Duration(0.1)
    start = np.array(start_joints)
    best, found = None, 0
    for _ in range(seeds):
        res = ik_service(req)
        if res.error_code.val != 1:
            continue
        found += 1
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        q = np.array([sol[n] for n in bench.joint_names])
        dist = float(np.linalg.norm(q - start))
        if best is None or dist < best[1]:
            best = (q.tolist(), dist)
    return (best[0], best[1], found) if best else (None, None, 0)


def cmd_build(args):
    import rospy
    from moveit_msgs.srv import GetPositionIK
    from pcd_io import load_pcd
    from scipy.spatial.transform import Rotation as R
    from common.crate_contents import crate_interior_from_boxes
    from common.grasp_motion import post_grasp_poses
    from move_robot.test_crate_reachability import (IKChecker, build_straight_down_pose, flip_approach_roll,
                                                    find_reachable_tilt, get_approach_vec, offset_pose,
                                                    tilt_pose, PRE_GRASP_OFFSET_M)

    rospy.init_node('planner_benchmark_build')
    bench = Bench(args.robot_name)
    cloud = args.pcd or sorted(glob.glob(os.path.join(ws_dir(), 'experiments', 'full_crate_clouds', '*.pcd')))[-1]
    xyz, _ = load_pcd(cloud)
    interior = crate_interior_from_boxes(bench.scene_boxes())
    contents = bench.contents_obstacle()
    use_contents = not args.no_contents
    points = grasp_points(xyz, interior, args.spacing_m, args.edge_margin_m, contents.top_crop_z)[:args.limit_points]
    yaws = np.arange(0, 180, args.yaw_step_deg)
    starts = ([] if args.no_ready else [("ready", READY_JOINTS)]) + \
        [(s[0], [float(v) for v in s[1:]]) for s in args.start or []]
    if not starts:
        raise SystemExit("No start poses: drop --no-ready or pass --start NAME j1 ... j7")
    place = None
    if args.place_joints:
        place_pose = bench.pink.get_planner(bench.move_group).frame_pose(args.place_joints,
                                                                         f'{args.robot_name}_hand_tcp')
        place = dict(joints=list(args.place_joints), pose=list(place_pose.translation) +
                     list(R.from_matrix(place_pose.rotation).as_quat()))
    tcp = bench.tcp_offset_m()
    print(f"Cloud {os.path.basename(cloud)}: {len(points)} grasp points x {len(yaws)} yaws x starts "
          f"{[n for n, _ in starts]}{' + transport' if place else ''}; "
          f"hand->TCP {tcp:.4f} m (finger extension {tcp - 0.1034:.3f} m)")

    ik = IKChecker(args.robot_name)
    ik_service = rospy.ServiceProxy(f'{bench.ns}/compute_ik', GetPositionIK)
    queries = []
    for i, (x, y, z) in enumerate(points):
        for yaw in yaws:
            base = build_straight_down_pose(x, y, z, float(yaw), bench.frame)
            # Grasp orientation per generator (reachability check without crate contents,
            # as simple_pick_point.py does it)
            choice = {}
            if ik.check_grasp(base):
                choice["topdown"] = (0, 0, False)
            elif ik.check_grasp(flip_approach_roll(base)):
                choice["topdown"] = (0, 0, True)
            else:
                choice["topdown"] = None
            if choice["topdown"] is not None:
                choice["tilted"] = choice["topdown"]  # the tilted generator tries the same two first
            else:
                ok, dx, dy, flip = find_reachable_tilt(ik, base)
                choice["tilted"] = (dx, dy, flip) if ok else None

            def resolve_goals(orientation):
                """{start_name: goal fields}. Like plan_movement.py's _grasp_orientations, the
                grasp rolled 180 degrees about its approach axis is used instead when its pick
                sequence is feasible too and its pre-grasp IK is nearer the start."""
                dx, dy, flip = orientation
                grasp = flip_approach_roll(base) if flip else tilt_pose(base, dx, dy) if (dx or dy) else base
                candidates = [(grasp, flip)]
                rolled = flip_approach_roll(grasp)
                if ik.check_grasp(rolled):
                    candidates.append((rolled, not flip))
                approach = get_approach_vec(grasp.pose)  # the same for both rolls
                # The goal IK sees the crate contents, as when grasp() plans to pre-grasp
                if use_contents:
                    contents.add([x, y, z], approach, xyz=xyz)
                resolved = {}
                for start_name, start_joints in starts:
                    best = None
                    for pose, is_flipped in candidates:
                        pre = offset_pose(pose, approach, PRE_GRASP_OFFSET_M)
                        goal, dist, found = nearest_ik(bench, ik_service, pre, start_joints, args.ik_seeds)
                        if goal is not None and (best is None or dist < best["direct_joint_distance"]):
                            best = dict(tilt_x_deg=dx, tilt_y_deg=dy, flip=is_flipped, unreachable_reason="",
                                        grasp_pose=pose_to_list(pose), pre_grasp_pose=pose_to_list(pre),
                                        goal_pose=pose_to_list(pre), goal_joints=goal,
                                        direct_joint_distance=dist, ik_solutions=found, reachable=True)
                    resolved[start_name] = best or dict(unreachable_reason="no_goal_ik")
                if use_contents:
                    contents.remove()  # the reachability checks run without it
                    for start_name, start_joints in starts:
                        if resolved[start_name]["unreachable_reason"] == "no_goal_ik" and any(
                                nearest_ik(bench, ik_service, offset_pose(pose, approach, PRE_GRASP_OFFSET_M),
                                           start_joints, 3)[0] is not None for pose, _ in candidates):
                            resolved[start_name]["unreachable_reason"] = "goal_in_crate_contents"
                if place is not None:
                    # Lift pose after this grasp, near the pre-grasp goal it is reached from
                    reached = [r for r in resolved.values() if r.get("reachable")]
                    transport = dict(unreachable_reason="no_lift_ik" if reached else "no_pre_grasp_goal")
                    if reached:
                        grasp_used = list_to_pose(reached[0]["grasp_pose"], bench.frame)
                        _, lift = post_grasp_poses(grasp_used, get_approach_vec(grasp_used.pose))
                        lift_q, _, _ = nearest_ik(bench, ik_service, lift, reached[0]["goal_joints"], args.ik_seeds)
                        if lift_q is not None:
                            # go_to_place_above: the place pose at the current (lift) height,
                            # planned to the IK solution nearest the current joints
                            above = list(place["pose"])
                            above[2] = lift.pose.position.z
                            above_q, dist, found = nearest_ik(bench, ik_service, list_to_pose(above, bench.frame),
                                                              lift_q, args.ik_seeds)
                            transport = dict(unreachable_reason="no_place_above_ik")
                            if above_q is not None:
                                transport = dict(start_joints=lift_q, goal_joints=above_q, goal_pose=above,
                                                 direct_joint_distance=dist, ik_solutions=found, reachable=True,
                                                 unreachable_reason="", tilt_x_deg=dx, tilt_y_deg=dy,
                                                 flip=reached[0]["flip"], grasp_pose=reached[0]["grasp_pose"])
                    resolved["__transport__"] = transport
                return resolved

            # Both generators share the goal when they choose the same orientation
            goals = {}
            for method, orientation in choice.items():
                if orientation is not None and orientation not in goals:
                    goals[orientation] = resolve_goals(orientation)
                for start_name, start_joints in starts + ([("lift", None)] if place else []):
                    move = "transport" if start_name == "lift" else "pre_grasp"
                    q = dict(id=len(queries), move=move, method=method, start_name=start_name,
                             start_joints=list(start_joints) if start_joints else None,
                             x=round(x, 4), y=round(y, 4), z=round(z, 4), yaw_deg=float(yaw),
                             tilt_x_deg=None, tilt_y_deg=None, flip=None, reachable=False,
                             unreachable_reason="no_orientation",
                             grasp_pose=None, pre_grasp_pose=None, goal_pose=None, goal_joints=None,
                             direct_joint_distance=None, ik_solutions=0)
                    if orientation is not None:
                        q.update(goals[orientation]["__transport__" if move == "transport" else start_name])
                    queries.append(q)
            print(f"[{i + 1}/{len(points)}] ({x:.3f}, {y:.3f}, {z:.3f}) yaw {yaw:.0f}: " +
                  ", ".join(f"{m}={orientation_label(o)}" for m, o in choice.items()))
            if rospy.is_shutdown():
                return

    out_dir = os.path.join(ws_dir(), 'experiments', 'planner_benchmark', time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'queries.json')
    meta = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), cloud=os.path.abspath(cloud), planning_frame=bench.frame,
                joint_names=bench.joint_names, hand_tcp_offset_m=tcp, crate_interior=list(interior),
                scene_objects=sorted(n for n, *_ in bench.scene_boxes()), crate_contents=use_contents,
                starts=dict(starts), place=place, args=vars(args))
    with open(out_path, 'w') as f:
        json.dump(dict(meta=meta, queries=queries), f, indent=1)

    for move in ("pre_grasp", "transport"):
        for method in ("topdown", "tilted"):
            qs = [q for q in queries if q["method"] == method and q["move"] == move]
            if not qs:
                continue
            reasons = {}
            for q in qs:
                if not q["reachable"]:
                    reasons[q["unreachable_reason"]] = reasons.get(q["unreachable_reason"], 0) + 1
            print(f"{move} {method}: {sum(q['reachable'] for q in qs)}/{len(qs)} queries reachable; "
                  f"unreachable: {reasons}")
    print(f"\nQueries: {out_path}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def plan_query(bench, planner, query, args):
    """One planning call. Returns (row fields, waypoints or None)."""
    from moveit_msgs.msg import RobotState
    from common.grasp_motion import PRE_GRASP_GRIPPER_WIDTH_M
    mg = bench.move_group
    row = dict(pink_reason="", error_code="")
    t0 = time.time()
    if planner == "pink":
        success, plan = bench.pink.plan(list_to_pose(query.get("goal_pose") or query["pre_grasp_pose"], bench.frame), mg,
                                        start_joint_values=query["start_joints"])
        row["pink_reason"] = bench.pink.last_result.reason
    else:
        start = RobotState()
        start.is_diff = True
        fingers = [f'{bench.robot_name}_finger_joint1', f'{bench.robot_name}_finger_joint2']
        start.joint_state.name = list(bench.joint_names) + fingers
        start.joint_state.position = list(query["start_joints"]) + [PRE_GRASP_GRIPPER_WIDTH_M / 2] * 2
        mg.set_start_state(start)
        if planner in JOINT_GOAL_PIPELINES:
            mg.set_planning_pipeline_id(planner)
        else:
            mg.set_planning_pipeline_id("ompl")
            mg.set_planner_id(planner)
        mg.set_planning_time(args.planning_time)
        mg.set_num_planning_attempts(args.num_planning_attempts)
        mg.set_joint_value_target(dict(zip(bench.joint_names, query["goal_joints"])))
        success, plan, _, error = mg.plan()
        row["error_code"] = error.val
        mg.clear_pose_targets()
    row["planning_time_s"] = round(time.time() - t0, 3)
    row["success"] = bool(success)
    waypoints = [list(p.positions) for p in plan.joint_trajectory.points] if success else None
    return row, waypoints


def path_metrics(bench, waypoints, query):
    planner = bench.pink.get_planner(bench.move_group)
    tcp = f'{bench.robot_name}_hand_tcp'
    ee = np.array([planner.frame_pose(q, tcp).translation for q in waypoints])
    joint_len = joint_path_length(waypoints)
    direct = query["direct_joint_distance"]
    ee_len = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    ee_straight = float(np.linalg.norm(ee[-1] - ee[0]))
    clearance = min((float(np.min(c)) for c in map(planner.clearances, waypoints) if len(c)), default=np.nan)
    return dict(num_waypoints=len(waypoints), joint_path_length=round(joint_len, 4),
                path_ratio=round(joint_len / direct, 3) if direct > 1e-6 else "",
                ee_path_length_m=round(ee_len, 4), ee_straight_m=round(ee_straight, 4),
                ee_ratio=round(ee_len / ee_straight, 3) if ee_straight > 1e-6 else "",
                min_clearance_mm=round(clearance * 1000, 1),
                goal_deviation_rad=round(float(np.linalg.norm(np.array(waypoints[-1]) -
                                                              np.array(query["goal_joints"]))), 4))


def plan_key(query):
    """Stable id of a query's start/goal pair: the same in every run and query file."""
    import hashlib
    values = np.round(list(query["start_joints"]) + list(query["goal_joints"]), 4).tolist()
    return query["move"] + ":" + hashlib.sha1(json.dumps(values).encode()).hexdigest()[:12]


def cmd_run(args):
    import rospy
    rospy.init_node('planner_benchmark_run')
    with open(args.queries) as f:
        data = json.load(f)
    meta, queries = data["meta"], data["queries"]
    queries = [q for q in queries if q["reachable"] and q["method"] in args.methods][:args.limit_queries]

    bench = Bench(args.robot_name)
    if bench.frame != meta["planning_frame"]:
        raise SystemExit(f"Planning frame {bench.frame} != the query file's {meta['planning_frame']}")
    tcp = bench.tcp_offset_m()
    if abs(tcp - meta["hand_tcp_offset_m"]) > 1e-3:
        raise SystemExit(f"hand->TCP is {tcp:.4f} m but the queries were built with {meta['hand_tcp_offset_m']:.4f} m: "
                         f"launch the sim with the same finger_extension_m/tcp_xyz (or rebuild the queries)")
    for p in args.planners:
        if p in JOINT_GOAL_PIPELINES and not bench.pipeline_loaded(p):
            raise SystemExit(f"Pipeline '{p}' is not loaded in move_group (stomp needs stomp:=true and "
                             f"/root/stomp_ws sourced)")

    contents = None
    if not meta.get("crate_contents", True) and not args.no_contents:
        print("Queries were built without crate contents, so planning without them too")
        args.no_contents = True
    if not args.no_contents:
        from pcd_io import load_pcd
        from scipy.spatial.transform import Rotation as R
        contents = bench.contents_obstacle()
        cloud_xyz, _ = load_pcd(meta["cloud"])

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.queries)),
                            f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    # Queries with the same start and goal (both generators chose the same orientation)
    # are planned once; the rows are written for each of them under one plan_key
    groups = {}
    for q in queries:
        q.setdefault("move", "pre_grasp")
        key = (q["move"], tuple(np.round(q["start_joints"], 4)), tuple(np.round(q["goal_joints"], 4)))
        groups.setdefault(key, []).append(q)
    groups = list(groups.values())
    rng = random.Random(args.seed)
    n_total = len(groups) * len(args.planners) * args.repeats
    print(f"{len(queries)} queries = {len(groups)} unique start/goal pairs x {len(args.planners)} planners x "
          f"{args.repeats} repeats = {n_total} plans; crate contents {'off' if contents is None else 'on'}")
    done = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        try:
            for gi, group in enumerate(groups):
                query = group[0]
                n_columns = ""
                if contents is not None:
                    # The crate contents are only in the scene for the move to pre-grasp
                    if query["move"] == "pre_grasp":
                        grasp = query["grasp_pose"]
                        approach = R.from_quat(grasp[3:]).apply([0, 0, 1])
                        n_columns = contents.add(grasp[:3], approach, xyz=cloud_xyz)
                    else:
                        contents.remove()
                results = []
                for repeat in range(args.repeats):
                    order = list(args.planners)
                    rng.shuffle(order)  # no planner always plans right after the scene update
                    for planner in order:
                        if rospy.is_shutdown():
                            return
                        row, waypoints = plan_query(bench, planner, query, args)
                        row.update(plan_key=plan_key(query), planner=planner, repeat=repeat, contents_columns=n_columns,
                                   direct_joint_distance=round(query["direct_joint_distance"], 4))
                        if waypoints:
                            row.update(path_metrics(bench, waypoints, query))
                        for q in group:
                            row.update({k: q[k] for k in ("move", "method", "start_name", "x", "y", "z", "yaw_deg",
                                                          "tilt_x_deg", "tilt_y_deg", "flip")}, query_id=q["id"])
                            writer.writerow(row)
                        f.flush()
                        results.append((planner, row["success"]))
                        done += 1
                print(f"[{gi + 1}/{len(groups)}] {query['move']} {'/'.join(q['method'] for q in group)} "
                      f"({query['x']:.3f}, {query['y']:.3f}) yaw {query['yaw_deg']:.0f}: " +
                      ", ".join(f"{p} {sum(s for pl, s in results if pl == p)}/{args.repeats}" for p in args.planners))
        finally:
            if contents is not None:
                contents.remove()
    print(f"\n{done}/{n_total} plans; results: {out_path}\n")
    summarize([out_path])


def cmd_show(args):
    """Replan one start/goal pair (plan_key) with one planner --tries times and show the
    path with the largest end-effector ratio in RViz: a DisplayTrajectory (robot animation)
    and a TCP line strip, both latched, with the crate contents carved as in run. The paths
    of a run are not stored, so for a sampling planner this finds a path like the logged
    outlier, not that exact path."""
    import rospy
    from scipy.spatial.transform import Rotation as R
    from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    rospy.init_node('planner_benchmark_show')
    with open(args.queries) as f:
        data = json.load(f)
    meta = data["meta"]
    matches = [q for q in data["queries"] if q["reachable"] and plan_key(dict(q, move=q.get("move", "pre_grasp"))) == args.plan_key]
    if not matches:
        raise SystemExit(f"No reachable query with plan_key {args.plan_key} in {args.queries}")
    query = dict(matches[0], move=matches[0].get("move", "pre_grasp"))
    bench = Bench(args.robot_name)

    contents = None
    if query["move"] == "pre_grasp" and meta.get("crate_contents", True) and not args.no_contents:
        from pcd_io import load_pcd
        contents = bench.contents_obstacle()
        cloud_xyz, _ = load_pcd(meta["cloud"])
        grasp = query["grasp_pose"]
        contents.add(grasp[:3], R.from_quat(grasp[3:]).apply([0, 0, 1]), xyz=cloud_xyz)
    try:
        best = None
        for i in range(args.tries):
            row, waypoints = plan_query(bench, args.planner, query, args)
            if not waypoints:
                print(f"try {i}: failed ({row['error_code'] or row['pink_reason']})")
                continue
            row.update(path_metrics(bench, waypoints, query))
            print(f"try {i}: path ratio {row['path_ratio']}, ee ratio {row['ee_ratio']}, "
                  f"min clearance {row['min_clearance_mm']} mm, {row['planning_time_s']} s")
            if best is None or (row['ee_ratio'] or 0) > (best[0]['ee_ratio'] or 0):
                best = (row, waypoints)
            if args.min_ee_ratio and (row['ee_ratio'] or 0) >= args.min_ee_ratio:
                break
        if best is None:
            raise SystemExit("No successful plan")
        row, waypoints = best

        start = RobotState()
        start.joint_state.name = list(bench.joint_names)
        start.joint_state.position = list(query["start_joints"])
        traj = JointTrajectory(joint_names=list(bench.joint_names))
        for k, q in enumerate(waypoints):
            traj.points.append(JointTrajectoryPoint(positions=q, time_from_start=rospy.Duration(0.05 * k)))
        display = DisplayTrajectory(trajectory_start=start, trajectory=[RobotTrajectory(joint_trajectory=traj)])
        planner = bench.pink.get_planner(bench.move_group)
        line = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns="tcp_path", id=0)
        line.scale.x = 0.004
        line.header.frame_id = bench.frame
        line.pose.orientation.w = 1.0
        line.color.r, line.color.g, line.color.a = 1.0, 0.4, 1.0
        line.points = [Point(*planner.frame_pose(q, f'{bench.robot_name}_hand_tcp').translation) for q in waypoints]
        pub_traj = rospy.Publisher('/planner_benchmark/display_path', DisplayTrajectory, queue_size=1, latch=True)
        pub_line = rospy.Publisher('/planner_benchmark/tcp_path', Marker, queue_size=1, latch=True)
        rospy.sleep(0.5)
        pub_traj.publish(display)
        pub_line.publish(line)
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(dict(plan_key=args.plan_key, planner=args.planner, metrics=row, waypoints=waypoints), f)
        print(f"\nShowing {args.planner} on {args.plan_key} ({query['method']} ({query['x']:.3f}, {query['y']:.3f}) "
              f"yaw {query['yaw_deg']:.0f}): ee ratio {row['ee_ratio']}, path ratio {row['path_ratio']}.\n"
              f"RViz (same ROS master): Trajectory display on /planner_benchmark/display_path, Marker on "
              f"/planner_benchmark/tcp_path. Ctrl-C to quit (removes the crate contents).")
        rospy.spin()
    finally:
        if contents is not None:
            contents.remove()


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def load_results(paths):
    import pandas as pd
    frames = []
    for path in paths:
        f = pd.read_csv(path)
        if "move" not in f:  # results from before transport queries existed
            f["move"] = "pre_grasp"
        if "plan_key" not in f:
            f["plan_key"] = os.path.abspath(path) + ":" + f["query_id"].astype(str)
        f["run"] = os.path.basename(path)
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d["success"] = d["success"].astype(str).str.lower() == "true"
    return d


def summarize(paths, decide_args=None):
    import pandas as pd
    d = load_results(paths)
    rows = []
    for (move, method, planner), g in d.groupby(["move", "method", "planner"]):
        ok = g[g["success"]]
        lo, hi = wilson(int(g["success"].sum()), len(g))
        rows.append(dict(
            move=move, method=method, planner=planner, plans=len(g), queries=g["query_id"].nunique(),
            success=f"{g['success'].mean() * 100:.1f}% [{lo * 100:.0f}-{hi * 100:.0f}]",
            t_med_s=ok["planning_time_s"].median(), t_p95_s=ok["planning_time_s"].quantile(0.95),
            ratio_med=ok["path_ratio"].median(), ratio_p95=ok["path_ratio"].quantile(0.95),
            ee_len_med_m=ok["ee_path_length_m"].median(), ee_ratio_med=ok["ee_ratio"].median(),
            clear_med_mm=ok["min_clearance_mm"].median(), clear_min_mm=ok["min_clearance_mm"].min(),
            goal_dev_max=ok["goal_deviation_rad"].max()))
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.3f}'.format):
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nsuccess: per plan, Wilson 95% CI. Times/lengths over successful plans. ratio = joint path / direct "
          "joint distance to the fixed goal; ee_ratio = TCP path / straight TCP distance; clearance to the crate "
          "walls (Pink capsules). goal_dev = joint distance of the end configuration from the fixed goal (Pink).")
    if decide_args is not None:
        print()
        decide(d, decide_args)


def decide(d, args):
    """Selection rule, fixed before looking at the data (see the thesis protocol), applied
    to the unique plans (a start/goal pair shared by both goal sets counts once), over
    all moves together, since the pipeline uses one planner for every planned move:
      1. success rate within --success-margin of the best planner;
      2. 95th percentile planning time (successful plans) at most --max-p95-time-s;
      3. lowest median path ratio; planners within --ratio-tie of it are tied, and the
         tie is broken by the lower median planning time.
    Each other planner is then compared with the chosen one on the same plans: McNemar
    (exact binomial on the discordant pairs) for success, Wilcoxon signed-rank on the
    per-start/goal median path ratio for path length."""
    import pandas as pd
    from scipy.stats import binomtest, wilcoxon
    u = d.drop_duplicates(["run", "plan_key", "planner", "repeat"])
    stats = {}
    for planner, g in u.groupby("planner"):
        ok = g[g["success"]]
        stats[planner] = dict(success=g["success"].mean(), t_p95=ok["planning_time_s"].quantile(0.95),
                              t_med=ok["planning_time_s"].median(), ratio_med=ok["path_ratio"].median(),
                              n=len(g))
    best_success = max(s["success"] for s in stats.values())
    print(f"Decision over {u['plan_key'].nunique()} unique start/goal pairs "
          f"({', '.join(f'{m}: {n}' for m, n in u.drop_duplicates('plan_key')['move'].value_counts().items())}):")
    eligible = []
    for planner, s in sorted(stats.items()):
        why = []
        if s["success"] < best_success - args.success_margin:
            why.append(f"success {s['success'] * 100:.1f}% < best {best_success * 100:.1f}% - "
                       f"{args.success_margin * 100:.0f} pp")
        if not s["t_p95"] <= args.max_p95_time_s:
            why.append(f"p95 planning time {s['t_p95']:.2f} s > {args.max_p95_time_s} s")
        print(f"  {planner:12s} success {s['success'] * 100:5.1f}%  p95 t {s['t_p95']:.2f} s  "
              f"median ratio {s['ratio_med']:.3f}  -> {'; '.join(why) if why else 'eligible'}")
        if not why:
            eligible.append(planner)
    if not eligible:
        print("No planner passes steps 1-2; relax the thresholds only with a reason you can report.")
        return
    best_ratio = min(stats[p]["ratio_med"] for p in eligible)
    tied = [p for p in eligible if stats[p]["ratio_med"] <= best_ratio + args.ratio_tie]
    chosen = min(tied, key=lambda p: stats[p]["t_med"])
    print(f"Lowest median path ratio {best_ratio:.3f}; tied within {args.ratio_tie}: {tied}; "
          f"chosen (lowest median planning time among them): {chosen}")

    ref = u[u["planner"] == chosen].set_index(["run", "plan_key", "repeat"])
    ref_ratio = u[(u["planner"] == chosen) & u["success"]].groupby("plan_key")["path_ratio"].median()
    print(f"\nPaired comparison with {chosen}:")
    for planner in sorted(stats):
        if planner == chosen:
            continue
        other = u[u["planner"] == planner].set_index(["run", "plan_key", "repeat"])
        both = ref[["success"]].join(other[["success"]], rsuffix="_o", how="inner")
        only_ref = int((both["success"] & ~both["success_o"]).sum())
        only_other = int((~both["success"] & both["success_o"]).sum())
        p_succ = binomtest(only_ref, only_ref + only_other).pvalue if only_ref + only_other else 1.0
        o_ratio = u[(u["planner"] == planner) & u["success"]].groupby("plan_key")["path_ratio"].median()
        pair = pd.concat([ref_ratio, o_ratio], axis=1, keys=["ref", "o"]).dropna()
        diff = pair["o"] - pair["ref"]
        p_ratio = wilcoxon(diff).pvalue if (diff.abs() > 1e-9).any() else 1.0
        print(f"  {planner:12s} success: only {chosen} {only_ref}, only {planner} {only_other} (McNemar p={p_succ:.3g}); "
              f"path ratio {planner} - {chosen}: median {diff.median():+.3f} over {len(pair)} pairs "
              f"(Wilcoxon p={p_ratio:.3g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help="Build the fixed-goal query file")
    b.add_argument('--robot-name', default='panda')
    b.add_argument('--pcd', default=None, help="Crate cloud (default: newest in experiments/full_crate_clouds)")
    b.add_argument('--spacing-m', type=float, default=0.05, help="Grasp point grid spacing (m)")
    b.add_argument('--edge-margin-m', type=float, default=0.03, help="Inset of the grid from the walls (m)")
    b.add_argument('--yaw-step-deg', type=float, default=45.0, help="Yaws 0, step, ... < 180")
    b.add_argument('--start', nargs=8, action='append', metavar=('NAME', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'),
                   help="Named start configuration (repeatable), e.g. --start capture 0.0 -0.6 ...")
    b.add_argument('--no-ready', action='store_true', help="Don't add the 'ready' start pose")
    b.add_argument('--place-joints', type=float, nargs=7, default=None,
                   help="Joints at the place pose (SavePosePlace): adds lift -> place-above transport queries")
    b.add_argument('--ik-seeds', type=int, default=30, help="IK solutions tried for the nearest joint goal")
    b.add_argument('--limit-points', type=int, default=None, help="Only the first N grasp points (smoke test)")
    b.add_argument('--no-contents', action='store_true',
                   help="Resolve the goals without the crate-contents height map (run then plans without it too)")

    r = sub.add_parser('run', help="Plan every query with every planner")
    r.add_argument('--robot-name', default='panda')
    r.add_argument('--queries', required=True)
    r.add_argument('--planners', nargs='+', default=['RRTConnect', 'BiTRRT', 'RRTstar', 'chomp'],
                   help="OMPL planner names, 'chomp', 'stomp', 'pink'")
    r.add_argument('--methods', nargs='+', default=['topdown', 'tilted'], choices=['topdown', 'tilted'])
    r.add_argument('--repeats', type=int, default=5)
    r.add_argument('--planning-time', type=float, default=5.0, help="as plan_movement.py")
    r.add_argument('--num-planning-attempts', type=int, default=100, help="as plan_movement.py")
    r.add_argument('--no-contents', action='store_true', help="Crate walls only, no crate-contents height map")
    r.add_argument('--limit-queries', type=int, default=None)
    r.add_argument('--seed', type=int, default=0, help="Planner order shuffle")

    v = sub.add_parser('show', help="Replan one start/goal pair and show the path in RViz")
    v.add_argument('--robot-name', default='panda')
    v.add_argument('--queries', required=True)
    v.add_argument('--plan-key', required=True, help="plan_key column of a results CSV")
    v.add_argument('--planner', required=True)
    v.add_argument('--tries', type=int, default=20, help="Plans tried; the one with the largest ee ratio is shown")
    v.add_argument('--min-ee-ratio', type=float, default=None, help="Stop at the first plan with at least this ee ratio")
    v.add_argument('--planning-time', type=float, default=5.0)
    v.add_argument('--num-planning-attempts', type=int, default=100)
    v.add_argument('--no-contents', action='store_true')
    v.add_argument('--save', default=None, help="Write the shown path (waypoints + metrics) to this JSON file")

    s = sub.add_parser('summarize', help="Summary table of one or more results CSVs")
    s.add_argument('results', nargs='+')
    s.add_argument('--decide', action='store_true', help="Apply the planner selection rule (see decide())")
    s.add_argument('--success-margin', type=float, default=0.02, help="Step 1: allowed success gap to the best")
    s.add_argument('--max-p95-time-s', type=float, default=2.0, help="Step 2: max 95th percentile planning time")
    s.add_argument('--ratio-tie', type=float, default=0.02, help="Step 3: median path ratios this close are tied")

    import rospy
    args = parser.parse_args(rospy.myargv()[1:])
    if args.cmd == 'build':
        cmd_build(args)
    elif args.cmd == 'run':
        cmd_run(args)
    elif args.cmd == 'show':
        cmd_show(args)
    else:
        summarize(args.results, args if args.decide else None)


if __name__ == '__main__':
    main()
