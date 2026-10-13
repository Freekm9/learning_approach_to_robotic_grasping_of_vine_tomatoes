#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pink (differential inverse kinematics on Pinocchio) as a motion "planner" for
plan_movement.py's go_to_pose, so it can be compared against the OMPL/CHOMP
pipelines on the same moves (select it with `~pipeline_id: pink`).

Pink is not a sampling-based planner: it integrates a joint velocity that solves a
small QP every step (move the goal frame towards its target, stay close to the start
posture, respect joint position/velocity limits), and keeps clear of the crate through
control-barrier constraints on robot-vs-crate distances. That makes it fast and
smooth, but local: it can get stuck in front of a wall, so a stalled run counts as a
planning failure (as an OMPL timeout would).

This module is ROS-free (only pinocchio/hppfcl/pink/numpy), so it can also be run
offline against the URDF and a .scene file: see test_pink_planner.py. Pinocchio and
hpp-fcl come from ros-noetic-pinocchio; pink, qpsolvers and the QP solvers from pip
(`pip install --no-deps pin-pink==3.1.0 qpsolvers loop-rate-limiters && pip install
daqp quadprog`; --no-deps so pip doesn't install a second Pinocchio next to ROS's).
The ROS Pinocchio python module is in /opt/ros/noetic/lib/python3.8/site-packages,
which isn't on the default PYTHONPATH; it is added below if missing."""

import sys
import time

ROS_PINOCCHIO_PATH = "/opt/ros/noetic/lib/python3.8/site-packages"
if ROS_PINOCCHIO_PATH not in sys.path:
    sys.path.append(ROS_PINOCCHIO_PATH)

import numpy as np
import hppfcl
import pinocchio as pin
import pink
import qpsolvers
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask

# Robot geometry on these links is never paired with the crate: link0/link1 sit on the
# floor box at the base and can't reach the crate anyway.
UNCHECKED_LINK_SUFFIXES = ("link0", "link0_sc", "link1", "link1_sc")


class PinkPlanResult(object):
    def __init__(self, success, waypoints, reason, planning_time_s, iterations,
                 position_error_m, orientation_error_rad, min_clearance_m):
        self.success = success
        self.waypoints = waypoints  # list of arm joint vectors, start first
        self.reason = reason
        self.planning_time_s = planning_time_s
        self.iterations = iterations
        self.position_error_m = position_error_m
        self.orientation_error_rad = orientation_error_rad
        self.min_clearance_m = min_clearance_m


class PinkPlanner(object):
    """Differential-IK motion generator for one arm, with crate walls as barriers.

    urdf_xml: robot_description string. package_dirs: directories containing the
    packages the URDF's package:// meshes point into. arm_joint_names: the joints Pink
    may move (MoveIt group's active joints); every other joint (the fingers) is locked
    at locked_joint_values (name -> value, default 0). boxes: iterable of
    (name, center_xyz, rotation_3x3, dims_xyz) in the planning frame (the URDF root).
    collision_geometry: "capsules" uses only Franka's coarse *_sc self-collision
    capsules for the arm plus the hand/finger geometry (fast), "meshes" the arm's full
    collision meshes instead (exact but slower)."""

    def __init__(self, urdf_xml, package_dirs, arm_joint_names, boxes,
                 locked_joint_values=None, collision_geometry="capsules",
                 floor_names=("floor",)):
        locked_joint_values = locked_joint_values or {}
        full_model = pin.buildModelFromXML(urdf_xml)
        full_geom = pin.buildGeomFromUrdfString(full_model, urdf_xml, pin.GeometryType.COLLISION,
                                                package_dirs=package_dirs)

        q_ref = pin.neutral(full_model)
        lock_ids = []
        for jid in range(1, full_model.njoints):
            name = full_model.names[jid]
            if name in arm_joint_names:
                continue
            lock_ids.append(jid)
            if name in locked_joint_values:
                q_ref[full_model.joints[jid].idx_q] = locked_joint_values[name]
        self.model, [self.geom_model] = pin.buildReducedModel(full_model, [full_geom], lock_ids, q_ref)
        self.data = self.model.createData()
        # Pinocchio orders joints by the kinematic tree; keep a map to MoveIt's order
        self.arm_joint_names = list(arm_joint_names)
        self._idx_q = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.arm_joint_names]

        self._select_robot_geometry(collision_geometry)
        robot_geom_ids = [i for i, g in enumerate(self.geom_model.geometryObjects)
                          if not self._link_name(g).endswith(UNCHECKED_LINK_SUFFIXES)]

        self.box_names = []
        for name, center, rot, dims in boxes:
            placement = pin.SE3(np.asarray(rot, dtype=float), np.asarray(center, dtype=float))
            obj = pin.GeometryObject("env_" + name, 0, 0, placement, hppfcl.Box(*[float(d) for d in dims]))
            gid = self.geom_model.addGeometryObject(obj)
            is_floor = any(f in name.lower() for f in floor_names)
            for rid in robot_geom_ids:
                # The hand is the only part that should ever get near the floor box
                # inside the crate; the floor pair for upper arm links just adds cost.
                if is_floor and "hand" not in self._link_name(self.geom_model.geometryObjects[rid]) \
                        and "finger" not in self._link_name(self.geom_model.geometryObjects[rid]):
                    continue
                self.geom_model.addCollisionPair(pin.CollisionPair(rid, gid))
            self.box_names.append(name)
        self.geom_data = pin.GeometryData(self.geom_model)

    def _link_name(self, geom):
        # buildGeomFromUrdf names geometry "<link>_<index>"
        return geom.name.rsplit("_", 1)[0]

    def _select_robot_geometry(self, collision_geometry):
        if collision_geometry not in ("capsules", "meshes"):
            raise ValueError("collision_geometry must be 'capsules' or 'meshes'")
        objs = list(self.geom_model.geometryObjects)
        sc_links = {self._link_name(g)[:-len("_sc")] for g in objs if self._link_name(g).endswith("_sc")}
        keep = []
        for g in objs:
            link = self._link_name(g)
            if link.endswith("_sc"):
                # The hand capsule is very coarse (it swallows the fingers); the hand's
                # own mesh/boxes are used instead, in both modes.
                use = collision_geometry == "capsules" and not link.endswith("hand_sc")
            elif link in sc_links and "hand" not in link:
                use = collision_geometry == "meshes"
            else:
                use = True  # hand, fingers, camera: no capsule alternative
            if use:
                keep.append(g)
        new_geom = pin.GeometryModel()
        for g in keep:
            new_geom.addGeometryObject(g)
        self.geom_model = new_geom

    def to_pinocchio_q(self, arm_q):
        q = pin.neutral(self.model)
        for i, v in zip(self._idx_q, arm_q):
            q[i] = v
        return q

    def to_arm_q(self, q):
        return [float(q[i]) for i in self._idx_q]

    def frame_pose(self, arm_q, frame):
        q = self.to_pinocchio_q(arm_q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.model.getFrameId(frame)].copy()

    def clearances(self, arm_q):
        """Distance per robot-vs-box collision pair at arm_q (negative = penetration)."""
        q = self.to_pinocchio_q(arm_q)
        pin.computeDistances(self.model, self.data, self.geom_model, self.geom_data, q)
        return np.array([r.min_distance for r in self.geom_data.distanceResults])

    def describe_closest_pair(self, arm_q):
        d = self.clearances(arm_q)
        k = int(np.argmin(d))
        cp = self.geom_model.collisionPairs[k]
        return (self.geom_model.geometryObjects[cp.first].name,
                self.geom_model.geometryObjects[cp.second].name, float(d[k]))

    def plan(self, start_arm_q, goal_frame, goal_pose, dt=0.01, max_time_s=20.0,
             position_tolerance_m=0.002, orientation_tolerance_rad=np.deg2rad(1.0),
             d_min=0.015, barrier_gain=1.0, n_barrier_pairs=8,
             orientation_cost=1.0, posture_cost=1e-3, lm_damping=1e-4,
             stall_window_s=2.0, stall_progress=1e-3, waypoint_every=5, solver=None):
        """Integrate differential IK from start_arm_q until goal_frame reaches goal_pose
        (pin.SE3 in the planning frame). Returns a PinkPlanResult whose waypoints are
        arm joint vectors in arm_joint_names order, every `waypoint_every` steps.

        dt is integration time, not wall time: the waypoints only form a path, which
        the caller time-parameterizes (MoveIt's retime_trajectory) like a Cartesian
        path. Planning fails if the goal isn't reached within max_time_s of integrated
        time, or if the task error doesn't drop by stall_progress (relative) over
        stall_window_s -- Pink got stuck in a local minimum, typically against a wall."""
        t0 = time.time()
        solver = solver or ("daqp" if "daqp" in qpsolvers.available_solvers else "quadprog")
        q = self.to_pinocchio_q(start_arm_q)
        configuration = pink.Configuration(self.model, self.data, q,
                                           collision_model=self.geom_model, collision_data=self.geom_data)

        ee_task = FrameTask(goal_frame, position_cost=1.0, orientation_cost=orientation_cost,
                            lm_damping=lm_damping)
        ee_task.set_target(goal_pose)
        posture_task = PostureTask(cost=posture_cost)
        posture_task.set_target(q)
        n_pairs = min(n_barrier_pairs, len(self.geom_model.collisionPairs))
        barriers = [SelfCollisionBarrier(n_collision_pairs=n_pairs, gain=barrier_gain,
                                         safe_displacement_gain=1.0, d_min=d_min)] if n_pairs else []
        frame_id = self.model.getFrameId(goal_frame)

        def errors():
            current = configuration.data.oMf[frame_id]
            pos_err = float(np.linalg.norm(goal_pose.translation - current.translation))
            rot_err = float(np.linalg.norm(pin.log3(current.rotation.T @ goal_pose.rotation)))
            return pos_err, rot_err

        waypoints = [self.to_arm_q(configuration.q)]
        min_clearance = float(np.min(self._current_distances())) if n_pairs else np.inf
        stall_steps = max(1, int(round(stall_window_s / dt)))
        history = []
        max_steps = int(round(max_time_s / dt))
        reason = "timeout"
        success = False
        step = 0
        pos_err, rot_err = errors()
        for step in range(1, max_steps + 1):
            try:
                velocity = pink.solve_ik(configuration, [ee_task, posture_task], dt, solver=solver,
                                         barriers=barriers, safety_break=False)
            except Exception as e:  # infeasible QP (e.g. barrier already violated)
                reason = f"qp_failed: {type(e).__name__}: {e}"
                break
            configuration.integrate_inplace(velocity, dt)
            if n_pairs:
                min_clearance = min(min_clearance, float(np.min(self._current_distances())))
            if step % waypoint_every == 0:
                waypoints.append(self.to_arm_q(configuration.q))

            pos_err, rot_err = errors()
            if pos_err < position_tolerance_m and rot_err < orientation_tolerance_rad:
                success = True
                reason = "converged"
                break
            # Normalize so position (m) and orientation (rad) both count: 1 cm ~ 0.1 rad
            history.append(pos_err + 0.1 * rot_err)
            if len(history) > stall_steps:
                old = history[-stall_steps - 1]
                if old - history[-1] < stall_progress * max(old, 1e-6):
                    reason = "stalled"
                    break

        final = self.to_arm_q(configuration.q)
        if waypoints[-1] != final:
            waypoints.append(final)
        if n_pairs and min_clearance < 0.0:
            success = False
            reason = "collision"
        return PinkPlanResult(success, waypoints, reason, time.time() - t0, step,
                              pos_err, rot_err, min_clearance)

    def _current_distances(self):
        return np.array([r.min_distance for r in self.geom_data.distanceResults])


def parse_scene_file(path):
    """Read box primitives from a MoveIt .scene text file into
    (name, center, rotation_3x3, dims) tuples (planning frame). Only the format the
    crate scenes in panda_moveit_config_vine_tomato use: one box per object."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    boxes = []
    i = 1  # first line is the scene name
    while i < len(lines) and lines[i] != ".":
        if not lines[i].startswith("*"):
            i += 1
            continue
        name = lines[i][1:].strip()
        obj_pos = np.array([float(v) for v in lines[i + 1].split()])
        obj_quat = np.array([float(v) for v in lines[i + 2].split()])  # x y z w
        n_shapes = int(lines[i + 3])
        j = i + 4
        for _ in range(n_shapes):
            shape = lines[j]
            if shape != "box":
                raise ValueError(f"Only box shapes are supported, got '{shape}' in {name}")
            dims = np.array([float(v) for v in lines[j + 1].split()])
            prim_pos = np.array([float(v) for v in lines[j + 2].split()])
            prim_quat = np.array([float(v) for v in lines[j + 3].split()])
            obj_rot = pin.Quaternion(obj_quat[3], *obj_quat[:3]).toRotationMatrix()
            prim_rot = pin.Quaternion(prim_quat[3], *prim_quat[:3]).toRotationMatrix()
            boxes.append((name, obj_rot @ prim_pos + obj_pos, obj_rot @ prim_rot, dims))
            j += 6  # dims, pos, quat, color, (blank-stripped) extra line
        i = j
    return boxes
