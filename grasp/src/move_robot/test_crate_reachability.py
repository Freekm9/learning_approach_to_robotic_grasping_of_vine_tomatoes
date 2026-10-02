#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep IK reachability over a ring of points near the crate walls, at a
range of yaw angles, and log which ones are reachable straight-down and
which need a tilted (pitch/roll) approach instead.

This is a pure IK/collision query against the live MoveIt planning scene --
it never commands the arm to move, so it's safe to run against any bringup
that has `compute_ik` and the crate collision scene up (Gazebo, MoveIt fake
execution, or real hardware); Gazebo/fake is the natural choice since there's
no reason to point it at the real robot for a query that never moves it.

Reachability test mirrors simple_pick_point.py's `_find_reachable_orientation`
exactly: try the straight-down orientation first (checking the pre-grasp/sink/
back-out/lift poses and straight lines grasp() actually visits, not just the grasp point itself); if
that's unreachable or in collision, try the same orientation with the wrist
rolled 180 degrees (free for a symmetric gripper -- see `flip_approach_roll`);
if that also fails, fall back through the same tilt table (closest to
straight-down first) until one clears IK, or none do.

Crate geometry (interior opening, wall extents) is read live from the MoveIt
planning scene rather than hardcoded, so this keeps working if the loaded
.scene file changes -- see `compute_crate_interior()`.

Usage:
    rosrun grasp test_crate_reachability.py --robot-name panda

    # denser ring, every 30 degrees, custom margin from the walls:
    rosrun grasp test_crate_reachability.py --spacing-m 0.02 --yaw-step-deg 30 \\
        --edge-margin-m 0.03

    # quick smoke test on a handful of points:
    rosrun grasp test_crate_reachability.py --limit-points 4

    # test every combined x-tilt/y-tilt orientation directly (not just the
    # closest-to-straight-down fallback winner) -- e.g. 20 deg x-tilt crossed
    # with 20 deg y-tilt, plus their negatives and straight-down:
    rosrun grasp test_crate_reachability.py --sweep-tilts \\
        --x-tilt-deg 0 20 -20 --y-tilt-deg 0 20 -20
"""
import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import rospy
import rospkg
import moveit_commander
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from moveit_msgs.msg import MoveItErrorCodes, DisplayRobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

from common.grasp_motion import post_grasp_poses, pre_grasp_gripper_state, CartesianMotionChecker

# Reverse lookup (int -> name) built from MoveItErrorCodes' generated int constants,
# so failed compute_ik calls can be logged as e.g. "NO_IK_SOLUTION" instead of a bare
# int -- diagnostic only, doesn't affect which poses count as reachable.
_ERROR_CODE_NAMES = {getattr(MoveItErrorCodes, name): name
                      for name in dir(MoveItErrorCodes)
                      if name.isupper() and isinstance(getattr(MoveItErrorCodes, name), int)}


def moveit_error_string(val):
    return _ERROR_CODE_NAMES.get(val, f"UNKNOWN({val})")


# Must match plan_movement.py's grasp() / simple_pick_point.py's _check_ik --
# a reachable grasp point doesn't guarantee the offset poses grasp() actually
# visits are reachable too, so every candidate is checked at all of them
# (back-out/lift: common/grasp_motion.py).
PRE_GRASP_OFFSET_M = 0.05
SINK_OFFSET_M = 0.0

# TRAC-IK (see kinematics.yaml) does randomized internal restarts within its timeout,
# so a single NO_IK_SOLUTION on a pose near the edge of the workspace can be a false
# negative -- retry a few times before treating a pose as genuinely unreachable. Must
# match simple_pick_point.py's IK_CHECK_RETRIES so this sweep reports the same verdict
# the live pipeline would reach.
IK_CHECK_RETRIES = 3

# Fallback orientations tried when the straight-down grasp is unreachable/in
# collision -- identical table to simple_pick_point.py's TILT_ANGLES_DEG /
# Y_TILT_ANGLES_DEG, so a point marked "reachable via tilt" here means
# simple_pick_point.py would actually find it too.
TILT_ANGLES_DEG = [0, 20, -20, 40, -40, 60, -60, 80, -80]
Y_TILT_ANGLES_DEG = [0, 20, -20, 40, -40, 60, -60, 80, -80]


# --------------------------------------------------------------------------
# Reachability check -- copied from simple_pick_point.py's _get_approach_vec /
# _offset_pose / _tilt_pose / _check_ik so this script tests exactly what
# that node would find, without needing simple_pick_point running.
# --------------------------------------------------------------------------

def get_approach_vec(pose):
    """Return the TCP z-axis (approach direction) in the planning frame. Mirrors plan_movement.py."""
    o = pose.orientation
    rot = R.from_quat([o.x, o.y, o.z, o.w])
    return rot.apply([0, 0, 1])


def offset_pose(pose_stamped, approach_vec, distance):
    """Return a copy of pose_stamped shifted `distance` back along approach_vec."""
    offset = copy.deepcopy(pose_stamped)
    offset.pose.position.x -= approach_vec[0] * distance
    offset.pose.position.y -= approach_vec[1] * distance
    offset.pose.position.z -= approach_vec[2] * distance
    return offset


def tilt_pose(pose_stamped, deg_x, deg_y):
    """Return a copy of pose_stamped rotated deg_x around the vine axis (TCP x-axis)
    and deg_y around the resulting gripper-closing axis (TCP y-axis). Mirrors
    simple_pick_point.py's _tilt_pose exactly."""
    o = pose_stamped.pose.orientation
    base_rot = R.from_quat([o.x, o.y, o.z, o.w])
    vine_axis = base_rot.apply([1, 0, 0])
    vine_axis /= np.linalg.norm(vine_axis)
    x_tilt_rot = R.from_rotvec(np.radians(deg_x) * vine_axis)
    x_tilted_rot = x_tilt_rot * base_rot
    y_axis = x_tilted_rot.apply([0, 1, 0])
    y_axis /= np.linalg.norm(y_axis)
    y_tilt_rot = R.from_rotvec(np.radians(deg_y) * y_axis)
    new_q = (y_tilt_rot * x_tilted_rot).as_quat()

    tilted = copy.deepcopy(pose_stamped)
    tilted.pose.orientation.x, tilted.pose.orientation.y, tilted.pose.orientation.z, tilted.pose.orientation.w = new_q
    return tilted


def flip_approach_roll(pose_stamped):
    """Return a copy of pose_stamped rolled 180 degrees around its own approach axis
    (TCP z). Mirrors simple_pick_point.py's _flip_approach_roll exactly -- a symmetric
    parallel-jaw gripper grasps identically either way, but this puts the wrist (last
    joint) at a completely different angle, which can clear a joint limit the original
    roll couldn't."""
    o = pose_stamped.pose.orientation
    base_rot = R.from_quat([o.x, o.y, o.z, o.w])
    flipped_rot = base_rot * R.from_rotvec([0, 0, np.pi])
    new_q = flipped_rot.as_quat()

    flipped = copy.deepcopy(pose_stamped)
    flipped.pose.orientation.x, flipped.pose.orientation.y, flipped.pose.orientation.z, flipped.pose.orientation.w = new_q
    return flipped


class IKChecker:
    def __init__(self, robot_name, visualize=False, visualize_pause_s=1.0):
        # Fully-qualified (leading-slash) name -- gazebo_moveit.launch/launch_moveit.launch
        # both run move_group inside <group ns="$(arg robot_name)">, so the plain relative
        # name "compute_ik" only resolves correctly if this script is itself launched
        # inside that same namespace. Building the absolute name here means it works from
        # a plain `rosrun`, matching how run_planner_experiment.py addresses its services.
        ik_service_name = f'/{robot_name}/compute_ik'
        rospy.loginfo(f"Waiting for {ik_service_name} ...")
        rospy.wait_for_service(ik_service_name, timeout=30.0)
        self.service = rospy.ServiceProxy(ik_service_name, GetPositionIK)
        self.robot_name = robot_name
        self.group_name = robot_name + "_manipulator"
        self.link_name = robot_name + "_hand_tcp"
        self.num_calls = 0
        self.motion_checker = CartesianMotionChecker(robot_name, ns=f'/{robot_name}/')

        # If set, publish every solved grasp pose's joint solution as a DisplayRobotState so
        # it shows up as a "ghost" robot in RViz -- this never commands the real/simulated
        # robot, it just renders a robot_state message, which is why it's safe to leave on
        # for a script whose whole point is to never move the arm.
        self.visualize = visualize
        self.visualize_pause_s = visualize_pause_s
        if visualize:
            self.display_pub = rospy.Publisher(f'/{robot_name}/display_robot_state',
                                                DisplayRobotState, queue_size=1, latch=True)

    def check_single(self, pose_stamped, capture_solution=False):
        """Return True if MoveIt can find a collision-free IK solution for this pose,
        retrying up to IK_CHECK_RETRIES times (see its comment) before giving up.

        capture_solution=True additionally publishes the found joint solution for RViz
        (see __init__) and pauses briefly so it's actually visible before the next point
        overwrites it -- only meant for the primary grasp pose, not the pre-grasp/sink/
        back-out/lift poses check_grasp() also checks, so the sweep doesn't slow to a crawl."""
        for attempt in range(IK_CHECK_RETRIES):
            ok, error_code, res = self._call_once(pose_stamped)
            if ok:
                if attempt > 0:
                    rospy.loginfo_throttle(1.0, f"IK succeeded on retry {attempt + 1}/{IK_CHECK_RETRIES}")
                if capture_solution and self.visualize:
                    self.display_pub.publish(DisplayRobotState(state=res.solution))
                    rospy.sleep(self.visualize_pause_s)
                return True
        rospy.loginfo_throttle(1.0, f"IK failed after {IK_CHECK_RETRIES} attempts: "
                                     f"error_code={error_code} ({moveit_error_string(error_code)})")
        return False

    def _call_once(self, pose_stamped):
        """Single compute_ik call. Returns (ok, error_code, response)."""
        self.num_calls += 1
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.link_name
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.robot_state = pre_grasp_gripper_state(self.robot_name)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.1)
        try:
            res = self.service(req)
            return res.error_code.val == 1, res.error_code.val, res
        except rospy.ServiceException as e:
            rospy.logwarn(f"IK service call failed: {e}")
            return False, None, None

    def check_grasp(self, pose_stamped):
        """Return True only if the pose AND the pre-grasp/sink/back-out/lift poses
        plan_movement.py's grasp() actually moves to all have a collision-free
        IK solution, and the straight lines between them can be followed
        completely. Mirrors simple_pick_point.py's _check_ik."""
        approach_vec = get_approach_vec(pose_stamped.pose)
        pre_grasp = offset_pose(pose_stamped, approach_vec, PRE_GRASP_OFFSET_M)
        sink = offset_pose(pose_stamped, approach_vec, -SINK_OFFSET_M)
        backout, lift = post_grasp_poses(sink, approach_vec)
        if not (self.check_single(pose_stamped, capture_solution=True)
                and self.check_single(pre_grasp)
                and self.check_single(sink)
                and self.check_single(backout)
                and self.check_single(lift)):
            return False
        ok, _ = self.motion_checker.check(pre_grasp, [sink, backout, lift])
        return ok


def find_reachable_tilt(ik, base_pose):
    """Try straight-down first; if unreachable, try the same orientation with the
    wrist rolled 180 degrees (free for a symmetric gripper -- see
    flip_approach_roll); if that also fails, fall back through the tilt table
    closest-to-straight-down first. Returns (reachable, tilt_x_deg, tilt_y_deg,
    flip) -- tilt is (0, 0) for a straight-down success, flip is True if the
    180-degree wrist roll was what worked, None/None/None if nothing in the
    table worked either. Mirrors simple_pick_point.py's
    _find_reachable_orientation, but reports which tilt/flip worked instead of
    just the resulting pose."""
    if ik.check_grasp(base_pose):
        return True, 0, 0, False

    flipped = flip_approach_roll(base_pose)
    if ik.check_grasp(flipped):
        return True, 0, 0, True

    tilts = [(dx, dy) for dx in TILT_ANGLES_DEG for dy in Y_TILT_ANGLES_DEG
             if not (dx == 0 and dy == 0)]
    tilts.sort(key=lambda t: abs(t[0]) + abs(t[1]))  # try smaller deviations first
    for dx, dy in tilts:
        candidate = tilt_pose(base_pose, dx, dy)
        if ik.check_grasp(candidate):
            return True, dx, dy, False

    return False, None, None, None


def combined_tilt_pairs(x_tilts_deg, y_tilts_deg):
    """Full cross product of the given x-tilt and y-tilt lists, deduplicated,
    straight-down (0, 0) always included, closest-to-straight-down first --
    used by --sweep-tilts to test every combination directly instead of
    stopping at find_reachable_tilt's first reachable one."""
    pairs = {(0.0, 0.0)}
    pairs.update((float(dx), float(dy)) for dx in x_tilts_deg for dy in y_tilts_deg)
    return sorted(pairs, key=lambda t: (abs(t[0]) + abs(t[1]), t))


def build_straight_down_pose(x, y, z, yaw_deg, frame_id):
    """A "straight-down" grasp means the TCP's approach axis (its local Z --
    see get_approach_vec()) points along -Z of the planning frame. roll=180 is
    what achieves that: at roll=pitch=0 the TCP frame is aligned with the
    planning frame's own axes, so its Z points along the planning frame's +Z
    (up, since panda_link0/world always has +Z up) -- flipping roll by 180
    deg points it down instead, and yaw then spins the (now downward-pointing)
    gripper around the world vertical axis, same as intended by the original
    (buggy) roll=0 version this replaces.

    NOTE: this deliberately does NOT mirror simple_pick_point.py's
    generate_grasp_pose() by building the pose in the camera frame and
    transforming it -- that only works there because the picture is taken
    from a known "looking down at the crate" arm pose at capture time. This
    script's camera is eye-in-hand (see camera_calibration_v8.launch's
    panda_hand_tcp -> camera_link static transform), so camera_frame's
    orientation relative to planning_frame moves with the arm's *current*
    joint state -- looking it up here would just reflect wherever the real
    robot happens to be parked when this script runs, not a real "down"
    direction. Constructing the down orientation directly, independent of
    any live robot/camera state, is what makes this reachability sweep a
    self-contained geometry check rather than an accidental readout of
    whatever pose the arm was left in."""
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    (pose.pose.orientation.x, pose.pose.orientation.y,
     pose.pose.orientation.z, pose.pose.orientation.w) = quaternion_from_euler(np.pi, 0, np.radians(yaw_deg))
    return pose


# --------------------------------------------------------------------------
# Crate geometry -- read live from the MoveIt planning scene instead of
# hardcoding wall coordinates, so this keeps working if the loaded .scene
# file changes (see small_crate.scene / crate_scene.scene / crate_scene_v2.scene
# in panda_moveit_config_vine_tomato/config/).
# --------------------------------------------------------------------------

def get_planning_scene_objects(robot_name, retries=5, delay_s=1.0):
    # Same namespacing issue as IKChecker's compute_ik call: PlanningSceneInterface's
    # ns argument must be the fully-qualified "/robot_name" (leading slash) since
    # get_planning_scene/apply_planning_scene otherwise resolve relative to this
    # script's own (unnamespaced, via plain rosrun) node.
    psi = moveit_commander.PlanningSceneInterface(ns=f'/{robot_name}')
    objects = {}
    for _ in range(retries):
        objects = psi.get_objects()
        if objects:
            break
        rospy.sleep(delay_s)
    if not objects:
        raise RuntimeError("No collision objects found in the planning scene -- "
                            "is the crate .scene file published (launch_moveit.launch)?")
    return objects


def box_world_pose_and_dims(collision_object):
    """Combine a CollisionObject's object-level pose with its (single) box
    primitive's pose to get the box's actual world center/rotation/dims.
    Returns None if the object isn't a box primitive (e.g. a mesh or an
    infinite shape_msgs/Plane -- some bringups add a ground-plane collision
    object that way) -- the caller is expected to skip and report those."""
    if not collision_object.primitives:
        return None
    prim = collision_object.primitives[0]
    prim_pose = collision_object.primitive_poses[0]
    obj_pose = collision_object.pose

    obj_rot = R.from_quat([obj_pose.orientation.x, obj_pose.orientation.y,
                            obj_pose.orientation.z, obj_pose.orientation.w])
    prim_pos = np.array([prim_pose.position.x, prim_pose.position.y, prim_pose.position.z])
    center = obj_rot.apply(prim_pos) + np.array([obj_pose.position.x, obj_pose.position.y, obj_pose.position.z])
    prim_rot = R.from_quat([prim_pose.orientation.x, prim_pose.orientation.y,
                             prim_pose.orientation.z, prim_pose.orientation.w])
    rot = obj_rot * prim_rot
    dims = np.array(prim.dimensions)  # SolidPrimitive.BOX_X/Y/Z order == x,y,z
    return center, rot, dims


def point_in_box(point_xyz, center, rot, dims):
    local = rot.inv().apply(np.asarray(point_xyz) - center)
    return bool(np.all(np.abs(local) <= dims / 2.0))


def compute_crate_interior(robot_name, grid_res_m, occupancy_margin_m, floor_clearance_m, top_clearance_m=None):
    """Return (x_lo, x_hi, y_lo, y_hi, test_z, walls) describing the crate's
    interior opening at the tested insertion depth.

    Method: exclude the one object that is almost certainly the world floor
    plane (named "floor", or -- if nothing matches -- whichever object has by
    far the largest footprint, since the ground plane in every .scene file
    here is 2m x 2m vs. a few tens of cm for an actual crate wall); take the
    lowest z any remaining wall box reaches as the crate floor reference;
    test a bit above that. Then rasterize the walls' footprint at that height
    into an occupancy grid and take the connected free region around the
    crate's centroid as the interior cavity -- this works for any wall
    layout (a rectangular frame, an L-shape, etc.) without hardcoding which
    box is "the left wall" vs. "the front wall".

    test_z is normally floor_clearance_m above the walls' lowest bottom (a
    deep-insertion test). Passing top_clearance_m instead tests near the rim:
    top_clearance_m below the walls' highest top -- useful for checking
    whether the crate is unreachable everywhere, or just deep down near the
    floor."""
    objects = get_planning_scene_objects(robot_name)
    boxes = {}
    for name, obj in objects.items():
        result = box_world_pose_and_dims(obj)
        if result is None:
            rospy.logwarn(f"Skipping planning-scene object '{name}' -- not a box primitive "
                           f"({len(obj.primitives)} primitives, {len(obj.meshes)} meshes, "
                           f"{len(obj.planes)} planes). Ignoring it for crate-interior detection.")
        else:
            boxes[name] = result
    if not boxes:
        raise RuntimeError(f"None of the {len(objects)} planning-scene object(s) "
                            f"({list(objects)}) are box primitives -- is the crate .scene file "
                            f"actually published? gazebo_moveit.launch does this via its "
                            f"load_crate_scene:=true publish_planning_scene node; if you launched "
                            f"move_group.launch directly instead, publish it yourself: "
                            f"rosrun moveit_ros_planning moveit_publish_scene_from_text "
                            f"$(rospack find panda_moveit_config_vine_tomato)/config/small_crate.scene")

    named_floor = [n for n in boxes if 'floor' in n.lower()]
    if named_floor:
        floor_ids = set(named_floor)
    else:
        areas = {n: dims[0] * dims[1] for n, (_, _, dims) in boxes.items()}
        floor_ids = {max(areas, key=areas.get)}
    walls = {n: v for n, v in boxes.items() if n not in floor_ids}
    if not walls:
        raise RuntimeError("Only found a floor plane in the planning scene -- no crate walls to test against.")

    wall_bottoms = [center[2] - dims[2] / 2.0 for center, _, dims in walls.values()]
    floor_z = min(wall_bottoms)
    if top_clearance_m is not None:
        wall_tops = [center[2] + dims[2] / 2.0 for center, _, dims in walls.values()]
        test_z = max(wall_tops) - top_clearance_m
    else:
        test_z = floor_z + floor_clearance_m

    centers = np.array([c for c, _, _ in walls.values()])
    dims_arr = np.array([d for _, _, d in walls.values()])
    outer_x = (centers[:, 0] - dims_arr[:, 0] / 2.0).min(), (centers[:, 0] + dims_arr[:, 0] / 2.0).max()
    outer_y = (centers[:, 1] - dims_arr[:, 1] / 2.0).min(), (centers[:, 1] + dims_arr[:, 1] / 2.0).max()

    xs = np.arange(outer_x[0] - occupancy_margin_m, outer_x[1] + occupancy_margin_m, grid_res_m)
    ys = np.arange(outer_y[0] - occupancy_margin_m, outer_y[1] + occupancy_margin_m, grid_res_m)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, test_z)], axis=1)

    occupied = np.zeros(pts.shape[0], dtype=bool)
    for center, rot, dims in walls.values():
        local = rot.inv().apply(pts - center)
        occupied |= np.all(np.abs(local) <= dims / 2.0, axis=1)
    occupied = occupied.reshape(X.shape)

    labels, _ = ndimage.label(~occupied)
    crate_centroid = centers.mean(axis=0)
    seed_i = int(np.argmin(np.abs(xs - crate_centroid[0])))
    seed_j = int(np.argmin(np.abs(ys - crate_centroid[1])))
    interior_label = labels[seed_i, seed_j]
    if interior_label == 0:
        raise RuntimeError("Crate centroid falls inside a wall's occupancy cell -- "
                            "check the loaded .scene file / occupancy_margin_m / grid_res_m.")

    interior_mask = labels == interior_label
    x_lo, x_hi = xs[np.any(interior_mask, axis=1)].min(), xs[np.any(interior_mask, axis=1)].max()
    y_lo, y_hi = ys[np.any(interior_mask, axis=0)].min(), ys[np.any(interior_mask, axis=0)].max()

    return x_lo, x_hi, y_lo, y_hi, test_z, walls


def rectangle_ring_points(x_lo, x_hi, y_lo, y_hi, spacing_m):
    """Evenly spaced points along the perimeter of the given rectangle
    (corners included, not duplicated), tagged with which wall they sit
    closest to for later grouping/plotting."""
    nx = max(2, int(round((x_hi - x_lo) / spacing_m)) + 1)
    ny = max(2, int(round((y_hi - y_lo) / spacing_m)) + 1)
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)

    points = []
    for x in xs:
        points.append((x, y_lo, "corner" if x in (x_lo, x_hi) else "y_min_wall"))
        points.append((x, y_hi, "corner" if x in (x_lo, x_hi) else "y_max_wall"))
    for y in ys[1:-1]:
        points.append((x_lo, y, "x_min_wall"))
        points.append((x_hi, y, "x_max_wall"))
    return points


def get_log_path():
    rospack = rospkg.RosPack()
    grasp_pkg_dir = rospack.get_path('grasp')
    catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pkg_dir)))
    log_dir = os.path.join(catkin_ws_dir, "experiments", "crate_reachability")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "reachability_log.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-name', default='panda')
    parser.add_argument('--edge-margin-m', type=float, default=0.02,
                         help="Inset of the test ring from the crate's interior walls (m)")
    parser.add_argument('--spacing-m', type=float, default=0.05,
                         help="Spacing between ring points along each wall (m)")
    parser.add_argument('--floor-clearance-m', type=float, default=0.02,
                         help="Insertion depth to test, measured up from the bottom of the crate walls (m). "
                              "Ignored if --top-clearance-m is given.")
    parser.add_argument('--top-clearance-m', type=float, default=None,
                         help="Test near the crate's rim instead of near the floor: this many meters "
                              "down from the walls' highest top. Overrides --floor-clearance-m.")
    parser.add_argument('--center-only', action='store_true',
                         help="Test only the single point at the crate interior's horizontal center, "
                              "instead of the full wall-hugging ring")
    parser.add_argument('--yaw-step-deg', type=float, default=45.0,
                         help="Yaw angles tested per point are 0, yaw-step, 2*yaw-step, ... up to (not including) 360")
    parser.add_argument('--grid-res-m', type=float, default=0.005,
                         help="Occupancy-grid resolution used to find the crate's interior opening (m)")
    parser.add_argument('--occupancy-margin-m', type=float, default=0.10,
                         help="Margin added around the walls' bounding box when building the occupancy grid (m)")
    parser.add_argument('--limit-points', type=int, default=None,
                         help="Only test the first N ring points (for a quick smoke test)")
    parser.add_argument('--visualize', action='store_true',
                         help="Publish each solved grasp pose's joint solution as a "
                              "DisplayRobotState (topic: /<robot-name>/display_robot_state) so "
                              "it can be watched live in RViz -- add a RobotState display there "
                              "pointed at that topic. Never commands the real/simulated robot.")
    parser.add_argument('--visualize-pause-s', type=float, default=1.0,
                         help="Seconds to pause after each solved pose when --visualize is set, "
                              "so it's actually visible before the next point overwrites it")
    parser.add_argument('--sweep-tilts', action='store_true',
                         help="Instead of the default fallback search (stop at the first "
                              "reachable tilt, closest-to-straight-down first), test every "
                              "combination of --x-tilt-deg x --y-tilt-deg directly, at both "
                              "wrist rolls (see flip_approach_roll), at each ring point/yaw "
                              "and log each one as its own row. Straight-down (0, 0) is "
                              "always included alongside the given combinations.")
    parser.add_argument('--x-tilt-deg', type=float, nargs='+', default=TILT_ANGLES_DEG,
                         help="Tilt angles (deg) around the vine axis (TCP x-axis) to cross "
                              "with --y-tilt-deg when --sweep-tilts is set. Ignored otherwise. "
                              f"Default: {TILT_ANGLES_DEG}")
    parser.add_argument('--y-tilt-deg', type=float, nargs='+', default=Y_TILT_ANGLES_DEG,
                         help="Tilt angles (deg) around the resulting gripper-closing axis "
                              "(TCP y-axis, applied after the x-tilt so it stays perpendicular "
                              "to the stem) to cross with --x-tilt-deg when --sweep-tilts is "
                              f"set. Ignored otherwise. Default: {Y_TILT_ANGLES_DEG}")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('test_crate_reachability')
    moveit_commander.roscpp_initialize(sys.argv)

    # Namespaced lookup first (gazebo_moveit.launch/launch_moveit.launch both set this
    # under /<robot_name>/planning_frame via the grasp pipeline's own launch files), the
    # unnamespaced one second in case it's ever set globally, and the obvious default last.
    planning_frame = rospy.get_param(f'/{args.robot_name}/planning_frame',
                                      rospy.get_param('/planning_frame', args.robot_name + '_link0'))

    rospy.loginfo("Reading crate geometry from the planning scene...")
    x_lo, x_hi, y_lo, y_hi, test_z, walls = compute_crate_interior(
        args.robot_name, args.grid_res_m, args.occupancy_margin_m, args.floor_clearance_m, args.top_clearance_m)
    print(f"Crate interior opening: x=[{x_lo:.3f}, {x_hi:.3f}]  y=[{y_lo:.3f}, {y_hi:.3f}]  "
          f"(from {len(walls)} wall object(s): {list(walls)})")
    if args.top_clearance_m is not None:
        print(f"Test insertion depth: z={test_z:.3f} ({args.top_clearance_m * 100:.0f}mm below the crate wall tops)")
    else:
        print(f"Test insertion depth: z={test_z:.3f} ({args.floor_clearance_m * 100:.0f}mm above the crate wall bottoms)")

    if args.center_only:
        ring_points = [((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, "center")]
    else:
        ring_x_lo, ring_x_hi = x_lo + args.edge_margin_m, x_hi - args.edge_margin_m
        ring_y_lo, ring_y_hi = y_lo + args.edge_margin_m, y_hi - args.edge_margin_m
        if ring_x_lo >= ring_x_hi or ring_y_lo >= ring_y_hi:
            raise RuntimeError(f"--edge-margin-m {args.edge_margin_m} is too large for this crate "
                                f"opening ({x_hi - x_lo:.3f} x {y_hi - y_lo:.3f} m).")
        ring_points = rectangle_ring_points(ring_x_lo, ring_x_hi, ring_y_lo, ring_y_hi, args.spacing_m)
    if args.limit_points:
        ring_points = ring_points[:args.limit_points]

    yaw_list = list(np.arange(0, 360, args.yaw_step_deg))

    tilt_pairs = None
    if args.sweep_tilts:
        tilt_pairs = combined_tilt_pairs(args.x_tilt_deg, args.y_tilt_deg)
        total = len(ring_points) * len(yaw_list) * len(tilt_pairs) * 2  # x2 for the wrist-roll flip
        print(f"{len(ring_points)} ring points x {len(yaw_list)} yaw angles x {len(tilt_pairs)} "
              f"tilt combinations x 2 wrist rolls = {total} reachability checks "
              f"(5 IK calls each: grasp + pre-grasp + sink + back-out + lift, plus a straight-line path check)\n")
    else:
        total = len(ring_points) * len(yaw_list)
        print(f"{len(ring_points)} ring points x {len(yaw_list)} yaw angles = {total} reachability checks "
              f"(5 IK calls each: grasp + pre-grasp + sink + back-out + lift, plus a straight-line path check)\n")

    ik = IKChecker(args.robot_name, visualize=args.visualize, visualize_pause_s=args.visualize_pause_s)

    log_path = get_log_path()
    fieldnames = ["timestamp", "point_idx", "x", "y", "z", "nearest_wall", "yaw_deg",
                  "reachable", "tilt_x_deg", "tilt_y_deg", "flip", "elapsed_s"]
    # Appending rows with different columns under an existing header makes the log
    # unreadable (pandas: "Expected 11 fields ... saw 12"), so a log written with
    # other columns is moved aside and a new one started.
    if os.path.exists(log_path):
        with open(log_path, newline='') as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != fieldnames:
            old_path = log_path.replace(".csv", time.strftime("_old_format_%Y%m%d_%H%M%S.csv"))
            os.rename(log_path, old_path)
            print(f"Log columns changed; moved previous log to {old_path}")
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        checked = 0

        def report_row(point_idx, x, y, nearest_wall, yaw_deg, tilt_x, tilt_y, flip, reachable, elapsed):
            nonlocal checked
            checked += 1
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "point_idx": point_idx,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(test_z, 4),
                "nearest_wall": nearest_wall,
                "yaw_deg": yaw_deg,
                "reachable": reachable,
                "tilt_x_deg": tilt_x,
                "tilt_y_deg": tilt_y,
                "flip": flip,
                "elapsed_s": round(elapsed, 3),
            })
            f.flush()

            flip_str = " flip" if flip else ""
            if reachable and tilt_x == 0 and tilt_y == 0:
                tilt_str = "straight-down" + flip_str
            elif reachable:
                tilt_str = f"tilt=({tilt_x},{tilt_y}){flip_str}"
            elif args.sweep_tilts:
                tilt_str = f"tilt=({tilt_x},{tilt_y}){flip_str} UNREACHABLE"
            else:
                tilt_str = "UNREACHABLE"
            print(f"[{checked}/{total}] point {point_idx} ({x:.3f}, {y:.3f}, {nearest_wall}) "
                  f"yaw={yaw_deg:5.1f}°  {tilt_str}  ({elapsed:.2f}s, {ik.num_calls} IK calls so far)")

        for point_idx, (x, y, nearest_wall) in enumerate(ring_points):
            for yaw_deg in yaw_list:
                base_pose = build_straight_down_pose(x, y, test_z, yaw_deg, planning_frame)

                if args.sweep_tilts:
                    # Write+print each combo immediately after it's checked, not after the
                    # whole tilt sweep for this yaw finishes -- otherwise ik.num_calls (and
                    # the printed progress) freezes for the whole batch and then jumps all at
                    # once, making already-real-time work look like it happened instantly.
                    for tilt_x, tilt_y in tilt_pairs:
                        tilted = base_pose if (tilt_x == 0 and tilt_y == 0) \
                            else tilt_pose(base_pose, tilt_x, tilt_y)
                        for flip in (False, True):
                            candidate = flip_approach_roll(tilted) if flip else tilted
                            t0 = time.time()
                            reachable = ik.check_grasp(candidate)
                            elapsed = time.time() - t0
                            report_row(point_idx, x, y, nearest_wall, yaw_deg,
                                       tilt_x, tilt_y, flip, reachable, elapsed)
                else:
                    t0 = time.time()
                    reachable, tilt_x, tilt_y, flip = find_reachable_tilt(ik, base_pose)
                    elapsed = time.time() - t0
                    report_row(point_idx, x, y, nearest_wall, yaw_deg,
                               tilt_x, tilt_y, flip, reachable, elapsed)

    print(f"\nDone. Log: {log_path}")


if __name__ == '__main__':
    main()
