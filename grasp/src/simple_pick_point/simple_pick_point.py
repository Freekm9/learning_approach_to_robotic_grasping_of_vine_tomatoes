import rospy
import numpy as np
import cv2
import copy
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_from_euler

from threading import Lock

from grasp.srv import pipeline_command, set_grasp_pose_command
from common.util import camera_info2rs_intrinsics, DepthImageFilter
from common.transforms import transform_pose
from common.grasp_motion import post_grasp_poses, pre_grasp_gripper_state, CartesianMotionChecker
from babuska_pick_point import BabuskaPicker

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

# Reverse lookup (int -> name) for logging failed compute_ik calls (e.g. "NO_IK_SOLUTION"
# instead of a bare int) -- diagnostic only, doesn't affect which poses count as reachable.
_ERROR_CODE_NAMES = {getattr(MoveItErrorCodes, name): name
                      for name in dir(MoveItErrorCodes)
                      if name.isupper() and isinstance(getattr(MoveItErrorCodes, name), int)}


def moveit_error_string(val):
    return _ERROR_CODE_NAMES.get(val, f"UNKNOWN({val})")

# Must match the pre-grasp/sink offsets in plan_movement.py's grasp() — IK is
# checked at these offset poses too, since a reachable grasp point doesn't guarantee
# the pre-grasp/sink poses (5cm back / 0cm further in along the approach vector) are
# reachable. The post-grasp back-out and lift come from common/grasp_motion.py. The sink offset matters most in practice: it's the
# deepest point the gripper actually travels to (plan_movement.py sinks 0cm past the
# supplied point before closing), so a candidate whose IK looks fine "above the truss"
# can still fail once the real grasp tries to move in that last bit closer.
PRE_GRASP_OFFSET_M = 0.05
SINK_OFFSET_M = 0.0

# TRAC-IK (see kinematics.yaml) does randomized internal restarts within its timeout,
# so a single NO_IK_SOLUTION on a pose near the edge of the workspace can be a false
# negative -- retry a few times before treating a pose as genuinely unreachable.
IK_CHECK_RETRIES = 3

# Diagnostic only: heights (in meters, along the approach vector, relative to the grasp
# point) sampled by _log_depth_profile when a grasp is rejected, to show exactly how far
# back the arm would need to stay to become reachable -- negative is deeper than the
# grasp point (sink direction), positive is further back/shallower (pre_grasp
# direction), in 5cm steps up to 25cm back.
DEPTH_PROBE_DISTANCES_M = sorted({-SINK_OFFSET_M, 0.0, PRE_GRASP_OFFSET_M, 0.10, 0.15, 0.20, 0.25})
DEPTH_PROBE_LABELS = {-SINK_OFFSET_M: "sink", 0.0: "grasp", PRE_GRASP_OFFSET_M: "pre_grasp"}

# Fallback orientations tried when the straight-down grasp is unreachable/in collision.
# Same angle set used for vision-based candidates (determine_grasp_candidates_manual.py).
# Tilt angles (degrees) around vine axis: 0° = top-down, ±90° = horizontal approach
TILT_ANGLES_DEG = [0, 20, -20, 40, -40, 60, -60, 80, -80]
# Roll around the (post-X-tilt) gripper-closing axis; stays perpendicular to the stem
Y_TILT_ANGLES_DEG = [0, 20, -20, 40, -40, 60, -60, 80, -80]

# How many reachable pre-grasp orientations to cycle the arm through for
# "Test Pre-Grasp Positions", and how long to hold at each one so the
# gripper's alignment on the truss can actually be seen.
MAX_PRE_GRASP_TEST_CANDIDATES = 5
PRE_GRASP_TEST_DWELL_S = 2.0


class SimplePickPoint():
    """Single-action grasping for quick testing: shows the current camera image,
    takes two clicks (grasp point, then a second point to set the stem direction),
    and drives the robot straight through set_grasp_pose + move_robot('grasp')."""

    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.bridge = CvBridge()

        self.image = None
        self.depth_image = None
        self.camera_info = None
        self.rs_intrinsics = None

        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.image_callback)
        self.depth_image_sub = rospy.Subscriber("camera/depth/image_rect_raw", Image, self.depth_image_callback)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.camera_frame = rospy.get_param('/camera_frame')
        self.planning_frame = rospy.get_param('/planning_frame')

        self.simple_pick_point_service = rospy.Service('simple_pick_point', pipeline_command, self.execute_command)
        self.test_pre_grasp_positions_service = rospy.Service('test_pre_grasp_positions', pipeline_command, self.execute_test_pre_grasp_positions)
        # Same as simple_pick_point, but the grasp point and stem direction come from
        # find_tomato_grasp (one click on the truss) instead of two manual clicks.
        self.babuska_pick_point_service = rospy.Service('babuska_pick_point', pipeline_command, self.execute_babuska_pick_point)
        # Detection only, for testing the detector: prints the grasp pose, never moves the robot.
        self.babuska_detect_service = rospy.Service('babuska_detect', pipeline_command, self.execute_babuska_detect)

        self.set_grasp_pose_service = None
        self.move_robot_service = None
        self.ik_service = None
        self.motion_checker = CartesianMotionChecker(rospy.get_param('/robot_name', 'panda'))

        self.collect_image = False
        self.collect_depth_image = False
        self.draw = False
        self.picker = "manual"  # "manual" (PickTwoPoints) or "babuska" (BabuskaPicker)
        self.points = None
        self.lock = Lock()

    def image_callback(self, image):
        if self.collect_image:
            self.image = image
            self.collect_image = False

    def depth_image_callback(self, depth_image):
        if self.collect_depth_image:
            self.depth_image = depth_image
            self.collect_depth_image = False

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)

    def execute_command(self, command, picker="manual"):
        grasp_pose = self._capture_and_pick_grasp_pose(picker)
        if grasp_pose is None:
            return "failure"

        self._log_grasp_pose(grasp_pose, "Requested grasp pose")

        grasp_pose = self._find_reachable_orientation(grasp_pose)
        if grasp_pose is None:
            print("\n⚠️ GRASP REJECTED: no reachable orientation found.\n")
            return "failure"

        return self.grasp(grasp_pose)

    def execute_babuska_pick_point(self, command):
        return self.execute_command(command, picker="babuska")

    def execute_babuska_detect(self, command):
        """Run the Babuska detector on a fresh frame and print the resulting grasp pose
        and whether it is reachable, without moving the robot."""
        grasp_pose = self._capture_and_pick_grasp_pose("babuska")
        if grasp_pose is None:
            return "failure"
        self._log_grasp_pose(grasp_pose, "Detected grasp pose")
        reachable = self._find_reachable_orientation(grasp_pose)
        if reachable is None:
            print("Detected grasp pose has no reachable orientation")
        else:
            self._log_grasp_pose(reachable, "Reachable grasp pose")
        return "success"

    def _log_grasp_pose(self, pose_stamped, label):
        """Print position + orientation (as roll/pitch/yaw, tf's static-frame/'sxyz'
        convention -- same as tf.transformations.quaternion_from_euler) for comparing
        a real pick against test_crate_reachability.py's synthetic straight-down poses
        (roll=180, pitch=0) at the same (x, y)."""
        p = pose_stamped.pose.position
        o = pose_stamped.pose.orientation
        rpy_deg = R.from_quat([o.x, o.y, o.z, o.w]).as_euler('xyz', degrees=True)
        print(f"{label} [{pose_stamped.header.frame_id}]: "
              f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
              f"rpy_deg=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f})")

    def execute_test_pre_grasp_positions(self, command):
        """Like execute_command, but instead of grasping, cycles the arm through
        every reachable pre-grasp orientation for the picked point (closest to
        straight-down first) so gripper alignment on the truss can be checked
        by eye before committing to a real pick."""
        grasp_pose = self._capture_and_pick_grasp_pose()
        if grasp_pose is None:
            return "failure"

        candidates = self._find_pre_grasp_test_candidates(grasp_pose)
        if not candidates:
            print("\n⚠️ NO REACHABLE PRE-GRASP ORIENTATIONS FOUND.\n")
            return "failure"

        print(f"Testing {len(candidates)} pre-grasp orientation(s)...")
        for i, candidate in enumerate(candidates):
            print(f"  [{i + 1}/{len(candidates)}] moving to pre-grasp candidate")
            if self._go_to_pre_grasp(candidate) != "success":
                print(f"  candidate {i + 1} failed to move, skipping")
                continue
            rospy.sleep(PRE_GRASP_TEST_DWELL_S)

        return "success"

    def _capture_and_pick_grasp_pose(self, picker="manual"):
        """Grab a fresh frame, let the user click a grasp point + stem direction
        (picker="manual") or select a truss for find_tomato_grasp (picker="babuska"),
        and return the resulting grasp pose in the planning frame. Returns None
        on any failure, having already printed why."""
        print("Grabbing frame")
        self.image, self.depth_image = None, None
        self.collect_image, self.collect_depth_image = True, True
        counter = 0
        while self.collect_image or self.collect_depth_image:
            rospy.sleep(0.1)
            counter += 1
            if counter > 10:
                print("No images coming in")
                return None

        rgb_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='rgb8')
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_image)

        # Draw from the main thread, since opencv can't handle it otherwise
        self.preprocessed_image = rgb_image
        self.picker = picker
        self.points = None
        self.draw = True
        while self.points is None:
            rospy.sleep(0.1)
        points = self.points
        self.points = None

        if not points:
            print("Pick cancelled")
            return None

        grasp_pose = self.generate_grasp_pose(points=points, depth_image=depth_image)
        if grasp_pose is None:
            print("Picked point has no valid depth")
            return None

        grasp_pose = transform_pose(grasp_pose, self.planning_frame, self.tfBuffer)
        if grasp_pose is None:
            print("Failed to transform picked point into planning frame")
            return None

        return grasp_pose

    def generate_grasp_pose(self, points, depth_image):
        (x_c, y_c), (x_d, y_d) = points
        depth_filter = DepthImageFilter(depth_image, self.rs_intrinsics, patch_size=30)
        depth = depth_filter.get_depth(y_c, x_c)
        if np.isnan(depth):
            return None
        depth = depth / 1000  # mm -> m
        xyz = depth_filter.deproject(y_c, x_c, depth=depth)
        yaw = np.arctan2(y_d - y_c, x_d - x_c)

        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = self.camera_frame
        grasp_pose.pose.position.x, grasp_pose.pose.position.y, grasp_pose.pose.position.z = xyz
        grasp_pose.pose.orientation.x, grasp_pose.pose.orientation.y, grasp_pose.pose.orientation.z, grasp_pose.pose.orientation.w = quaternion_from_euler(0, 0, yaw)
        return grasp_pose

    def grasp(self, grasp_pose):
        if self.set_grasp_pose_service is None:
            rospy.wait_for_service('set_grasp_pose', timeout=30)
            self.set_grasp_pose_service = rospy.ServiceProxy('set_grasp_pose', set_grasp_pose_command)
        if self.move_robot_service is None:
            rospy.wait_for_service('move_robot', timeout=30)
            self.move_robot_service = rospy.ServiceProxy('move_robot', pipeline_command)

        self.set_grasp_pose_service(grasp_pose)
        result = self.move_robot_service("grasp")
        return result.success

    def _get_approach_vec(self, pose):
        """Return the TCP z-axis (approach direction) in the planning frame. Mirrors plan_movement.py."""
        o = pose.orientation
        rot = R.from_quat([o.x, o.y, o.z, o.w])
        return rot.apply([0, 0, 1])

    def _offset_pose(self, pose_stamped, approach_vec, distance):
        """Return a copy of pose_stamped shifted `distance` back along approach_vec."""
        offset = copy.deepcopy(pose_stamped)
        offset.pose.position.x -= approach_vec[0] * distance
        offset.pose.position.y -= approach_vec[1] * distance
        offset.pose.position.z -= approach_vec[2] * distance
        return offset

    def _tilt_pose(self, pose_stamped, deg_x, deg_y):
        """Return a copy of pose_stamped rotated deg_x around the vine axis (TCP x-axis)
        and deg_y around the resulting gripper-closing axis (TCP y-axis). Mirrors the
        6DOF expansion in determine_grasp_candidates_manual.py."""
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

    def _flip_approach_roll(self, pose_stamped):
        """Return a copy of pose_stamped rolled 180° around its own approach axis (TCP z).
        A symmetric parallel-jaw gripper grasps the stem identically either way, but this
        puts the wrist (last joint) at a completely different angle -- which can clear a
        joint limit that the original roll couldn't, at zero cost to grasp quality."""
        o = pose_stamped.pose.orientation
        base_rot = R.from_quat([o.x, o.y, o.z, o.w])
        flipped_rot = base_rot * R.from_rotvec([0, 0, np.pi])
        new_q = flipped_rot.as_quat()

        flipped = copy.deepcopy(pose_stamped)
        flipped.pose.orientation.x, flipped.pose.orientation.y, flipped.pose.orientation.z, flipped.pose.orientation.w = new_q
        return flipped

    def _log_depth_profile(self, pose_stamped, orientation_label):
        """Print IK reachability at a range of heights along the approach vector,
        centered on the grasp point (0m) -- negative is deeper (sink direction),
        positive is further back/shallower (pre_grasp direction). Shows
        exactly how far back the arm would need to stay to become reachable, i.e.
        whether the bottleneck is depth-related at all or the point is out of reach
        even well clear of the crate."""
        approach_vec = self._get_approach_vec(pose_stamped.pose)
        print(f"  Approach-depth reachability profile ({orientation_label}, "
              f"0cm=grasp point, +=shallower/back, -=deeper):")
        for dist in DEPTH_PROBE_DISTANCES_M:
            probe = self._offset_pose(pose_stamped, approach_vec, dist)
            ok = self._check_ik_single(probe)
            label = DEPTH_PROBE_LABELS.get(dist)
            label_str = f" ({label})" if label else ""
            print(f"    {dist * 100:+5.1f}cm{label_str}: {'reachable' if ok else 'UNREACHABLE'}")

    def _find_reachable_orientation(self, grasp_pose):
        """Try the straight-down orientation first; if it's unreachable or in collision,
        try the same orientation with the wrist rolled 180° (free for a symmetric gripper,
        see _flip_approach_roll); if that also fails, fall back to other tilt/roll
        combinations (closest to straight-down first) until one clears IK, including the
        pre-grasp offset and the post-grasp back-out/lift (see _check_ik)."""
        if self._check_ik(grasp_pose, verbose=True):
            return grasp_pose

        self._log_depth_profile(grasp_pose, "straight-down")

        flipped = self._flip_approach_roll(grasp_pose)
        if self._check_ik(flipped):
            print("Found reachable orientation via 180° wrist-roll flip (straight-down)")
            return flipped

        self._log_depth_profile(flipped, "180° flip")

        print("Straight-down grasp unreachable, checking other gripper orientations...")
        tilts = [(deg_x, deg_y) for deg_x in TILT_ANGLES_DEG for deg_y in Y_TILT_ANGLES_DEG
                  if not (deg_x == 0 and deg_y == 0)]
        tilts.sort(key=lambda t: abs(t[0]) + abs(t[1]))  # try smaller deviations first
        for deg_x, deg_y in tilts:
            candidate = self._tilt_pose(grasp_pose, deg_x, deg_y)
            if self._check_ik(candidate):
                print(f"Found reachable orientation at x_tilt={deg_x}°, y_tilt={deg_y}°")
                return candidate

        return None

    def _find_pre_grasp_test_candidates(self, grasp_pose):
        """Return up to MAX_PRE_GRASP_TEST_CANDIDATES reachable orientations for
        the picked point, closest-to-straight-down first -- the same search
        _find_reachable_orientation does, except it collects every reachable
        candidate instead of stopping at the first one."""
        candidates = []
        if self._check_ik(grasp_pose):
            candidates.append(grasp_pose)

        flipped = self._flip_approach_roll(grasp_pose)
        if self._check_ik(flipped):
            candidates.append(flipped)

        tilts = [(deg_x, deg_y) for deg_x in TILT_ANGLES_DEG for deg_y in Y_TILT_ANGLES_DEG
                  if not (deg_x == 0 and deg_y == 0)]
        tilts.sort(key=lambda t: abs(t[0]) + abs(t[1]))  # closest to straight-down first
        for deg_x, deg_y in tilts:
            if len(candidates) >= MAX_PRE_GRASP_TEST_CANDIDATES:
                break
            candidate = self._tilt_pose(grasp_pose, deg_x, deg_y)
            if self._check_ik(candidate):
                candidates.append(candidate)

        return candidates

    def _go_to_pre_grasp(self, grasp_pose):
        """Set grasp_pose on plan_movement.py and move only to the pre-grasp
        offset above it (no sink, no gripper close) -- for visually checking
        gripper alignment, not for actually picking."""
        if self.set_grasp_pose_service is None:
            rospy.wait_for_service('set_grasp_pose', timeout=30)
            self.set_grasp_pose_service = rospy.ServiceProxy('set_grasp_pose', set_grasp_pose_command)
        if self.move_robot_service is None:
            rospy.wait_for_service('move_robot', timeout=30)
            self.move_robot_service = rospy.ServiceProxy('move_robot', pipeline_command)

        self.set_grasp_pose_service(grasp_pose)
        return self.move_robot_service("go_to_pre_grasp").success

    def _check_ik(self, pose_stamped, verbose=False):
        """Return True only if the grasp pose AND the pre-grasp/back-out/lift poses that
        plan_movement.py actually moves to all have a collision-free IK solution, and
        the straight lines between them (pre-grasp -> sink -> back-out -> lift, as
        executed by grasp()) can be followed completely.

        verbose=True additionally checks (and prints the failure reason for) all
        offsets instead of short-circuiting on the first failure -- only meant
        for the initial straight-down attempt, to pinpoint e.g. a sink-offset
        collision with the crate wall vs. no IK solution at all."""
        approach_vec = self._get_approach_vec(pose_stamped.pose)
        pre_grasp = self._offset_pose(pose_stamped, approach_vec, PRE_GRASP_OFFSET_M)
        sink = self._offset_pose(pose_stamped, approach_vec, -SINK_OFFSET_M)
        backout, lift = post_grasp_poses(sink, approach_vec)
        poses = [("grasp", pose_stamped), ("pre_grasp", pre_grasp), ("sink", sink),
                 ("back-out", backout), ("lift", lift)]
        if not verbose:
            return (all(self._check_ik_single(pose) for _, pose in poses)
                    and self._check_motion(pre_grasp, sink, backout, lift))

        results = {label: self._check_ik_single(pose, label=label) for label, pose in poses}
        return all(results.values()) and self._check_motion(pre_grasp, sink, backout, lift, verbose=True)

    def _check_motion(self, pre_grasp, sink, backout, lift, verbose=False):
        """Return True if grasp()'s straight-line moves, pre-grasp -> sink (descent),
        -> back-out -> lift, can each be followed completely, chained from an IK
        solution at the pre-grasp pose."""
        ok, reason = self.motion_checker.check(pre_grasp, [sink, backout, lift],
                                               labels=["grasp", "back-out", "lift"])
        if verbose and not ok:
            print(f"    straight-line motion: {reason}")
        return ok

    def _check_ik_single(self, pose_stamped, label=None):
        """Return True if MoveIt can find a collision-free IK solution for this pose,
        retrying up to IK_CHECK_RETRIES times (see its comment -- TRAC-IK's randomized
        restarts mean a single NO_IK_SOLUTION near the edge of the workspace can be a
        false negative). When label is given, print the failure reason on the final
        failed attempt -- used for the verbose straight-down diagnostic."""
        ok, error_code = False, None
        for attempt in range(IK_CHECK_RETRIES):
            ok, error_code = self._call_ik_once(pose_stamped)
            if ok:
                if label is not None and attempt > 0:
                    print(f"    {label} offset: IK succeeded on retry {attempt + 1}/{IK_CHECK_RETRIES}")
                return True
        if label is not None:
            print(f"    {label} offset: IK failed after {IK_CHECK_RETRIES} attempts "
                  f"(error_code={error_code}, {moveit_error_string(error_code)})")
        return False

    def _call_ik_once(self, pose_stamped):
        """Single compute_ik call. Returns (ok, error_code); error_code is None if the
        service itself was unavailable/errored rather than returning a MoveIt error code."""
        if self.ik_service is None:
            try:
                rospy.wait_for_service('compute_ik', timeout=2.0)
                self.ik_service = rospy.ServiceProxy('compute_ik', GetPositionIK)
            except rospy.ROSException:
                print("Warning: compute_ik not available, skipping IK filter")
                return True, None
        robot_name = rospy.get_param('/robot_name', 'panda')
        req = GetPositionIKRequest()
        req.ik_request.group_name = robot_name + "_manipulator"
        req.ik_request.ik_link_name = robot_name + "_hand_tcp"
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.robot_state = pre_grasp_gripper_state(robot_name)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.1)
        try:
            res = self.ik_service(req)
            return res.error_code.val == 1, res.error_code.val
        except rospy.ServiceException as e:
            print(f"IK service call failed: {e}")
            return False, None

    def draw_pick_point(self, rgb_image):
        if self.picker == "babuska":
            picker = BabuskaPicker()
            picker.reset(copy.deepcopy(rgb_image))
            picker.draw()
            self.points = picker.points() if picker.save else False
            return
        picker = PickTwoPoints()
        picker.reset(copy.deepcopy(rgb_image))
        picker.draw()
        if picker.save and picker.center is not None and picker.direction is not None:
            self.points = (picker.center, picker.direction)
        else:
            self.points = False


class PickTwoPoints():
    """First click sets the grasp point, second sets the stem direction from it.
    A third click starts over. Enter confirms, escape cancels."""

    def __init__(self):
        self.lock = Lock()

    def reset(self, image):
        self.image = image
        self.image_copy = copy.deepcopy(image)
        self.center = None
        self.direction = None
        self.save = False

    def click_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.direction is not None:
            self.image = copy.deepcopy(self.image_copy)
            self.center, self.direction = None, None
        elif self.center is not None:
            self.direction = (x, y)
            cv2.arrowedLine(self.image, self.center, self.direction, (255, 255, 0), 2)
        else:
            self.center = (x, y)
            cv2.circle(self.image, (x, y), 5, (255, 255, 0), -1)

    def draw(self):
        cv2.namedWindow('simple_pick_point')
        cv2.setMouseCallback('simple_pick_point', self.click_event)
        while True:
            cv2.imshow('simple_pick_point', self.image[..., ::-1])
            pressed_key = cv2.waitKey(20) & 0xFF
            if pressed_key == 27:  # Esc cancels
                break
            if pressed_key == 13 and self.direction is not None:  # Enter confirms
                self.save = True
                break
        try:
            cv2.destroyWindow('simple_pick_point')
        except cv2.error:
            print("Window already closed. Ignoring")
        cv2.waitKey(100)
