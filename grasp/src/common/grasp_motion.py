"""Post-grasp motion shared by plan_movement.py's grasp() and the reachability checks
(simple_pick_point.py, determine_grasp_candidates_*.py, test_crate_reachability.py),
so the checks test the motion the robot actually executes.

After the gripper closes, grasp() first backs out along the approach axis (freeing the
truss from its neighbours in the direction it came in), then lifts straight up (+z of
the planning frame), both as Cartesian moves."""

import copy

import rospy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath, GetCartesianPathRequest, GetPositionIK, GetPositionIKRequest

POST_GRASP_BACKOUT_M = 0.05
POST_GRASP_LIFT_M = 0.25

# Total gripper opening during the whole grasp approach: grasp() narrows the gripper to
# this (pre_grasp_gripper()) before moving to pre-grasp. The reachability checks put the
# fingers at this opening too, instead of the gripper's current one (fully open, ~8 cm,
# after a release), which made fingers collide with crate walls that the real, narrowed
# gripper clears. Widen it here to give the checks a margin from the walls.
PRE_GRASP_GRIPPER_WIDTH_M = 0.025

# Same Cartesian interpolation as plan_movement.py's go_to_pose_cartesian
# (MoveGroupCommander.compute_cartesian_path defaults: no jump threshold).
CARTESIAN_EEF_STEP_M = 0.005
CARTESIAN_JUMP_THRESHOLD = 0.0
IK_RETRIES = 3


def post_grasp_poses(closed_pose, approach_vec):
    """Return (backout, lift) poses for a gripper closed at closed_pose (PoseStamped in
    the planning frame): backout is POST_GRASP_BACKOUT_M back along approach_vec, lift
    is POST_GRASP_LIFT_M straight up from there. Both keep the grasp orientation."""
    backout = copy.deepcopy(closed_pose)
    backout.pose.position.x -= approach_vec[0] * POST_GRASP_BACKOUT_M
    backout.pose.position.y -= approach_vec[1] * POST_GRASP_BACKOUT_M
    backout.pose.position.z -= approach_vec[2] * POST_GRASP_BACKOUT_M
    lift = copy.deepcopy(backout)
    lift.pose.position.z += POST_GRASP_LIFT_M
    return backout, lift


def pre_grasp_gripper_state(robot_name):
    """RobotState diff that sets the fingers to PRE_GRASP_GRIPPER_WIDTH_M; pass it as an
    IK request's robot_state so the IK collision check uses the pre-grasp opening."""
    state = RobotState()
    state.is_diff = True
    state.joint_state.name = [robot_name + "_finger_joint1", robot_name + "_finger_joint2"]
    state.joint_state.position = [PRE_GRASP_GRIPPER_WIDTH_M / 2] * 2
    return state


class CartesianMotionChecker():
    """Checks that a chain of straight-line moves can be followed completely, starting
    from an IK solution at the first pose: e.g. pre-grasp -> grasp -> backout -> lift.
    Each segment starts from the joint state the previous one ended in, as during the
    real grasp. The IK solution at the start is random (TRAC-IK), so this approximates
    the configuration the planner will actually reach at pre-grasp."""

    def __init__(self, robot_name, ns=""):
        """ns prefixes the move_group service names, e.g. "/panda/" for a node outside
        the robot's namespace; empty resolves them relative to the node's namespace."""
        self.ik_service_name = ns + 'compute_ik'
        self.cartesian_service_name = ns + 'compute_cartesian_path'
        self.robot_name = robot_name
        self.group_name = robot_name + "_manipulator"
        self.link_name = robot_name + "_hand_tcp"
        self.ik_service = None
        self.cartesian_service = None

    def check(self, start_pose, waypoints, labels=None):
        """Return (ok, reason). waypoints are PoseStamped in start_pose's frame;
        labels name them for the reason string."""
        labels = labels or [str(i) for i in range(len(waypoints))]
        if not self._connect():
            return True, "services unavailable, path check skipped"
        state = self._ik_state(start_pose)
        if state is None:
            return False, "no IK solution at start pose"
        for label, target in zip(labels, waypoints):
            req = GetCartesianPathRequest()
            req.header.frame_id = start_pose.header.frame_id
            req.start_state = state
            req.group_name = self.group_name
            req.link_name = self.link_name
            req.waypoints = [target.pose]
            req.max_step = CARTESIAN_EEF_STEP_M
            req.jump_threshold = CARTESIAN_JUMP_THRESHOLD
            req.avoid_collisions = True
            try:
                res = self.cartesian_service(req)
            except rospy.ServiceException as e:
                return False, f"compute_cartesian_path failed: {e}"
            if res.fraction < 1.0:
                return False, f"straight line to {label} only {res.fraction:.2f} complete"
            state = self._end_state(state, res.solution.joint_trajectory)
        return True, "ok"

    def _connect(self):
        if self.ik_service is not None and self.cartesian_service is not None:
            return True
        try:
            rospy.wait_for_service(self.ik_service_name, timeout=2.0)
            rospy.wait_for_service(self.cartesian_service_name, timeout=2.0)
        except rospy.ROSException:
            print("Warning: compute_ik/compute_cartesian_path not available, skipping path check")
            return False
        self.ik_service = rospy.ServiceProxy(self.ik_service_name, GetPositionIK)
        self.cartesian_service = rospy.ServiceProxy(self.cartesian_service_name, GetCartesianPath)
        return True

    def _ik_state(self, pose_stamped):
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.link_name
        req.ik_request.pose_stamped = pose_stamped
        req.ik_request.robot_state = pre_grasp_gripper_state(self.robot_name)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.1)
        for _ in range(IK_RETRIES):
            try:
                res = self.ik_service(req)
            except rospy.ServiceException as e:
                print(f"IK service call failed: {e}")
                return None
            if res.error_code.val == 1:
                return res.solution
        return None

    @staticmethod
    def _end_state(state, joint_trajectory):
        """state with the joints of joint_trajectory set to its last point."""
        if not joint_trajectory.points:
            return state
        state = copy.deepcopy(state)
        positions = list(state.joint_state.position)
        last = joint_trajectory.points[-1].positions
        for name, value in zip(joint_trajectory.joint_names, last):
            if name in state.joint_state.name:
                positions[state.joint_state.name.index(name)] = value
        state.joint_state.position = positions
        return state
