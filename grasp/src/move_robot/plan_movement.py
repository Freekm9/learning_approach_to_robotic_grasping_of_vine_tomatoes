#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import sys
import shutil
import time
import csv
import threading
import cv2
import pyrealsense2 as rs
import numpy as np
import quaternion
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, WrenchStamped
from std_msgs.msg import Float32MultiArray, Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from scipy.spatial.transform import Rotation as R

from grasp.srv import pipeline_command, pipeline_commandResponse, set_truss_data_command, set_truss_data_commandResponse, set_grasp_pose_command, set_grasp_pose_commandResponse
from common.transforms import transform_pose
from common.moveit_util import all_close, create_collision_object
from common.util import flip_z_rotation_stamped_pose
from common.grasp_motion import post_grasp_poses, PRE_GRASP_GRIPPER_WIDTH_M
from common.crate_contents import CrateContentsObstacle, crate_interior_from_boxes, CRATE_CONTENTS_ID
from common.trajectory_log import TrajectoryLog

import tf2_ros
import tf2_geometry_msgs

import actionlib
import franka_gripper.msg
from pilz_robot_programming import *

import copy
import os

# Pre-grasp offset back along the approach axis; must match simple_pick_point.py's IK checks.
PRE_GRASP_OFFSET_M = 0.05
# MoveIt pipelines that reject pose goals, so go_to_pose always resolves IK for them
JOINT_GOAL_PIPELINES = ("chomp", "stomp")
# Speed of the straight-line descent from pre-grasp to grasp in grasp(), as fractions
# of the joint velocity/acceleration limits (other Cartesian moves default to 0.15/0.3).
GRASP_DESCENT_VELOCITY_SCALING = 0.05
GRASP_DESCENT_ACCELERATION_SCALING = 0.1
# Speed of the post-grasp back-out and straight-up lift in grasp() (Cartesian moves;
# their distances are in common/grasp_motion.py, shared with the reachability checks).
POST_GRASP_LIFT_VELOCITY_SCALING = 0.15
POST_GRASP_LIFT_ACCELERATION_SCALING = 0.3
# Cartesian return to the SavePose pose (GoToSavedPose)
SAVED_POSE_CARTESIAN_VELOCITY_SCALING = 0.2
SAVED_POSE_CARTESIAN_ACCELERATION_SCALING = 0.3
# Cartesian moves of DropOff (to above the place pose, down to it, retreat up)
DROP_OFF_CARTESIAN_VELOCITY_SCALING = 0.2
DROP_OFF_CARTESIAN_ACCELERATION_SCALING = 0.3
# Goal selection for ~nearest_ik_goal: TRAC-IK (Speed mode) returns the first solution
# it finds, not the nearest, so ask several times and keep the closest one.
IK_GOAL_ATTEMPTS = 6
IK_GOAL_TIMEOUT_S = 0.1
# ~motion_method 'waypoints': straight transit to an entry pose this far back from the
# grasp point along the approach axis, then a straight approach to pre-grasp. Also the
# length of the corridor kept free of crate contents, so the approach runs through it.
WAYPOINT_ENTRY_OFFSET_M = 0.30
WAYPOINT_TRANSIT_VELOCITY_SCALING = 0.3   # as the planned moves in go_to_pose
WAYPOINT_TRANSIT_ACCELERATION_SCALING = 0.3
WAYPOINT_APPROACH_VELOCITY_SCALING = 0.15  # as go_to_pose_cartesian's default
WAYPOINT_APPROACH_ACCELERATION_SCALING = 0.3
# Luuk van den Bent's motion (~motion_method 'luuk'), from his last upstream commit
# d003c2a: Pilz LIN moves, pre-grasp straight up from the grasp point, straight descent
# to just below it (his impedance move, here on the position controller), straight-up lift.
LUUK_PRE_GRASP_HEIGHT_M = 0.03
LUUK_GRASP_SINK_M = 0.01
LUUK_POST_GRASP_HEIGHT_M = 0.25
LUUK_LIN_VELOCITY_SCALING = 0.7
LUUK_LIN_ACCELERATION_SCALING = 0.3

class Planner(object):

    def __init__(self, NODE_NAME):
        super(Planner, self).__init__()
        self.node_name = NODE_NAME

        self.succesfull_grasp_force_difference = 0.3 #0.15 for fake
        self.force_limit = 6 #Newton

        self.grasp_pose = None
        self.truss_pose = None
        self.aruco1_pose = None
        self.aruco2_pose = None
        self.force_z = None
        self.ik_service = None
        self._pink = None  # PinkMoveItPlanner, created on first use of pipeline_id 'pink'
        # Trusses in the crate as an obstacle for the planned move to pre-grasp (see
        # common/crate_contents.py and _plan_to_pre_grasp); re-read per grasp so they can
        # be toggled with rosparam set without restarting.
        self.avoid_crate_contents_default = rospy.get_param('~avoid_crate_contents', True)

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')
        self.camera_frame = rospy.get_param('/camera_frame')
        self.robot_name = rospy.get_param('/robot_name')

        # Kept the force subscriber to check grasp success
        self.force_ext_sub = rospy.Subscriber("franka_state_controller/F_ext", WrenchStamped, self.force_ext_callback, queue_size=1)

        self.move_robot_service = rospy.Service('move_robot', pipeline_command, self.plan_movement)
        self.set_truss_data_service = rospy.Service('set_truss_data', set_truss_data_command, self.set_truss_pose)
        self.set_grasp_pose_service = rospy.Service('set_grasp_pose', set_grasp_pose_command, self.set_grasp_pose)

        #MOVEIT SETUP
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.move_group = moveit_commander.MoveGroupCommander(self.robot_name+"_arm")
        self.move_group_ee = moveit_commander.MoveGroupCommander(self.robot_name+"_manipulator")
        self.move_group_ee.set_end_effector_link(self.robot_name+'_hand_tcp')
        self.move_group_camera = moveit_commander.MoveGroupCommander(self.robot_name+"_camera")

        self.gripper_grasp_action = actionlib.SimpleActionClient('franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.gripper_move_action = actionlib.SimpleActionClient('franka_gripper/move', franka_gripper.msg.MoveAction)

        # Planner under test + experiment logging, exposed as params so a benchmark
        # sweep (e.g. RRTstar vs RRTConnect vs BiTRRT, or ompl vs chomp) doesn't
        # require code edits. planner_id only applies within the ompl pipeline --
        # chomp has a single fixed algorithm (see config/chomp_planning.yaml) and
        # ignores it.
        self._default_pipeline_id = rospy.get_param('~pipeline_id', 'ompl')
        self._default_planner_id = rospy.get_param('~planner_id', 'BiTRRT')
        self.planning_time = rospy.get_param('~planning_time', 5.0)
        self.num_planning_attempts = rospy.get_param('~num_planning_attempts', 100)
        self._default_nearest_ik_goal = rospy.get_param('~nearest_ik_goal', True)
        # 'ours' or 'luuk' (Luuk van den Bent's original grasp/place motion, for comparison).
        self._default_motion_method = rospy.get_param('~motion_method', 'ours')
        # Cancellable execution for the 'luuk' straight-line moves (stop on contact)
        self.execute_trajectory_client = None

        rospack = rospkg.RosPack()
        grasp_pckg_dir = rospack.get_path('grasp')
        catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
        self.experiments_dir = os.path.join(catkin_ws_dir, "experiments", "planning_metrics")
        self.videos_dir = os.path.join(self.experiments_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)
        self.metrics_csv_path = os.path.join(self.experiments_dir, "metrics.csv")
        self._metrics_fields = [
            "timestamp", "movement", "is_retry", "planner_id", "planning_time_limit", "num_planning_attempts",
            "planning_success", "planning_duration_s", "path_length", "num_waypoints",
            "execution_success", "execution_duration_s", "goal_reached", "result",
            "video_file", "gripper_video_file",
            "goal_type", "direct_joint_distance", "replans",
        ]
        self._init_metrics_csv()
        rospy.loginfo(f"Logging planning metrics to {self.metrics_csv_path}")
        # Every executed trajectory (planned, executed joint states, crate contents), for
        # inspection and replay with replay_trajectory.py; see common/trajectory_log.py
        self.traj_log = TrajectoryLog(os.path.join(catkin_ws_dir, "experiments", "trajectories"),
                                      enabled=rospy.get_param('~log_trajectories', True))

        # Video recording of BOTH a tripod-mounted documentation camera ("side", a
        # RealSense D435) and the gripper-mounted camera ("gripper", the D405 also
        # used for truss detection), one pair of clips per outer go_to_pose call,
        # for documenting/comparing planner attempts alongside the metrics above.
        self.record_video = rospy.get_param('~record_video', True)

        # Side camera: identified by serial rather than a V4L2 index, since the D405
        # on the gripper is also a RealSense and can end up at whatever index a bare
        # V4L2 index picks, silently recording the wrong camera. Its pyrealsense2
        # pipeline is opened fresh per clip (not kept running for the node's
        # lifetime), mirroring the previous webcam-based implementation's precaution
        # against a device left open-but-idle between attempts being suspended by
        # the kernel's USB power management. Nothing else in this ROS graph already
        # streams this camera, so owning the device directly like this is fine.
        self.side_camera_serial = rospy.get_param('~video_camera_serial', '213322070505')
        self._side_video_pipeline = None
        self._side_video_writer = None
        self._side_video_thread = None
        self._side_recording_event = threading.Event()

        # Gripper camera: unlike the side camera, this one is already being streamed
        # by the camera driver node for detection/picking (simple_pick_point.py
        # subscribes to the same topic) -- opening a second raw pyrealsense2
        # pipeline against the same physical device would fight the driver for it,
        # so this recorder taps the already-published ROS topic instead of owning
        # the camera itself. Frame size comes from the first message of each clip
        # rather than being hardcoded, since it just mirrors whatever the driver is
        # already configured to publish.
        self.gripper_camera_topic = rospy.get_param('~gripper_camera_topic', 'camera/color/image_raw')
        self._gripper_bridge = CvBridge()
        self._gripper_video_writer = None
        self._gripper_clip_path = None
        self._gripper_recording_event = threading.Event()
        self._gripper_writer_lock = threading.Lock()
        self._gripper_image_sub = rospy.Subscriber(self.gripper_camera_topic, Image, self._gripper_image_callback, queue_size=1)

        # Must be one of the D435's supported discrete color-stream rates (6/15/30/60 at
        # 640x480) -- unlike a generic UVC webcam, librealsense won't loosely negotiate
        # an arbitrary fps and will fail pipeline.start() outright if it doesn't match.
        # Also used as the gripper clip's VideoWriter fps regardless of the driver's
        # actual publish rate, same simplification as the side camera already made.
        self.video_fps = rospy.get_param('~video_fps', 30)
        self._video_call_depth = 0  # only the outermost go_to_pose call (not flip-retries) opens a clip
        self._active_clip_name = None
        self._active_gripper_clip_name = None
        if self.record_video:
            self._init_side_video_capture()
        rospy.on_shutdown(self._release_video_capture)

        rospy.sleep(2) #Needed
        ceiling = create_collision_object(robot=self.robot, id='ceiling', dimensions=[2, 2, 0.02], pose=[0, 0, 1])
        wall1 = create_collision_object(robot=self.robot, id='wall1', dimensions=[0.02, 1, 1], pose=[-0.5, 0, 0.5], orientation=[0,0,0])
        wall2 = create_collision_object(robot=self.robot, id='wall2', dimensions=[0.02, 1, 1], pose=[0, -0.2, 0.5], orientation=[0,0,np.pi/2])

        #self.scene.add_object(ceiling)
        #self.scene.add_object(wall1)
        #self.scene.add_object(wall2)

        for move_group in [self.move_group, self.move_group_ee, self.move_group_camera]:
            move_group.get_current_pose()

        command = pipeline_command()
        command.command = "save_pose"
        self.plan_movement(command) #Save pose at start

    @property
    def pipeline_id(self):
        """Re-read on every access, same rationale as planner_id below: lets an
        external script switch between planning pipelines (e.g. 'ompl', 'chomp')
        between trials via `rosparam set .../pipeline_id` without restarting."""
        return rospy.get_param('~pipeline_id', self._default_pipeline_id)

    @property
    def planner_id(self):
        """Re-read on every access (rather than cached once at startup) so an external
        script can switch planners between trials via `rosparam set .../planner_id`
        without restarting this node -- restarting means re-homing/re-activating FCI,
        which is slow to do between every trial of a planner comparison."""
        return rospy.get_param('~planner_id', self._default_planner_id)

    @property
    def nearest_ik_goal(self):
        """Plan to the IK solution of the goal pose nearest the current joints instead
        of letting OMPL pick any IK solution (a far one, e.g. the wrist turning the long
        way round, is the main source of overly long paths). Re-read on every access so
        it can be switched off with rosparam set for comparison runs."""
        return rospy.get_param('~nearest_ik_goal', self._default_nearest_ik_goal)

    @property
    def motion_method(self):
        """Re-read on every access, like planner_id, so run_planner_experiment.py can
        interleave 'ours' and 'luuk' trials without restarting this node."""
        method = rospy.get_param('~motion_method', self._default_motion_method)
        if method not in ('ours', 'luuk', 'waypoints'):
            rospy.logwarn_throttle(10, f"Unknown motion_method '{method}', using 'ours'")
            return 'ours'
        return method

    @property
    def planner_label(self):
        """Short label identifying the current planning configuration, for video
        filenames and the metrics CSV's planner_id column. planner_id only means
        something within the ompl pipeline, so label by pipeline_id otherwise
        (e.g. 'chomp') rather than printing a meaningless leftover ompl planner name.
        Luuk's motion always uses Pilz LIN, so it is labelled 'luuk'; 'waypoints' rows
        (including its planner fallbacks, see goal_type) are labelled 'waypoints'."""
        if self.motion_method in ('luuk', 'waypoints'):
            return self.motion_method
        return self.planner_id if self.pipeline_id == "ompl" else self.pipeline_id

    def force_ext_callback(self, data):
        self.force_z = data.wrench.force.z

    def set_truss_pose(self, data):
        data_stamped = PoseStamped()
        data_stamped.header.frame_id = data.poses.header.frame_id
        data_stamped.pose = data.poses.poses[0]
        self.truss_pose = transform_pose(data_stamped, self.planning_frame, self.tfBuffer)
        return 'success'

    def set_grasp_pose(self, data):
        self.grasp_pose = transform_pose(data.grasp_pose, self.planning_frame, self.tfBuffer)
        return 'success'

    # Movement commands bracketing one full pick-and-place attempt for
    # documentation-video purposes: recording starts on the first of these and
    # keeps running continuously -- via the same _video_call_depth nesting
    # grasp()/go_to_pose() already use -- through every command the pipeline
    # sends in between (detection/mapping calls don't go through this method
    # at all, only actual robot-motion commands do), stopping only once one of
    # the "stop after" commands has finished. go_to_place_retreat covers the
    # crate-placement pipeline, go_to_center the non-crate one. Deliberately
    # excludes go_to_saved_pose, which resets for the *next* attempt rather
    # than being part of this one.
    ATTEMPT_RECORDING_START_COMMANDS = {'go_to_truss'}
    ATTEMPT_RECORDING_STOP_AFTER_COMMANDS = {'go_to_place_retreat', 'go_to_center'}

    def plan_movement(self, data):
        movement = data.command
        print("Planning movement command: ", movement)

        if movement in self.ATTEMPT_RECORDING_START_COMMANDS and self._video_call_depth == 0:
            self._video_call_depth += 1
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_attempt"
            self._start_recording(clip_stem)

        try:
            return self._dispatch_movement(movement)
        finally:
            if movement in self.ATTEMPT_RECORDING_STOP_AFTER_COMMANDS and self._video_call_depth > 0:
                self._video_call_depth -= 1
                if self._video_call_depth == 0:
                    self._stop_recording()

    def _dispatch_movement(self, movement):
        if movement == 'save_pose':
            self.saved_pose_ee = self.move_group_ee.get_current_pose()
            self.saved_joints_ee = self.move_group_ee.get_current_joint_values()
            print('Current pose is saved')
            return "success"

        elif movement == 'save_pose_place':
            self.saved_pose_ee_place = self.move_group_ee.get_current_pose()
            print('Current place pose is saved')
            return "success"

        elif movement == 'create_crate':
            return self.create_crate_collision_object()

        elif movement == 'open_gripper':
            return self.open_gripper()

        elif movement == 'pre_grasp_gripper':
            return self.pre_grasp_gripper()

        elif movement == 'close_gripper':
            return self.close_gripper()

        elif movement == 'grasp':
            return self.grasp()

        elif movement == 'drop_off':
            return self.drop_off()

        elif movement == 'go_to_saved_pose':
            return self.go_to_saved_pose()

        elif movement == 'go_to_pre_grasp':
            # Moves only to the pre-grasp offset above the currently set grasp_pose
            # (no sink, no gripper close) -- used by "Test Pre-Grasp Positions" in the
            # GUI to visually check gripper alignment before committing to a real pick.
            if self.motion_method == 'luuk':
                return self._go_to_pose_luuk(self._luuk_offset_z(self.grasp_pose, LUUK_PRE_GRASP_HEIGHT_M),
                                             self.move_group_ee, allow_flip=True, movement='test_pre_grasp')
            approach_vec = self._get_approach_vec(self.grasp_pose)
            result, _ = self._plan_to_pre_grasp(self.grasp_pose, approach_vec, movement='test_pre_grasp')
            return result

        elif movement == 'check_grasp_success':
            return self.check_grasp_success()

        goal, move_group, allow_flip = self.find_goal_pose(movement=movement)

        if self.motion_method == 'luuk':
            # His find_goal_pose only really set retry_with_impedance for go_to_place_above
            # (go_to_place and go_to_place_retreat had a bare `retry_with_impedance` line).
            return self._go_to_pose_luuk(goal, move_group, allow_flip=allow_flip, movement=movement,
                                         retry_straight_line=movement == 'go_to_place_above')
        return self.go_to_pose(goal, move_group, allow_flip=allow_flip, movement=movement)

    def find_goal_pose(self, movement=None):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.planning_frame
        move_group = self.move_group_ee
        allow_flip = False

        if movement == 'go_to_saved_pose':
            self.open_gripper()
            goal_pose = self.saved_pose_ee

        elif movement == 'go_to_center':
            goal_pose = copy.deepcopy(self.saved_pose_ee)
            goal_pose.pose.position.z = self.grasp_pose.pose.position.z + 0.05
            allow_flip = True

        elif movement == 'go_to_place_above':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            goal_pose.pose.position.z = self.move_group_ee.get_current_pose().pose.position.z
            allow_flip = True

        elif movement == 'go_to_place':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            allow_flip = True

        elif movement == 'go_to_place_retreat':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            goal_pose.pose.position.z = goal_pose.pose.position.z + 0.25
            allow_flip = True

        elif movement == 'go_to_truss':
            move_group = self.move_group_camera
            goal_pose = self.truss_pose
            goal_pose.pose.position.z = goal_pose.pose.position.z + 0.15 #23 #Camera ~15 cm above
            allow_flip = True

        else:
            goal_pose = move_group.get_current_pose()

        return goal_pose, move_group, allow_flip


    def _resolve_ik(self, goal_pose, move_group, timeout=0.5):
        """Resolve a Cartesian goal_pose to a joint-space RobotState via IK, seeded
        from move_group's current state. Needed for chomp: unlike ompl (which
        accepts a pose goal and samples IK itself during planning),
        chomp_interface/CHOMPPlanner only accepts joint-space goals and rejects
        position/orientation goal constraints outright ("Only joint-space goals
        are supported"). Returns None if no collision-free IK solution exists."""
        if self.ik_service is None:
            rospy.wait_for_service('compute_ik', timeout=30)
            self.ik_service = rospy.ServiceProxy('compute_ik', GetPositionIK)
        req = GetPositionIKRequest()
        req.ik_request.group_name = move_group.get_name()
        req.ik_request.ik_link_name = move_group.get_end_effector_link()
        req.ik_request.pose_stamped = goal_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state = self.robot.get_current_state()  # seed: where the arm is now
        req.ik_request.timeout = rospy.Duration(timeout)
        try:
            res = self.ik_service(req)
        except rospy.ServiceException as e:
            print(f"IK service call failed: {e}")
            return None
        if res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS:
            return None
        # compute_ik returns the whole robot's joint state (arm + gripper fingers,
        # etc.), but set_joint_value_target rejects any variable that isn't part
        # of move_group's own group -- filter down to just this group's joints.
        group_joint_names = set(move_group.get_active_joints())
        solution = res.solution.joint_state
        return {name: position for name, position in zip(solution.name, solution.position)
                if name in group_joint_names}

    def _nearest_ik_goal(self, goal_pose, move_group):
        """Joint goal for goal_pose closest (joint-space distance) to the current joints,
        over IK_GOAL_ATTEMPTS collision-free IK solutions of goal_pose itself (grasp()
        chooses between the two grasp orientations in _grasp_orientations). Returns
        (joints dict, distance) or None."""
        names = move_group.get_active_joints()
        current = np.array(move_group.get_current_joint_values())
        best = None
        for _ in range(IK_GOAL_ATTEMPTS):
            joints = self._resolve_ik(goal_pose, move_group, timeout=IK_GOAL_TIMEOUT_S)
            if joints is None:
                continue
            dist = float(np.linalg.norm(np.array([joints[n] for n in names]) - current))
            if best is None or dist < best[1]:
                best = (joints, dist)
        return best

    # control robot to desired goal position using only MoveIt (position controller)
    # movement/is_retry are only used for metrics + video labeling, see _log_metrics
    # joint_goal: IK solution of goal_pose to plan to (ompl), e.g. one already checked by
    # _grasp_orientations; by default the nearest one is looked up here (~nearest_ik_goal).
    def go_to_pose(self, goal_pose, move_group, allow_flip=False, movement=None, is_retry=False, joint_goal=None):
        if goal_pose == None:
            return 'failure'

        self._video_call_depth += 1
        if self._video_call_depth == 1:
            label = movement or "go_to_pose"
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_{label}"
            self._start_recording(clip_stem)

        metrics = self._new_metrics_row(movement, is_retry)

        try:
            # pink (pink_moveit.py) runs here in this node rather than in
            # move_group, so there is no MoveIt pipeline to select for it.
            if self.pipeline_id != "pink":
                move_group.set_planning_pipeline_id(self.pipeline_id)
            if self.pipeline_id == "ompl":
                # Asymptotically-optimal: keeps improving path cost (joint-space
                # length) over the allotted time instead of stopping at the first
                # solution like RRTConnect does
                move_group.set_planner_id(self.planner_id)
            # chomp (config/chomp_planning.yaml) has one fixed algorithm and no
            # named planner_id to select -- leave move_group's planner_id alone
            # there so it doesn't carry over a stale ompl planner name.
            move_group.set_planning_time(self.planning_time)
            move_group.set_num_planning_attempts(self.num_planning_attempts)
            #move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            #move_group.set_planner_id("LIN")
            move_group.set_max_acceleration_scaling_factor(0.3)
            move_group.set_max_velocity_scaling_factor(0.3)

            if self.pipeline_id != "pink":
                # Every MoveIt pipeline gets the same kind of goal (the nearest IK
                # solution), so a planner comparison isn't skewed by which IK branch
                # each planner happens to plan to.
                current = np.array(move_group.get_current_joint_values())
                if joint_goal is not None:
                    target = np.array([joint_goal[n] for n in move_group.get_active_joints()])
                    nearest = (joint_goal, float(np.linalg.norm(target - current)))
                elif self.nearest_ik_goal:
                    nearest = self._nearest_ik_goal(goal_pose, move_group)
                elif self.pipeline_id in JOINT_GOAL_PIPELINES:
                    ik_joint_state = self._resolve_ik(goal_pose, move_group)
                    nearest = None if ik_joint_state is None else (ik_joint_state, float(np.linalg.norm(
                        np.array([ik_joint_state[n] for n in move_group.get_active_joints()]) - current)))
                    if nearest is not None:
                        metrics["goal_type"] = "first_ik"
                else:
                    nearest = None
                if nearest is None and self.pipeline_id in JOINT_GOAL_PIPELINES:
                    # chomp/stomp only accept joint-space goals ("Only joint-space goals are
                    # supported"), so without an IK solution there is nothing to plan to
                    print(f"IK failed for {self.pipeline_id} goal (no collision-free solution found)")
                    metrics["planning_success"] = False
                    metrics["result"] = "failure"
                    if allow_flip and goal_pose != self.saved_pose_ee:
                        flipped_goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                        return self.go_to_pose(goal_pose=flipped_goal_pose, move_group=move_group,
                                                movement=movement, is_retry=True) #retry flipped
                    return 'failure'
                if nearest is not None:
                    move_group.set_joint_value_target(nearest[0])
                    metrics["goal_type"] = metrics["goal_type"] or "nearest_ik"
                    metrics["direct_joint_distance"] = nearest[1]
                else:
                    # IK found nothing (or disabled): let OMPL sample IK itself, as before
                    move_group.set_pose_target(goal_pose)
                    metrics["goal_type"] = "pose"

            plan_start = time.time()
            if self.pipeline_id == "pink":
                if self._pink is None:
                    from move_robot.pink_moveit import PinkMoveItPlanner
                    self._pink = PinkMoveItPlanner(self.robot, self.scene, self.tfBuffer)
                success, plan = self._pink.plan(goal_pose, move_group)
                if success:
                    # Same 0.3 velocity/acceleration scaling as the MoveIt pipelines above
                    plan = self._retime(move_group, plan, 0.3, 0.3)
            else:
                success, plan, _, error = move_group.plan()
            metrics["planning_duration_s"] = time.time() - plan_start
            metrics["planning_success"] = success
            if success:
                metrics["path_length"] = self._joint_path_length(plan)
                metrics["num_waypoints"] = len(plan.joint_trajectory.points)

            print("PLANNING WAS : ", success)
            if not success and goal_pose != self.saved_pose_ee:
                if allow_flip:
                    flipped_goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                    metrics["result"] = "failure"
                    return self.go_to_pose(goal_pose=flipped_goal_pose, move_group=move_group,
                                            movement=movement, is_retry=True) #retry flipped
                else:
                    metrics["result"] = "failure"
                    return 'failure'
            elif success:
                exec_start = time.time()
                with self.traj_log.recording(movement, "planned", plan=plan, **self._traj_info()) as rec:
                    plan_executed = move_group.execute(plan, wait=True)
                    rec["success"] = plan_executed
                metrics["execution_duration_s"] = time.time() - exec_start
                metrics["execution_success"] = plan_executed
                move_group.stop()
                move_group.clear_pose_targets()
                if not plan_executed:
                    print("MoveIt execution failed.")
                    metrics["result"] = "failure"
                    return 'failure'

            if goal_pose == self.saved_pose_ee: #Also reset joints when going to saved pose
                move_group.set_planner_id("PTP")
                self._go_logged(move_group, self.saved_joints_ee, movement)
                move_group.stop()
                move_group.clear_pose_targets()
                rospy.sleep(1) #sleep for camera to refocus etc

            #Test if plan succeeded
            current_pose = move_group.get_current_pose().pose
            close = all_close(goal_pose.pose, current_pose, 0.05)
            print("MOVEMENT WAS CLOSE?: ", close)
            metrics["goal_reached"] = close
            metrics["result"] = "success" if close else "failure"
            return metrics["result"]
        finally:
            self._log_metrics(metrics)
            self._video_call_depth -= 1
            if self._video_call_depth == 0:
                self._stop_recording()

    def _traj_info(self):
        return dict(planner=self.planner_label, pipeline=self.pipeline_id, video=self._active_clip_name or "")

    def _go_logged(self, move_group, joints, movement):
        """move_group.go(joints), with the target and the executed joint states logged."""
        with self.traj_log.recording(movement, "go", target=dict(zip(move_group.get_active_joints(), joints)),
                                     **self._traj_info()) as rec:
            rec["success"] = move_group.go(joints, wait=True)
        return rec.get("success")

    def _new_metrics_row(self, movement, is_retry):
        metrics = {field: "" for field in self._metrics_fields}
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["movement"] = movement or ""
        metrics["is_retry"] = is_retry
        metrics["planner_id"] = self.planner_label
        metrics["planning_time_limit"] = self.planning_time
        metrics["num_planning_attempts"] = self.num_planning_attempts
        metrics["video_file"] = self._active_clip_name or ""
        metrics["gripper_video_file"] = self._active_gripper_clip_name or ""
        return metrics

    def _joint_path_length(self, plan):
        """Sum of joint-space Euclidean distances between consecutive trajectory waypoints."""
        points = plan.joint_trajectory.points
        length = 0.0
        for p1, p2 in zip(points[:-1], points[1:]):
            length += np.linalg.norm(np.array(p2.positions) - np.array(p1.positions))
        return length

    def _init_metrics_csv(self):
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._metrics_fields).writeheader()

    def _log_metrics(self, row):
        # Logging must never fail the movement itself: an exception here propagates out of
        # the move_robot service, and the pipeline then treats a completed move as failed.
        try:
            os.makedirs(self.experiments_dir, exist_ok=True)
            write_header = not os.path.exists(self.metrics_csv_path)
            with open(self.metrics_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as e:
            print(f"Could not log planning metrics: {e}")

    def _init_side_video_capture(self):
        """Quick probe at startup so a missing/misidentified side camera shows up
        immediately instead of only failing on the first grasp attempt. Also checks
        the product name, not just that the serial is present: the gripper's D405 is
        also a RealSense, so if side_camera_serial were ever misconfigured to point
        at it, a serial-only check would happily accept it and silently record the
        gripper camera twice (once here, once via the topic-based recorder) instead
        of the tripod camera. Disabling just record_video here (rather than a
        side-only flag) is deliberate: if the identity check itself is unreliable
        enough to be wrong about which camera this is, don't trust it for the
        gripper-topic recorder's independence either -- fail closed on both."""
        device = next((d for d in rs.context().query_devices()
                        if d.get_info(rs.camera_info.serial_number) == self.side_camera_serial), None)
        if device is None:
            rospy.logwarn(f"Side camera (serial {self.side_camera_serial}) not found, disabling video recording")
            self.record_video = False
            return
        name = device.get_info(rs.camera_info.name)
        if 'D435' not in name:
            rospy.logwarn(f"Camera at serial {self.side_camera_serial} is a '{name}', not a D435 -- disabling video recording rather than risk recording from the gripper camera")
            self.record_video = False

    def _start_recording(self, clip_stem):
        """Start both the side and gripper clips for this attempt (best-effort --
        each camera's failure is independent and doesn't block the other), and
        record their filenames in self._active_clip_name / _active_gripper_clip_name."""
        self._active_clip_name = self._start_side_recording(clip_stem)
        self._active_gripper_clip_name = self._start_gripper_recording(clip_stem)

    def _stop_recording(self):
        self._stop_side_recording()
        self._stop_gripper_recording()
        self._active_clip_name = None
        self._active_gripper_clip_name = None

    def _start_side_recording(self, clip_stem):
        if not self.record_video:
            return None
        width, height = 640, 480
        self._side_video_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.side_camera_serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.video_fps)
        try:
            self._side_video_pipeline.start(config)
        except RuntimeError as e:
            rospy.logwarn(f"Side camera (serial {self.side_camera_serial}) unavailable ({e}), skipping recording for this attempt")
            self._side_video_pipeline = None
            return None
        # Auto-exposure/white-balance need a moment to settle; the first frames
        # right after opening are often black or garbage
        for _ in range(3):
            self._side_video_pipeline.wait_for_frames()

        clip_name = clip_stem + "_side.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._side_video_writer = cv2.VideoWriter(os.path.join(self.videos_dir, clip_name), fourcc, self.video_fps, (width, height))
        self._side_recording_event.set()
        self._side_video_thread = threading.Thread(target=self._side_record_loop, daemon=True)
        self._side_video_thread.start()
        return clip_name

    def _side_record_loop(self):
        consecutive_failures = 0
        while self._side_recording_event.is_set():
            try:
                frames = self._side_video_pipeline.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
            except RuntimeError:
                color_frame = None
            if color_frame:
                consecutive_failures = 0
                self._side_video_writer.write(np.asanyarray(color_frame.get_data()))
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    rospy.logwarn("Side camera stopped returning frames (device dropped?), ending this clip early")
                    break

    def _stop_side_recording(self):
        if self._side_video_pipeline is None:
            return
        self._side_recording_event.clear()
        if self._side_video_thread is not None:
            self._side_video_thread.join(timeout=2.0)
            self._side_video_thread = None
        if self._side_video_writer is not None:
            self._side_video_writer.release()
            self._side_video_writer = None
        self._side_video_pipeline.stop()
        self._side_video_pipeline = None

    def _start_gripper_recording(self, clip_stem):
        """Unlike the side camera, this doesn't own the device -- it just starts
        writing whatever arrives on self._gripper_image_callback (see the
        subscriber set up once in __init__) to a new file. The VideoWriter itself
        is created lazily on the first frame, once we actually know the driver's
        frame size, rather than assuming a fixed resolution."""
        if not self.record_video:
            return None
        clip_name = clip_stem + "_gripper.mp4"
        with self._gripper_writer_lock:
            self._gripper_clip_path = os.path.join(self.videos_dir, clip_name)
            self._gripper_recording_event.set()
        return clip_name

    def _gripper_image_callback(self, msg):
        if not self._gripper_recording_event.is_set():
            return
        try:
            frame = self._gripper_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, f"Gripper camera frame conversion failed: {e}")
            return
        with self._gripper_writer_lock:
            if not self._gripper_recording_event.is_set():
                return
            if self._gripper_video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._gripper_video_writer = cv2.VideoWriter(self._gripper_clip_path, fourcc, self.video_fps, (width, height))
            self._gripper_video_writer.write(frame)

    def _stop_gripper_recording(self):
        with self._gripper_writer_lock:
            self._gripper_recording_event.clear()
            if self._gripper_video_writer is not None:
                self._gripper_video_writer.release()
                self._gripper_video_writer = None
            self._gripper_clip_path = None

    def _release_video_capture(self):
        self._stop_recording()

    # move end effector straight to goal pose using cartesian path planning,
    # rather than joint-space (OMPL) planning, for a smooth linear approach
    def go_to_pose_cartesian(self, goal_pose, move_group, eef_step=0.005,
                             velocity_scaling=0.15, acceleration_scaling=0.3, movement="go_to_pose_cartesian"):
        if goal_pose == None:
            return 'failure'

        waypoints = [move_group.get_current_pose().pose, copy.deepcopy(goal_pose.pose)]
        plan, fraction = move_group.compute_cartesian_path(waypoints, eef_step)

        print("CARTESIAN PLANNING FRACTION: ", fraction)
        if fraction < 1.0:
            return 'failure'

        # compute_cartesian_path does not time-parameterize the trajectory,
        # so it must be retimed before execution or the controller rejects it
        plan = self._retime(move_group, plan, velocity_scaling, acceleration_scaling)

        with self.traj_log.recording(movement, "cartesian", plan=plan, **self._traj_info()) as rec:
            plan_executed = move_group.execute(plan, wait=True)
            rec["success"] = plan_executed
        move_group.stop()
        move_group.clear_pose_targets()
        if not plan_executed:
            print("MoveIt cartesian execution failed.")
            return 'failure'

        current_pose = move_group.get_current_pose().pose
        close = all_close(goal_pose.pose, current_pose, 0.05)
        print("MOVEMENT WAS CLOSE?: ", close)
        if close:
            return 'success'
        else:
            return 'failure'

    def _retime(self, move_group, plan, velocity_scaling, acceleration_scaling):
        """Time-parameterize an untimed path (Cartesian or Pink) for execution."""
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            plan,
            velocity_scaling_factor=velocity_scaling,
            acceleration_scaling_factor=acceleration_scaling,
        )

        # The retimer can leave adjacent waypoints with an identical (or
        # decreasing) time_from_start, e.g. for a near-zero-length final
        # segment. The controller rejects non-strictly-increasing timestamps,
        # so drop any waypoint that doesn't advance time.
        points = plan.joint_trajectory.points
        strictly_increasing_points = []
        last_time = -1.0
        for point in points:
            t = point.time_from_start.to_sec()
            if t > last_time:
                strictly_increasing_points.append(point)
                last_time = t
        plan.joint_trajectory.points = strictly_increasing_points
        return plan

    def _get_approach_vec(self, pose_stamped):
        """Return the TCP z-axis (approach direction) in the planning frame."""
        o = pose_stamped.pose.orientation
        rot = R.from_quat([o.x, o.y, o.z, o.w])
        return rot.apply([0, 0, 1])

    @property
    def avoid_crate_contents(self):
        return rospy.get_param('~avoid_crate_contents', self.avoid_crate_contents_default)

    def _crate_contents_obstacle(self):
        """CrateContentsObstacle for the crate currently in the planning scene."""
        from move_robot.test_crate_reachability import box_world_pose_and_dims
        boxes = []
        for name, obj in self.scene.get_objects().items():
            box = box_world_pose_and_dims(obj) if name != CRATE_CONTENTS_ID else None
            if box is not None:
                boxes.append((name,) + box)
        return CrateContentsObstacle(
            self.scene, self.tfBuffer, self.planning_frame, crate_interior_from_boxes(boxes),
            cloud_topic=rospy.get_param('~crate_contents_cloud_topic', 'camera/depth/color/points'),
            cell_m=rospy.get_param('~crate_contents_cell_m', 0.02),
            carve_radius_m=rospy.get_param('~crate_contents_carve_radius_m', 0.04),
            corridor_length_m=WAYPOINT_ENTRY_OFFSET_M,
            finger_margin_m=rospy.get_param('~crate_contents_finger_margin_m', 0.015),
            carve_mode=rospy.get_param('~crate_contents_carve_mode', 'cylinder'))

    def _capture_crate_contents(self):
        """Point cloud (planning frame) of the crate for _plan_to_pre_grasp, or None if
        disabled/unavailable -- planning then just ignores the trusses, as before."""
        if not self.avoid_crate_contents:
            return None
        try:
            return self._crate_contents_obstacle().capture_xyz()
        except Exception as e:
            rospy.logwarn(f"Crate contents: no point cloud ({e}); planning to pre-grasp without the trusses as obstacle")
            return None

    def _pre_grasp_pose(self, grasp_pose, approach_vec):
        """Pre-grasp: PRE_GRASP_OFFSET_M back along the approach axis from grasp_pose."""
        pre_grasp_pose = copy.deepcopy(grasp_pose)
        pre_grasp_pose.pose.position.x -= approach_vec[0] * PRE_GRASP_OFFSET_M
        pre_grasp_pose.pose.position.y -= approach_vec[1] * PRE_GRASP_OFFSET_M
        pre_grasp_pose.pose.position.z -= approach_vec[2] * PRE_GRASP_OFFSET_M
        return pre_grasp_pose

    def _roll_about_approach(self, pose_stamped):
        """pose_stamped turned 180° about its own approach (TCP z) axis: the same grasp
        for the symmetric parallel jaws, with the same approach direction. Unlike
        flip_z_rotation_stamped_pose (180° about the planning frame's z), this also keeps
        the approach of a tilted grasp."""
        rolled = copy.deepcopy(pose_stamped)
        o = rolled.pose.orientation
        q = (R.from_quat([o.x, o.y, o.z, o.w]) * R.from_euler('z', np.pi)).as_quat()
        o.x, o.y, o.z, o.w = q
        return rolled

    def _straight_lines_ok(self, joints, targets):
        """Whether go_to_pose_cartesian's straight lines through targets (PoseStamped, in
        order) are fully plannable when starting from the joints dict. Nothing moves."""
        move_group = self.move_group_ee
        state = moveit_msgs.msg.RobotState(is_diff=True)
        state.joint_state.name = list(joints)
        state.joint_state.position = list(joints.values())
        try:
            for target in targets:
                move_group.set_start_state(state)
                path, fraction = move_group.compute_cartesian_path([copy.deepcopy(target.pose)], 0.005)
                if fraction < 1.0 or not path.joint_trajectory.points:
                    return False
                state.joint_state.name = path.joint_trajectory.joint_names
                state.joint_state.position = list(path.joint_trajectory.points[-1].positions)
            return True
        finally:
            move_group.set_start_state_to_current_state()

    def _grasp_orientations(self, grasp_pose, approach_vec):
        """[(grasp pose, pre-grasp joint goal or None)] for grasp_pose and its 180° roll
        about the approach axis, best first. With ~nearest_ik_goal the order is: the
        straight descent, back-out and lift are plannable from the pre-grasp IK solution,
        then that solution's distance to the current joints; otherwise the supplied
        orientation first, as before. The wrist can need well over 2 rad more to reach one
        orientation than the other, which is most of the extra length of long pre-grasp
        paths."""
        options = [grasp_pose, self._roll_about_approach(grasp_pose)]
        if not self.nearest_ik_goal:
            return [(option, None) for option in options]
        ranked = []
        for option in options:
            nearest = self._nearest_ik_goal(self._pre_grasp_pose(option, approach_vec), self.move_group_ee)
            if nearest is None:
                ranked.append(((True, np.inf), option, None))
                continue
            backout_pose, post_grasp_pose = post_grasp_poses(option, approach_vec)
            lines_ok = self._straight_lines_ok(nearest[0], [option, backout_pose, post_grasp_pose])
            ranked.append(((not lines_ok, nearest[1]), option, nearest[0]))
        print("Grasp orientations (straight lines blocked, pre-grasp joint distance rad): "
              f"supplied {ranked[0][0]}, rolled 180° {ranked[1][0]}")
        ranked.sort(key=lambda r: r[0])  # stable: ties keep the supplied orientation first
        return [(option, joints) for _, option, joints in ranked]

    def _plan_to_pre_grasp(self, grasp_pose, approach_vec, movement, contents_xyz=None):
        """go_to_pose to the pre-grasp pose of grasp_pose with the crate contents (except
        a corridor around this approach) in the planning scene, so the arm doesn't sweep
        through the trusses on its way there. Removed again afterwards: the descent, grasp
        and lift happen inside the corridor, with the target truss in the gripper.

        Tries both orientations from _grasp_orientations, best first. Returns
        (result, grasp pose of the orientation reached): the descent, back-out and lift
        must use that same orientation, otherwise the straight-line descent would have to
        turn the wrist half a turn."""
        added = False
        if self.avoid_crate_contents:
            try:
                obstacle = self._crate_contents_obstacle()
                if contents_xyz is None:
                    contents_xyz = obstacle.capture_xyz()
                p, o = grasp_pose.pose.position, grasp_pose.pose.orientation
                added = obstacle.add([p.x, p.y, p.z], approach_vec, xyz=contents_xyz,
                                     grasp_quat=[o.x, o.y, o.z, o.w]) > 0
                if added:
                    self.traj_log.scene = dict(columns=[[round(v, 4) for v in c] for c in obstacle.columns],
                                               floor_z=obstacle.floor_z, cell_m=obstacle.cell_m)
            except Exception as e:
                rospy.logwarn(f"Crate contents: could not add obstacle ({e}); planning without it")
        try:
            if self.motion_method == 'waypoints':
                return self._waypoints_to_pre_grasp(grasp_pose, approach_vec, movement)
            for i, (option, joint_goal) in enumerate(self._grasp_orientations(grasp_pose, approach_vec)):
                # No allow_flip: go_to_pose's own flip would change only the pre-grasp
                # pose and not the rest of the grasp; the loop tries the other orientation.
                if self.go_to_pose(goal_pose=self._pre_grasp_pose(option, approach_vec), move_group=self.move_group_ee,
                                   allow_flip=False, movement=movement, is_retry=i > 0,
                                   joint_goal=joint_goal) == "success":
                    return "success", option
            return "failure", None
        finally:
            self.traj_log.scene = None
            if added:
                obstacle.remove()

    def _entry_pose(self, grasp_pose, approach_vec):
        """Waypoint above the crate: WAYPOINT_ENTRY_OFFSET_M back along the approach axis."""
        entry = copy.deepcopy(grasp_pose)
        entry.pose.position.x -= approach_vec[0] * WAYPOINT_ENTRY_OFFSET_M
        entry.pose.position.y -= approach_vec[1] * WAYPOINT_ENTRY_OFFSET_M
        entry.pose.position.z -= approach_vec[2] * WAYPOINT_ENTRY_OFFSET_M
        return entry

    def _cartesian_path_from(self, start_joints, target):
        """Collision-checked straight line from start_joints (dict, or None for the current
        state) to target. Returns (plan, end joints dict) or (None, None). Nothing moves."""
        move_group = self.move_group_ee
        try:
            if start_joints is not None:
                state = moveit_msgs.msg.RobotState(is_diff=True)
                state.joint_state.name = list(start_joints)
                state.joint_state.position = list(start_joints.values())
                move_group.set_start_state(state)
            else:
                move_group.set_start_state_to_current_state()
            path, fraction = move_group.compute_cartesian_path([copy.deepcopy(target.pose)], 0.005)
        finally:
            move_group.set_start_state_to_current_state()
        if fraction < 1.0 or len(path.joint_trajectory.points) < 2:
            return None, None
        end = path.joint_trajectory.points[-1].positions
        return path, dict(zip(path.joint_trajectory.joint_names, end))

    def _waypoint_routes(self, grasp_pose, approach_vec):
        """Both grasp orientations (as _grasp_orientations) with their waypoint route:
        [(grasp pose, entry pose, transit plan or None, approach ok, descent ok)], best
        first: transit and approach straight lines both feasible, then descent/back-out/
        lift feasible, then the shorter transit (joint space)."""
        routes = []
        for option in (grasp_pose, self._roll_about_approach(grasp_pose)):
            entry = self._entry_pose(option, approach_vec)
            pre_grasp = self._pre_grasp_pose(option, approach_vec)
            transit, entry_joints = self._cartesian_path_from(None, entry)
            approach, pre_joints = self._cartesian_path_from(entry_joints, pre_grasp) if transit else (None, None)
            descent_ok = False
            if approach is not None:
                backout_pose, post_grasp_pose = post_grasp_poses(option, approach_vec)
                descent_ok = self._straight_lines_ok(pre_joints, [option, backout_pose, post_grasp_pose])
            length = self._joint_path_length(transit) if transit else np.inf
            routes.append(((approach is None, not descent_ok, length), option, entry, transit, approach is not None, descent_ok))
        print("Waypoint routes (approach blocked, descent blocked, transit rad): "
              f"supplied {routes[0][0]}, rolled 180° {routes[1][0]}")
        routes.sort(key=lambda r: r[0])
        return [r[1:] for r in routes]

    def _waypoints_to_pre_grasp(self, grasp_pose, approach_vec, movement):
        """motion_method 'waypoints' (crate contents already in the scene): straight
        transit to the entry pose above the crate, then straight along the approach axis
        to pre-grasp, through the corridor kept free of crate contents. A blocked transit
        is planned with the planner instead (go_to_pose: nearest IK goal, current
        pipeline); a blocked approach falls back to planning to pre-grasp. Returns
        (result, grasp pose of the orientation used), like _plan_to_pre_grasp."""
        for i, (option, entry, transit, approach_ok, descent_ok) in enumerate(self._waypoint_routes(grasp_pose, approach_vec)):
            if not descent_ok and i > 0:
                break  # neither orientation has a feasible descent: stop after the first try
            pre_grasp = self._pre_grasp_pose(option, approach_vec)
            if transit is not None:
                reached_entry = self._run_cartesian_segment(entry, f"{movement}_transit", WAYPOINT_TRANSIT_VELOCITY_SCALING,
                                                           WAYPOINT_TRANSIT_ACCELERATION_SCALING, is_retry=i > 0) == "success"
            else:
                print("Straight transit blocked, planning to the entry pose")
                reached_entry = self.go_to_pose(goal_pose=entry, move_group=self.move_group_ee, allow_flip=False,
                                                movement=f"{movement}_transit", is_retry=i > 0) == "success"
            if not reached_entry:
                continue
            # Re-planned from where the arm actually is, so it is checked from the real state
            if self._run_cartesian_segment(pre_grasp, f"{movement}_approach", WAYPOINT_APPROACH_VELOCITY_SCALING,
                                           WAYPOINT_APPROACH_ACCELERATION_SCALING, is_retry=i > 0) == "success":
                return "success", option
            print("Straight approach blocked, planning to the pre-grasp pose")
            if self.go_to_pose(goal_pose=pre_grasp, move_group=self.move_group_ee, allow_flip=False,
                               movement=f"{movement}_approach", is_retry=True) == "success":
                return "success", option
        return "failure", None

    def _run_cartesian_segment(self, target, movement, velocity_scaling, acceleration_scaling, is_retry=False):
        """Straight line from the current pose to target (collision-checked against the
        planning scene), executed and logged to the metrics CSV like a go_to_pose move."""
        move_group = self.move_group_ee
        metrics = self._new_metrics_row(movement, is_retry)
        metrics["goal_type"] = "cartesian"
        try:
            plan_start = time.time()
            plan, _ = self._cartesian_path_from(None, target)
            metrics["planning_duration_s"] = time.time() - plan_start
            metrics["planning_success"] = plan is not None
            if plan is None:
                metrics["result"] = "failure"
                return "failure"
            metrics["path_length"] = self._joint_path_length(plan)
            metrics["num_waypoints"] = len(plan.joint_trajectory.points)
            plan = self._retime(move_group, plan, velocity_scaling, acceleration_scaling)
            exec_start = time.time()
            with self.traj_log.recording(movement, "cartesian", plan=plan, **self._traj_info()) as rec:
                executed = move_group.execute(plan, wait=True)
                rec["success"] = executed
            metrics["execution_duration_s"] = time.time() - exec_start
            metrics["execution_success"] = executed
            move_group.stop()
            close = executed and all_close(target.pose, move_group.get_current_pose().pose, 0.05)
            metrics["goal_reached"] = close
            metrics["result"] = "success" if close else "failure"
            return metrics["result"]
        finally:
            self._log_metrics(metrics)

    def grasp(self):
        # Own the recording for the whole pre-grasp -> descent -> retreat sequence as a
        # single clip (same nesting mechanism go_to_pose uses via _video_call_depth), so
        # the two go_to_pose() calls below don't each start/stop their own separate clip
        # and go_to_pose_cartesian's descent -- which never manages recording itself --
        # isn't left out of the footage in the gap between them.
        self._video_call_depth += 1
        if self._video_call_depth == 1:
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_grasp"
            self._start_recording(clip_stem)
        try:
            if self.motion_method == 'luuk':
                return self._grasp_luuk()
            supplied_grasp_pose = self.grasp_pose
            approach_vec = self._get_approach_vec(supplied_grasp_pose)
            # The arm is still at the pose the grasp point was picked from, so this
            # cloud shows the whole crate: grab it before anything moves.
            contents_xyz = self._capture_crate_contents()

            if self.pre_grasp_gripper() == "success":
                # Pre-grasp: retreat along approach axis before the target. grasp_pose is
                # the supplied grasp, possibly rolled 180° about its approach axis (same
                # grasp point and approach); the gripper closes exactly at the grasp point.
                result, grasp_pose = self._plan_to_pre_grasp(supplied_grasp_pose, approach_vec, movement="grasp_pre_grasp",
                                                             contents_xyz=contents_xyz)
                if result == "success":
                    # Slow final descent onto the truss (pre-grasp -> grasp)
                    if self.go_to_pose_cartesian(goal_pose=grasp_pose, move_group=self.move_group_ee,
                                                 velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
                                                 acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
                                                 movement="grasp_descent") == "success":
                        if self.close_gripper(save=True) == "success":
                            # Back out along the approach axis to free the truss from its
                            # neighbours, then lift it straight up, keeping the grasp orientation
                            backout_pose, post_grasp_pose = post_grasp_poses(grasp_pose, approach_vec)
                            for label, target in (("back-out", backout_pose), ("lift", post_grasp_pose)):
                                if self.go_to_pose_cartesian(goal_pose=target, move_group=self.move_group_ee,
                                                             velocity_scaling=POST_GRASP_LIFT_VELOCITY_SCALING,
                                                             acceleration_scaling=POST_GRASP_LIFT_ACCELERATION_SCALING,
                                                             movement=f"grasp_{label}") != "success":
                                    break
                            else:
                                return "success"
                            # A straight line can be cut short by IK/collisions; the truss
                            # is already in the gripper, so still get it out with the
                            # regular planner.
                            print(f"Straight-line post-grasp {label} failed, falling back to planned post-grasp move")
                            return self.go_to_pose(goal_pose=post_grasp_pose, move_group=self.move_group_ee, allow_flip=True, movement="grasp_post_grasp")
            return "failure"
        finally:
            self._video_call_depth -= 1
            if self._video_call_depth == 0:
                self._stop_recording()

    def go_to_saved_pose(self):
        # Straight-line (Cartesian) return to the SavePose pose. A straight line can be
        # blocked (crate walls, joint limits), so like grasp()'s lift it falls back to
        # the regular planner, which also does the saved-joints reset in go_to_pose().
        if getattr(self, 'saved_pose_ee', None) is None:
            print("No pose saved -- press SavePose first")
            return "failure"
        self.open_gripper()
        self._video_call_depth += 1
        if self._video_call_depth == 1:
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_go_to_saved_pose"
            self._start_recording(clip_stem)
        try:
            if self.motion_method == 'luuk':
                return self._go_to_pose_luuk(self.saved_pose_ee, self.move_group_ee, movement='go_to_saved_pose')
            if self.go_to_pose_cartesian(goal_pose=self.saved_pose_ee, move_group=self.move_group_ee,
                                         velocity_scaling=SAVED_POSE_CARTESIAN_VELOCITY_SCALING,
                                         acceleration_scaling=SAVED_POSE_CARTESIAN_ACCELERATION_SCALING,
                                         movement="go_to_saved_pose") == "success":
                # Same joint reset go_to_pose() does, so the camera view is identical each time
                self._go_logged(self.move_group_ee, self.saved_joints_ee, "go_to_saved_pose")
                self.move_group_ee.stop()
                self.move_group_ee.clear_pose_targets()
                rospy.sleep(1)  # sleep for camera to refocus etc
                return "success"
            print("Cartesian path to saved pose failed, falling back to the planner")
            return self.go_to_pose(self.saved_pose_ee, self.move_group_ee, allow_flip=False, movement='go_to_saved_pose')
        finally:
            self._video_call_depth -= 1
            if self._video_call_depth == 0:
                self._stop_recording()

    def _drop_off_move(self, movement):
        if self.motion_method == 'luuk':
            goal, move_group, allow_flip = self.find_goal_pose(movement=movement)
            return self._go_to_pose_luuk(goal, move_group, allow_flip=allow_flip, movement=movement,
                                         retry_straight_line=movement == 'go_to_place_above')
        # Straight-line (Cartesian) move for one drop_off() step, falling back to the
        # regular planner if the line is blocked (same pattern as go_to_saved_pose()).
        goal, move_group, allow_flip = self.find_goal_pose(movement=movement)
        if self.go_to_pose_cartesian(goal_pose=goal, move_group=move_group,
                                     velocity_scaling=DROP_OFF_CARTESIAN_VELOCITY_SCALING,
                                     acceleration_scaling=DROP_OFF_CARTESIAN_ACCELERATION_SCALING,
                                     movement=movement) == "success":
            return "success"
        print(f"Cartesian {movement} failed, falling back to the planner")
        return self.go_to_pose(goal, move_group, allow_flip=allow_flip, movement=movement)

    def drop_off(self):
        # Carry the held truss to the saved place pose (SavePosePlace), release it and
        # retreat: the same go_to_place_above -> go_to_place -> open_gripper ->
        # go_to_place_retreat chain the crate pipeline runs, as one command/clip so it
        # can follow Simple Pick Point. Owns the recording like grasp() does.
        if getattr(self, 'saved_pose_ee_place', None) is None:
            print("No drop-off pose saved -- move the arm there and press SavePosePlace first")
            return "failure"
        self._video_call_depth += 1
        if self._video_call_depth == 1:
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_drop_off"
            self._start_recording(clip_stem)
        try:
            for movement in ('go_to_place_above', 'go_to_place'):
                if self._drop_off_move(movement) != "success":
                    print(f"Drop-off failed at {movement}")
                    return "failure"
            if self.open_gripper() != "success":
                print("Drop-off failed to open the gripper")
                return "failure"
            return self._drop_off_move('go_to_place_retreat')
        finally:
            self._video_call_depth -= 1
            if self._video_call_depth == 0:
                self._stop_recording()

    # ---- Luuk van den Bent's motion (~motion_method 'luuk') ------------------------------
    # Ported from plan_movement.py in his last upstream commit (d003c2a) as the baseline
    # for comparing grasp/place motion. Kept as he wrote it except:
    # - his two Cartesian impedance-controller moves (the grasp descent, and the fallback
    #   when LIN can't plan go_to_place_above) are straight lines on the position
    #   (trajectory) controller instead, at his speed, stopped on contact
    #   (_straight_line_luuk); there is no controller switching;
    # - contact is a rise in F_ext z of more than force_limit (6 N) over its value at the
    #   start of the move (he compared the absolute value with 6 N);
    # - metrics + video logging like go_to_pose; a failed execution is retried once
    #   (he switched controllers away and back and retried without limit);
    # - his extra ceiling/wall collision objects are not added (both methods use the crate scene).

    def _luuk_offset_z(self, pose_stamped, dz):
        pose = copy.deepcopy(pose_stamped)
        pose.pose.position.z += dz
        return pose

    def _grasp_luuk(self):
        supplied_grasp_pose = self.grasp_pose
        if self.pre_grasp_gripper() == "success":
            pre_grasp_pose = self._luuk_offset_z(supplied_grasp_pose, LUUK_PRE_GRASP_HEIGHT_M)
            grasp_pose = self._luuk_offset_z(supplied_grasp_pose, -LUUK_GRASP_SINK_M)
            if self._go_to_pose_luuk(pre_grasp_pose, self.move_group_ee, allow_flip=True, movement="grasp_pre_grasp") == "success":
                if self._go_to_pose_luuk(grasp_pose, self.move_group_ee, mode="straight_line", movement="grasp_descent") == "success":
                    if self.close_gripper(save=True) == "success":
                        post_grasp_pose = self._luuk_offset_z(supplied_grasp_pose, LUUK_POST_GRASP_HEIGHT_M)
                        return self._go_to_pose_luuk(post_grasp_pose, self.move_group_ee, allow_flip=True, movement="grasp_post_grasp")
        return "failure"

    def _go_to_pose_luuk(self, goal_pose, move_group, mode="lin", allow_flip=False, retry_straight_line=False,
                         speed=1, movement=None, is_retry=False, execution_retries=1):
        """His go_to_pose: Pilz LIN, retried with the goal rolled 180° about the planning
        frame's z if allowed, or as a straight-line move if retry_straight_line (his
        impedance retry). mode="straight_line": _straight_line_luuk at speed (his mm per
        20 Hz step) for his impedance moves."""
        if goal_pose == None:
            return 'failure'
        self._video_call_depth += 1
        if self._video_call_depth == 1:
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_{movement or 'go_to_pose'}"
            self._start_recording(clip_stem)
        metrics = self._new_metrics_row(movement, is_retry)
        metrics["goal_type"] = f"luuk_{mode}"
        try:
            if mode == "lin":
                move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
                move_group.set_planner_id("LIN")
                move_group.set_max_acceleration_scaling_factor(LUUK_LIN_ACCELERATION_SCALING)
                move_group.set_max_velocity_scaling_factor(LUUK_LIN_VELOCITY_SCALING)
                move_group.set_pose_target(goal_pose)
                plan_start = time.time()
                success, plan, _, error = move_group.plan()
                metrics["planning_duration_s"] = time.time() - plan_start
                metrics["planning_success"] = success
                if success:
                    metrics["path_length"] = self._joint_path_length(plan)
                    metrics["num_waypoints"] = len(plan.joint_trajectory.points)
                if retry_straight_line and not success:
                    print("Could not plan LIN due to: ", error, "retrying as a straight line (his impedance retry)")
                    metrics["result"] = "failure"
                    return self._go_to_pose_luuk(goal_pose, move_group, mode="straight_line", speed=10,
                                                 movement=movement, is_retry=True)

                print("PLANNING WAS : ", success)
                if not success and goal_pose != self.saved_pose_ee:
                    metrics["result"] = "failure"
                    if allow_flip:
                        flipped_goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                        return self._go_to_pose_luuk(flipped_goal_pose, move_group, movement=movement, is_retry=True)
                    return 'failure'
                elif success:
                    exec_start = time.time()
                    with self.traj_log.recording(movement, "planned", plan=plan, **self._traj_info()) as rec:
                        plan_executed = move_group.execute(plan, wait=True)
                        rec["success"] = plan_executed
                    metrics["execution_duration_s"] = time.time() - exec_start
                    metrics["execution_success"] = plan_executed
                    move_group.stop()
                    move_group.clear_pose_targets()
                    if not plan_executed:
                        metrics["result"] = "failure"
                        if execution_retries <= 0:
                            return 'failure'
                        return self._go_to_pose_luuk(goal_pose, move_group, movement=movement, is_retry=True,
                                                     execution_retries=execution_retries - 1)

            elif mode == "straight_line":
                # As his impedance move: success once the move has run (to the goal or to
                # contact), without the arrival check below; goal_reached is still logged.
                exec_start = time.time()
                executed, plan = self._straight_line_luuk(goal_pose, move_group, speed)
                metrics["execution_duration_s"] = time.time() - exec_start
                metrics["planning_success"] = plan is not None
                if plan is not None:
                    metrics["path_length"] = self._joint_path_length(plan)
                    metrics["num_waypoints"] = len(plan.joint_trajectory.points)
                metrics["execution_success"] = executed
                metrics["goal_reached"] = all_close(goal_pose.pose, move_group.get_current_pose().pose, 0.05)
                metrics["result"] = "success" if executed else "failure"
                return metrics["result"]

            else:
                print("UNKNOWN MODE: ", mode)
                metrics["result"] = "failure"
                return 'failure'

            if goal_pose == self.saved_pose_ee:  # Also reset joints when going to saved pose
                move_group.set_planner_id("PTP")
                self._go_logged(move_group, self.saved_joints_ee, movement)
                move_group.stop()
                move_group.clear_pose_targets()
                rospy.sleep(1)  # sleep for camera to refocus etc

            current_pose = move_group.get_current_pose().pose
            close = all_close(goal_pose.pose, current_pose, 0.05)
            print("MOVEMENT WAS CLOSE?: ", close)
            metrics["goal_reached"] = close
            metrics["result"] = "success" if close else "failure"
            return metrics["result"]
        finally:
            self._log_metrics(metrics)
            self._video_call_depth -= 1
            if self._video_call_depth == 0:
                self._stop_recording()

    def _straight_line_luuk(self, goal_pose, move_group, speed=1):
        """Position-controller version of his cartesian_go_to_pose: a straight line (with
        orientation slerp) from the current pose to goal_pose that takes as long as his
        equilibrium-pose interpolation did (steps of speed mm / speed mrad at 20 Hz), and
        is stopped when contact is detected. Returns (executed, plan or None)."""
        start_pose = move_group.get_current_pose()
        if not all_close(start_pose, goal_pose, 0.1, only_check_angle=True):
            goal_pose = flip_z_rotation_stamped_pose(goal_pose)  # as in his impedance branch

        plan, fraction = move_group.compute_cartesian_path([start_pose.pose, copy.deepcopy(goal_pose.pose)], 0.005)
        print("STRAIGHT LINE PLANNING FRACTION: ", fraction)
        if fraction < 1.0 or len(plan.joint_trajectory.points) < 2:
            return False, None
        plan = self._retime(move_group, plan, LUUK_LIN_VELOCITY_SCALING, LUUK_LIN_ACCELERATION_SCALING)

        # His duration: max(distance / step, angle / step) steps at 20 Hz
        p0, p1 = start_pose.pose.position, goal_pose.pose.position
        dist = np.linalg.norm([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])
        o0, o1 = start_pose.pose.orientation, goal_pose.pose.orientation
        theta = np.arccos(np.clip(abs(o0.x*o1.x + o0.y*o1.y + o0.z*o1.z + o0.w*o1.w), -1, 1))
        step = 0.001 * speed
        target_duration = max(np.floor(dist / step), np.floor(theta / step)) / 20.0
        duration = plan.joint_trajectory.points[-1].time_from_start.to_sec()
        if duration > 0 and target_duration > duration:
            k = target_duration / duration  # slow down: stretch time, scale velocities/accelerations
            for point in plan.joint_trajectory.points:
                point.time_from_start = rospy.Duration.from_sec(point.time_from_start.to_sec() * k)
                point.velocities = [v / k for v in point.velocities]
                point.accelerations = [a / k**2 for a in point.accelerations]

        if self.execute_trajectory_client is None:
            self.execute_trajectory_client = actionlib.SimpleActionClient('execute_trajectory', moveit_msgs.msg.ExecuteTrajectoryAction)
        if not self.execute_trajectory_client.wait_for_server(rospy.Duration(5)):
            print("MoveIt execute_trajectory action not available")
            return False, plan
        force_at_start = self.force_z
        with self.traj_log.recording("luuk_straight_line", "cartesian", plan=plan, **self._traj_info()) as rec:
            self.execute_trajectory_client.send_goal(moveit_msgs.msg.ExecuteTrajectoryGoal(trajectory=plan))
            rate = rospy.Rate(20)
            while not rospy.is_shutdown():
                if self.execute_trajectory_client.wait_for_result(rospy.Duration(0.001)):
                    result = self.execute_trajectory_client.get_result()
                    rec["success"] = result is not None and result.error_code.val == moveit_msgs.msg.MoveItErrorCodes.SUCCESS
                    return rec["success"], plan
                if (force_at_start is not None and self.force_z is not None
                        and self.force_z - force_at_start > self.force_limit):
                    self.execute_trajectory_client.cancel_goal()
                    self.execute_trajectory_client.wait_for_result(rospy.Duration(2))
                    print("CONTACT DETECTED, STOPPING THE STRAIGHT LINE")
                    rec["success"], rec["stopped_on_contact"] = True, True
                    return True, plan
                rate.sleep()
            self.execute_trajectory_client.cancel_goal()
            rec["success"] = False
            return False, plan

    def open_gripper(self):
        rospy.sleep(0.5)
        force_data = rospy.wait_for_message("franka_state_controller/F_ext", WrenchStamped, timeout=5)
        if force_data is None:
            print("NO FORCE DATA INCOMING")
        else:
            self.ext_force_no_grasp = force_data.wrench.force.z
        self.gripper_move_action.wait_for_server()
        gripper = franka_gripper.msg.MoveGoal()
        gripper.width = 0.1
        gripper.speed = 0.05
        self.gripper_move_action.send_goal(gripper)
        self.gripper_move_action.wait_for_result()
        result =  self.gripper_move_action.get_result()
        return "success" if result else "failure"

    def pre_grasp_gripper(self):
        self.gripper_move_action.wait_for_server()
        gripper = franka_gripper.msg.MoveGoal()
        gripper.width = PRE_GRASP_GRIPPER_WIDTH_M
        gripper.speed = 0.1
        self.gripper_move_action.send_goal(gripper)
        self.gripper_move_action.wait_for_result()
        result =  self.gripper_move_action.get_result()
        return "success" if result else "failure"

    def close_gripper(self, save=False):
        self.gripper_grasp_action.wait_for_server()
        gripper = franka_gripper.msg.GraspGoal()
        gripper.width = 0.001
        gripper.epsilon.inner = 0.001
        gripper.epsilon.outer = 0.04
        gripper.speed = 0.005
        gripper.force = 40

        self.gripper_grasp_action.send_goal(gripper)
        self.gripper_grasp_action.wait_for_result()
        result =  self.gripper_grasp_action.get_result()

        if save:
            gripper_width = rospy.wait_for_message("franka_gripper/joint_states", JointState, timeout=5)
            gripper_width = gripper_width.position[0]
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            pointcloud_dir = os.path.join(catkin_ws_dir, "experiments/pointcloud")
            with open(os.path.join(pointcloud_dir, "gripper_width_at_grasp.txt"), 'w') as file:
                file.write(str(gripper_width))

        return "success"

    def check_grasp_success(self, save=True):
        force_data = rospy.wait_for_message("franka_state_controller/F_ext", WrenchStamped, timeout=5)
        if force_data is None:
            print("No force data coming in")
            return "failure"
        force = force_data.wrench.force.z
        result = abs(force - self.ext_force_no_grasp) > self.succesfull_grasp_force_difference
        print("FORCE BEFORE GRASP: ",self.ext_force_no_grasp, ".. FORCE AFTER GRASP: ", force, "...GRASP WAS :", result)

        if save:
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            pointcloud_dir = os.path.join(catkin_ws_dir, "experiments/pointcloud")
            pointcloud_success_dir = os.path.join(pointcloud_dir, "success")
            pointcloud_failure_dir = os.path.join(pointcloud_dir, "failure")
            if not os.path.exists(pointcloud_success_dir):
                os.makedirs(pointcloud_success_dir)
            if not os.path.exists(pointcloud_failure_dir):
                os.makedirs(pointcloud_failure_dir)
            file_list = list()
            for file in os.listdir(pointcloud_dir):
                if file.endswith(".txt") and not file.startswith("gripper"):
                    file_list.append(file)
            if len(file_list) > 1:
                print("MORE THAN ONE UNLABELED POINTCLOUD, CHECK WHAT IS GOING ON!!!")
                return "failure"
            file_path = os.path.join(pointcloud_dir, file_list[0])
            if result:
                dest_dir = pointcloud_success_dir
            else:
                dest_dir = pointcloud_failure_dir
            shutil.move(file_path, os.path.join(dest_dir, file_list[0]))
            file_path_depth_image = file_path[:-4]+".png"
            shutil.move(file_path_depth_image, os.path.join(dest_dir, file_list[0][:-4]+".png"))
            distances = np.load(os.path.join(pointcloud_dir, "distances.npy"))
            distance = distances[int(file_list[0][-5])]
            with open(os.path.join(dest_dir, file_list[0][:-4]+"distance.txt"), 'w') as file:
                file.write(str(distance))
        return "success"
