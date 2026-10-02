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

import tf2_ros
import tf2_geometry_msgs

import actionlib
import franka_gripper.msg
from pilz_robot_programming import *

import copy
import os

# Speed of the straight-line descent from pre-grasp to grasp in grasp(), as fractions
# of the joint velocity/acceleration limits (other Cartesian moves default to 0.15/0.3).
GRASP_DESCENT_VELOCITY_SCALING = 0.05
GRASP_DESCENT_ACCELERATION_SCALING = 0.1
# Speed of the post-grasp back-out and straight-up lift in grasp() (Cartesian moves;
# their distances are in common/grasp_motion.py, shared with the reachability checks).
POST_GRASP_LIFT_VELOCITY_SCALING = 0.15
POST_GRASP_LIFT_ACCELERATION_SCALING = 0.3

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
        ]
        self._init_metrics_csv()
        rospy.loginfo(f"Logging planning metrics to {self.metrics_csv_path}")

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
    def planner_label(self):
        """Short label identifying the current planning configuration, for video
        filenames and the metrics CSV's planner_id column. planner_id only means
        something within the ompl pipeline, so label by pipeline_id otherwise
        (e.g. 'chomp') rather than printing a meaningless leftover ompl planner name."""
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

        elif movement == 'go_to_pre_grasp':
            # Moves only to the pre-grasp offset above the currently set grasp_pose
            # (no sink, no gripper close) -- used by "Test Pre-Grasp Positions" in the
            # GUI to visually check gripper alignment before committing to a real pick.
            # 0.05 must match the pre-grasp offset grasp() uses below.
            approach_vec = self._get_approach_vec(self.grasp_pose)
            pre_grasp_pose = copy.deepcopy(self.grasp_pose)
            pre_grasp_pose.pose.position.x -= approach_vec[0] * 0.05
            pre_grasp_pose.pose.position.y -= approach_vec[1] * 0.05
            pre_grasp_pose.pose.position.z -= approach_vec[2] * 0.05
            return self.go_to_pose(pre_grasp_pose, self.move_group_ee, allow_flip=True, movement='test_pre_grasp')

        elif movement == 'check_grasp_success':
            return self.check_grasp_success()

        goal, move_group, allow_flip = self.find_goal_pose(movement=movement)

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


    def _resolve_ik(self, goal_pose, move_group):
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
        req.ik_request.timeout = rospy.Duration(0.5)
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

    # control robot to desired goal position using only MoveIt (position controller)
    # movement/is_retry are only used for metrics + video labeling, see _log_metrics
    def go_to_pose(self, goal_pose, move_group, allow_flip=False, movement=None, is_retry=False):
        if goal_pose == None:
            return 'failure'

        self._video_call_depth += 1
        if self._video_call_depth == 1:
            label = movement or "go_to_pose"
            clip_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.planner_label}_{label}"
            self._start_recording(clip_stem)

        metrics = {field: "" for field in self._metrics_fields}
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["movement"] = movement or ""
        metrics["is_retry"] = is_retry
        metrics["planner_id"] = self.planner_label
        metrics["planning_time_limit"] = self.planning_time
        metrics["num_planning_attempts"] = self.num_planning_attempts
        metrics["video_file"] = self._active_clip_name or ""
        metrics["gripper_video_file"] = self._active_gripper_clip_name or ""

        try:
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

            if self.pipeline_id == "chomp":
                # chomp only accepts joint-space goals, so resolve the Cartesian
                # goal_pose to a joint state ourselves rather than set_pose_target.
                ik_joint_state = self._resolve_ik(goal_pose, move_group)
                if ik_joint_state is None:
                    print("IK failed for chomp goal (no collision-free solution found)")
                    metrics["planning_success"] = False
                    metrics["result"] = "failure"
                    if allow_flip and goal_pose != self.saved_pose_ee:
                        flipped_goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                        return self.go_to_pose(goal_pose=flipped_goal_pose, move_group=move_group,
                                                movement=movement, is_retry=True) #retry flipped
                    return 'failure'
                move_group.set_joint_value_target(ik_joint_state)
            else:
                move_group.set_pose_target(goal_pose)

            plan_start = time.time()
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
                plan_executed = move_group.execute(plan, wait=True)
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
                move_group.go(self.saved_joints_ee, wait=True)
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
        with open(self.metrics_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._metrics_fields).writerow(row)

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
                             velocity_scaling=0.15, acceleration_scaling=0.3):
        if goal_pose == None:
            return 'failure'

        waypoints = [move_group.get_current_pose().pose, copy.deepcopy(goal_pose.pose)]
        plan, fraction = move_group.compute_cartesian_path(waypoints, eef_step)

        print("CARTESIAN PLANNING FRACTION: ", fraction)
        if fraction < 1.0:
            return 'failure'

        # compute_cartesian_path does not time-parameterize the trajectory,
        # so it must be retimed before execution or the controller rejects it
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

        plan_executed = move_group.execute(plan, wait=True)
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

    def _get_approach_vec(self, pose_stamped):
        """Return the TCP z-axis (approach direction) in the planning frame."""
        o = pose_stamped.pose.orientation
        rot = R.from_quat([o.x, o.y, o.z, o.w])
        return rot.apply([0, 0, 1])

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
            supplied_grasp_pose = self.grasp_pose
            approach_vec = self._get_approach_vec(supplied_grasp_pose)

            if self.pre_grasp_gripper() == "success":
                # Pre-grasp: retreat along approach axis before the target
                pre_grasp_pose = copy.deepcopy(supplied_grasp_pose)
                pre_grasp_pose.pose.position.x -= approach_vec[0] * 0.05
                pre_grasp_pose.pose.position.y -= approach_vec[1] * 0.05
                pre_grasp_pose.pose.position.z -= approach_vec[2] * 0.05

                # Grasp: close exactly at the supplied grasp point (no extra sink)
                grasp_pose = copy.deepcopy(supplied_grasp_pose)

                if self.go_to_pose(goal_pose=pre_grasp_pose, move_group=self.move_group_ee, allow_flip=True, movement="grasp_pre_grasp") == "success":
                    # Slow final descent onto the truss (pre-grasp -> grasp)
                    if self.go_to_pose_cartesian(goal_pose=grasp_pose, move_group=self.move_group_ee,
                                                 velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
                                                 acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING) == "success":
                        if self.close_gripper(save=True) == "success":
                            # Back out along the approach axis to free the truss from its
                            # neighbours, then lift it straight up, keeping the grasp orientation
                            backout_pose, post_grasp_pose = post_grasp_poses(grasp_pose, approach_vec)
                            for label, target in (("back-out", backout_pose), ("lift", post_grasp_pose)):
                                if self.go_to_pose_cartesian(goal_pose=target, move_group=self.move_group_ee,
                                                             velocity_scaling=POST_GRASP_LIFT_VELOCITY_SCALING,
                                                             acceleration_scaling=POST_GRASP_LIFT_ACCELERATION_SCALING) != "success":
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
