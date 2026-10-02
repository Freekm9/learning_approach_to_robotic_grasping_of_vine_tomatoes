import rospy
import os
import numpy as np
import torch
import cv2
import copy
from cv_bridge import CvBridge
import pyrealsense2 as rs
import pcl_ros
import rospkg
from scipy.spatial.transform import Rotation as R

import tf2_ros
import tf2_geometry_msgs
import tf2_sensor_msgs

import sensor_msgs.point_cloud2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from threading import Lock
import datetime

from grasp.srv import find_grasp_candidates_command
from common.transforms import find_transform, transform_pose, transform_pose_array
from common.grasp_motion import post_grasp_poses, pre_grasp_gripper_state, CartesianMotionChecker
from common.util import camera_info2rs_intrinsics, pointcloud2numpy

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

# Tilt angles (degrees) around vine axis: 0° = top-down, ±90° = horizontal approach
TILT_ANGLES_DEG = [0, 45, -45, 90, -90]
# Roll around the (post-X-tilt) gripper-closing axis; stays perpendicular to the stem
Y_TILT_ANGLES_DEG = [0, 45, -45, 90, -90]

# Must match the pre-grasp/sink offsets in plan_movement.py's grasp() — IK is
# checked at these offset poses too, since a reachable grasp point doesn't guarantee
# the pre-grasp/sink poses (5cm back / 0cm further in along the approach vector) are
# reachable. The post-grasp back-out and lift come from common/grasp_motion.py.
PRE_GRASP_OFFSET_M = 0.05
SINK_OFFSET_M = 0.0


class DetermineGraspCandidatesManual():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.detection_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')
        self.bridge = CvBridge()
        self.rs_intrinsics = None
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')
        self.motion_checker = CartesianMotionChecker(rospy.get_param('/robot_name', 'panda'))
        self.camera_frame = rospy.get_param('/camera_frame')
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.grasp_candidates_raw_debug_pub = rospy.Publisher('grasp_candidates_raw_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_debug_pub = rospy.Publisher('grasp_candidates_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_decoded_debug_pub = rospy.Publisher('grasp_candidates_decoded_debug', Image, queue_size=1, latch=True)
        self.find_grasp_candidates_manual_services = rospy.Service('find_grasp_candidates_manual', find_grasp_candidates_command, self.execute_command)
        self.grasp_candidates_pub = rospy.Publisher('grasp_candidates', PoseArray, queue_size=1, latch=True)
        self.grasp_candidates_all_debug_pub = rospy.Publisher('grasp_candidates_all_debug', PoseArray, queue_size=1, latch=True)
        self.draw = False
        self.bboxes = None
        self.lock = Lock()
        self.ik_service = None

    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)

    def execute_command(self, map):
        print("ANNOTATE GRASP POSE(S)")
        try:
            grasp_candidates = self.determine_grasp_candidates_manual(pointcloud=map.map)
            return grasp_candidates
        except Exception as e:
            print(e)
            return None

    def determine_grasp_candidates_manual(self, pointcloud):
        rgb_image, projection_history, numpy_x, numpy_y, numpy_z= self.pointcloud_to_image(pointcloud)
        #rescale
        image_size = 640
        pad_size = int((rgb_image.shape[1]-rgb_image.shape[0])/2)
        preprocessed_image = cv2.copyMakeBorder(rgb_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        image_scale = rgb_image.shape[1]/image_size
        preprocessed_image = cv2.resize(preprocessed_image, (image_size,image_size), interpolation = cv2.INTER_AREA)

        #Draw annotation from main thread, since opencv cant handle it otherwise
        self.preprocessed_image = preprocessed_image
        self.draw = True
        with self.lock:
            while self.bboxes is None:
                rospy.sleep(1)

        bboxes = copy.deepcopy(self.bboxes)
        save_annotation = copy.deepcopy(self.save_annotation)
        self.bboxes, self.save_annotation = None, None
        self.publish_debug_image(debug_image=copy.deepcopy(preprocessed_image), bboxes=copy.deepcopy(bboxes), save=save_annotation)

        if len(bboxes) == 0:
            return None

        base_candidates = PoseArray()
        base_candidates.header.frame_id = self.camera_frame
        for bbox in bboxes:
            bbox[[1,2,3,4,5,6,8,9,11,12]] *= image_scale*image_size
            bbox[[2,4,6,9,12]] -= pad_size
            grasp_pose = self.generate_grasp_pose(projection_history=projection_history, bbox=bbox, numpy_x=numpy_x, numpy_y=numpy_y, numpy_z=numpy_z)
            base_candidates.poses.append(grasp_pose)

        base_candidates = transform_pose_array(base_candidates, self.planning_frame, self.tfBuffer)

        # Expand to 6DOF candidates and filter by IK reachability
        expanded = self._expand_6dof(base_candidates)
        self.grasp_candidates_all_debug_pub.publish(expanded)  # unfiltered, for rviz debugging
        valid_poses = []
        for pose in expanded.poses:
            ps = PoseStamped()
            ps.header.frame_id = self.planning_frame
            ps.pose = pose
            if self._check_ik(ps):
                valid_poses.append(pose)
            else:
                print("\n⚠️ GRASP REJECTED: unreachable or in collision.\n")

        if not valid_poses:
            print("No valid 6DOF candidates found. Please annotate a different point.")
            return None

        expanded.poses = valid_poses
        print(f"6DOF expansion: {len(expanded.poses)} reachable candidates from {len(base_candidates.poses)} annotations")
        self.grasp_candidates_pub.publish(expanded)
        return expanded

    def _expand_6dof(self, candidates):
        """Expand each yaw-only candidate into (X tilt) x (Y tilt) variants."""
        expanded = PoseArray()
        expanded.header = candidates.header
        for pose in candidates.poses:
            q = np.array([pose.orientation.x, pose.orientation.y,
                          pose.orientation.z, pose.orientation.w])
            base_rot = R.from_quat(q)
            # Vine axis = TCP x-axis in planning frame; tilt around it to change approach angle
            vine_axis = base_rot.apply([1, 0, 0])
            vine_axis /= np.linalg.norm(vine_axis)
            for deg_x in TILT_ANGLES_DEG:
                x_tilt_rot = R.from_rotvec(np.radians(deg_x) * vine_axis)
                x_tilted_rot = x_tilt_rot * base_rot
                # Gripper-closing axis after the X tilt; rolling around it keeps the
                # gripper perpendicular to the stem while adding another DOF
                y_axis = x_tilted_rot.apply([0, 1, 0])
                y_axis /= np.linalg.norm(y_axis)
                for deg_y in Y_TILT_ANGLES_DEG:
                    y_tilt_rot = R.from_rotvec(np.radians(deg_y) * y_axis)
                    new_q = (y_tilt_rot * x_tilted_rot).as_quat()
                    new_pose = copy.deepcopy(pose)
                    new_pose.orientation.x = new_q[0]
                    new_pose.orientation.y = new_q[1]
                    new_pose.orientation.z = new_q[2]
                    new_pose.orientation.w = new_q[3]
                    expanded.poses.append(new_pose)
        return expanded

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

    def _check_ik(self, pose_stamped):
        """Return True only if the grasp pose AND the pre-grasp/back-out/lift poses that
        plan_movement.py actually moves to all have a collision-free IK solution, and
        the straight lines between them can be followed completely. A pose can be reachable on its own while
        its pre-grasp offset is inside a wall/crate or out of reach, which otherwise
        surfaces later as a MoveIt planning TIMED_OUT during the real grasp attempt."""
        approach_vec = self._get_approach_vec(pose_stamped.pose)
        pre_grasp = self._offset_pose(pose_stamped, approach_vec, PRE_GRASP_OFFSET_M)
        sink = self._offset_pose(pose_stamped, approach_vec, -SINK_OFFSET_M)
        backout, lift = post_grasp_poses(sink, approach_vec)
        if not (self._check_ik_single(pose_stamped)
                and self._check_ik_single(pre_grasp)
                and self._check_ik_single(sink)
                and self._check_ik_single(backout)
                and self._check_ik_single(lift)):
            return False
        # The straight lines grasp() follows: pre-grasp -> sink -> back-out -> lift
        ok, _ = self.motion_checker.check(pre_grasp, [sink, backout, lift])
        return ok

    def _check_ik_single(self, pose_stamped):
        """Return True if MoveIt can find a collision-free IK solution for this pose."""
        if self.ik_service is None:
            try:
                rospy.wait_for_service('compute_ik', timeout=2.0)
                self.ik_service = rospy.ServiceProxy('compute_ik', GetPositionIK)
            except rospy.ROSException:
                print("Warning: compute_ik not available, skipping IK filter")
                return True
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
            return res.error_code.val == 1
        except rospy.ServiceException as e:
            print(f"IK service call failed: {e}")
            return False

    def generate_grasp_pose(self, projection_history, bbox, numpy_x, numpy_y, numpy_z):
        grasp_center = (int(bbox[5]), int(bbox[6]))
        keypoint_left = (int(bbox[8]), int(bbox[9]))
        keypoint_right = (int(bbox[11]), int(bbox[12]))
        grasp_orientation = np.arctan2(keypoint_right[1] - keypoint_left[1], keypoint_right[0] - keypoint_left[0])

        grasp_pose = Pose()
        chosen_point = np.array(grasp_center)
        available_points = np.array(projection_history)
        index = np.sum((available_points-chosen_point)**2, axis=1, keepdims=True).argmin(axis=0)
        grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z = numpy_x[index], numpy_y[index], numpy_z[index]
        grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w = quaternion_from_euler(0, 0, grasp_orientation)
        return grasp_pose

    def pointcloud_to_image(self, pointcloud):
        transform = find_transform(pointcloud.header.frame_id, self.camera_frame, self.tfBuffer)
        pointcloud_camera_frame = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
        reconstructed_image = np.zeros((self.rs_intrinsics.height, self.rs_intrinsics.width, 3))
        numpy_x, numpy_y, numpy_z, numpy_rgb = pointcloud2numpy(pointcloud_camera_frame)

        projection_history = list()
        image_y = (numpy_y * self.rs_intrinsics.fy) / numpy_z + self.rs_intrinsics.ppy
        image_x = (numpy_x * self.rs_intrinsics.fx) / numpy_z + self.rs_intrinsics.ppx
        valid_y = np.logical_and(image_y >= 0, image_y < reconstructed_image.shape[0])
        valid_x = np.logical_and(image_x >= 0, image_x < reconstructed_image.shape[1])
        valid_z = np.logical_and(numpy_z > 0, np.isfinite(numpy_z))
        valid_idx = np.logical_and(np.logical_and(valid_y, valid_x), valid_z)
        for i in range(len(valid_idx)):
            if valid_idx[i]:
                reconstructed_image[int(image_y[i]), int(image_x[i]), :] = numpy_rgb[i]
                projection_history.append([image_x[i], image_y[i]])
        return reconstructed_image.astype(np.uint8), projection_history, numpy_x, numpy_y, numpy_z

    def publish_debug_image(self, debug_image, bboxes, save=False):
        if save:
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            image_save_dir = os.path.join(catkin_ws_dir, "data_pose/images/")
            labels_save_dir = os.path.join(catkin_ws_dir, "data_pose/labels/")
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            if not os.path.exists(labels_save_dir):
                os.makedirs(labels_save_dir)
            now = datetime.datetime.now()
            date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(image_save_dir, f"{date_time}.jpg"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            with open(os.path.join(labels_save_dir, f"{date_time}.txt"), "w") as file:
                for i, label in enumerate(bboxes):
                    file.write(' '.join(str(e) for e in label))
                    file.write('\n')

        self.grasp_candidates_raw_debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8"))
        decoded_debug_image = copy.deepcopy(debug_image)
        color_blue = (0,0,255)
        for i in range(bboxes.shape[0]):
            if i ==1:
                color_blue = (0,255,255)
            bbox = bboxes[i]
            bbox[[1,2,3,4,5,6,8,9,11,12]] *= debug_image.shape[0]
            x,y,w,h = bbox[1:5]
            debug_image = cv2.rectangle(debug_image, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (255,255,0), 2)
            x_c,y_c,x_l,y_l,x_r,y_r = int(bbox[5]), int(bbox[6]), int(bbox[8]), int(bbox[9]), int(bbox[11]), int(bbox[12])
            debug_image = cv2.circle(debug_image, (x_c,y_c), 5, color_blue, -1)
            debug_image = cv2.circle(debug_image, (x_l,y_l), 5, color_blue, -1)
            debug_image = cv2.circle(debug_image, (x_r,y_r), 5, color_blue, -1)
            p1, p2 = (x_c -(y_r-y_l), y_c +(x_r-x_l)), (x_c +(y_r-y_l), y_c -(x_r-x_l))
            decoded_debug_image = cv2.line(decoded_debug_image, p1, p2, color_blue, 2)
            decoded_debug_image = cv2.circle(decoded_debug_image, (x_c,y_c), 5, color_blue, -1)

        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        ros_decoded_image_msg = self.bridge.cv2_to_imgmsg(decoded_debug_image, encoding="rgb8")
        self.grasp_candidates_debug_pub.publish(ros_image_msg)
        self.grasp_candidates_decoded_debug_pub.publish(ros_decoded_image_msg)

    def draw_poses(self, rgb_image):
        annotate_object = DrawKeypoints(rgb_image)
        annotate_object.draw()
        self.bboxes = np.asarray(annotate_object.labels, dtype=np.float64)
        self.save_annotation = annotate_object.save

class DrawKeypoints():
    def __init__(self, image):
        self.lock = Lock()
        self.image, self.image_copy = image, copy.deepcopy(image)
        self.labels = []
        self.save = False
        self._reset()

    def _reset(self):
        self.x_c, self.y_c, self.x_l, self.y_l, self.x_r, self.y_r= None, None, None, None, None, None

    def click_event(self, event,x,y,flags,param):
        if event == cv2.EVENT_MBUTTONDOWN and self.x_c is not None and self.x_l is not None and self.x_r is not None:
            diff_x = abs(self.x_r-self.x_l)
            diff_y = abs(self.y_r-self.y_l)
            center = [int(diff_x/2+self.x_l), int(diff_y/2+min(self.y_l, self.y_r))]
            bbox_size = 1.5*max(diff_x, diff_y)
            bbox_tl_x = int(center[0]-bbox_size/2)
            bbox_tl_y = int(center[1]-bbox_size/2)
            bbox_br_x = int(center[0]+bbox_size/2)
            bbox_br_y = int(center[1]+bbox_size/2)
            cv2.rectangle(self.image, (bbox_tl_x, bbox_tl_y), (bbox_br_x,bbox_br_y), (255,255,0), 1)

            img_w = self.image.shape[1]
            img_h = self.image.shape[0]
            self.labels.append([int(0), center[0]/img_w, center[1]/img_h, bbox_size/img_w, bbox_size/img_h, self.x_c/img_w, self.y_c/img_h, 2, self.x_l/img_w, self.y_l/img_h, 2, self.x_r/img_w, self.y_r/img_h, 2])

            p1 = (int(self.x_c - (self.y_r - self.y_l)), int(self.y_c + (self.x_r - self.x_l)))
            p2 = (int(self.x_c + (self.y_r - self.y_l)), int(self.y_c - (self.x_r - self.x_l)))
            cv2.line(self.image, p1, p2, (255,0,0), 2)
            self.image_copy = copy.deepcopy(self.image)
            self._reset()
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.x_r is not None:
            self._reset()
            return

        if event == cv2.EVENT_LBUTTONDOWN and self.x_l is not None:
            self.x_r, self.y_r = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)

        elif event == cv2.EVENT_LBUTTONDOWN and self.x_c is not None:
            self.x_l, self.y_l = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.image_copy is not None:
                self.image = self.image_copy
            self.image_copy = copy.deepcopy(self.image)
            self.x_c, self.y_c = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)

    def draw(self):
        cv2.namedWindow('select_grasp_candidates')
        cv2.setMouseCallback('select_grasp_candidates', self.click_event)
        while(True):
            cv2.imshow('select_grasp_candidates', self.image[...,::-1])
            pressedKey = cv2.waitKey(20) & 0xFF
            if pressedKey == 27:
                break
            if pressedKey == 13:
                self.save = True
                break
        try:
            cv2.destroyWindow('select_grasp_candidates')
        except cv2.error:
            print("Window already closed. Ignoring")
        cv2.waitKey(100)
