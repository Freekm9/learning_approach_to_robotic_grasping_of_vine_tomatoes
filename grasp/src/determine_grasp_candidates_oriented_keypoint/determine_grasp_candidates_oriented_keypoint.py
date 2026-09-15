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

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from grasp.srv import find_grasp_candidates_command
from common.transforms import find_transform, transform_pose, transform_pose_array
from common.util import camera_info2rs_intrinsics, pointcloud2numpy, pointcloud2image
from common.download_model import download_from_google_drive

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

# Tilt angles (degrees) around vine axis: 0° = top-down, ±90° = horizontal approach
TILT_ANGLES_DEG = [0, 45, -45, 90, -90]
# Roll around the (post-X-tilt) gripper-closing axis; stays perpendicular to the stem
Y_TILT_ANGLES_DEG = [0, 45, -45, 90, -90]

# Must match the pre-grasp/sink/retreat offsets in plan_movement.py's grasp() — IK is
# checked at these offset poses too, since a reachable grasp point doesn't guarantee
# the pre-grasp/sink/retreat poses (5cm back / 1cm further in / 25cm back along the
# approach vector) are reachable. The sink offset matters most in practice: it's the
# deepest point the gripper actually travels to (plan_movement.py sinks 1cm past the
# supplied point before closing), so a candidate whose IK looks fine "above the truss"
# can still fail once the real grasp tries to move in that last bit closer.
PRE_GRASP_OFFSET_M = 0.05
SINK_OFFSET_M = 0.01
RETREAT_OFFSET_M = 0.25


class DetermineGraspCandidatesOrientedKeypoint():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.detection_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')#todo
        self.bridge = CvBridge()
        self.rs_intrinsics = None
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')
        self.camera_frame = rospy.get_param('/camera_frame')
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.grasp_candidates_raw_debug_pub = rospy.Publisher('grasp_candidates_raw_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_debug_pub = rospy.Publisher('grasp_candidates_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_decoded_debug_pub = rospy.Publisher('grasp_candidates_decoded_debug', Image, queue_size=1, latch=True)
        self.find_grasp_candidates_oriented_keypoint_services = rospy.Service('find_grasp_candidates_oriented_keypoint', find_grasp_candidates_command, self.execute_command)
        self.grasp_candidates_pub = rospy.Publisher('grasp_candidates', PoseArray, queue_size=1, latch=True)
        self.ik_service = None

        # file_id = "1P_ycIIrk-8BNsBZQKjOKzg8xTvoUB7yQ"
        # download_from_google_drive(file_id, self.detection_model_path)

        self.model = torch.hub.load(os.path.dirname(os.path.realpath(__file__)), 'custom', path_or_model=self.detection_model_path, source='local', force_reload=True)
        self.model.conf = 0.25
        self.model.iou = 0.45

    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)

    def execute_command(self, map):
        print("CALCULATING GRASP CANDIDATES")
        try:
            grasp_candidates = self.determine_grasp_candidates_oriented_keypoint(pointcloud=map.map)
            return grasp_candidates
        except Exception as e:
            print(e)
            return None

    def determine_grasp_candidates_oriented_keypoint(self, pointcloud):
        transform = find_transform(pointcloud.header.frame_id, self.camera_frame, self.tfBuffer)
        pointcloud_camera_frame = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
        rgb_image, projection_history, numpy_x, numpy_y, numpy_z= pointcloud2image(pointcloud_camera_frame, self.rs_intrinsics)
        #rescale
        image_size = 640
        pad_size = int((rgb_image.shape[1]-rgb_image.shape[0])/2)
        preprocessed_image = cv2.copyMakeBorder(rgb_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        image_scale = rgb_image.shape[1]/image_size
        preprocessed_image = cv2.resize(preprocessed_image, (image_size,image_size), interpolation = cv2.INTER_AREA)

        #Run model
        results = self.model(preprocessed_image)
        bboxes = results.pred[0].cpu().numpy()

        self.publish_debug_image(debug_image=copy.deepcopy(preprocessed_image), bboxes=copy.deepcopy(bboxes))
        if len(bboxes) == 0:
            return None

        # Build base candidates (yaw-only, one per detection)
        base_candidates = PoseArray()
        base_candidates.header.frame_id = self.camera_frame
        for bbox in bboxes:
            bbox[[0,1,2,3,6,7]] *= image_scale
            bbox[[1,3,7]] -= pad_size
            grasp_pose = self.generate_grasp_pose(projection_history=projection_history, bbox=bbox, numpy_x=numpy_x, numpy_y=numpy_y, numpy_z=numpy_z)
            base_candidates.poses.append(grasp_pose)
        base_candidates = transform_pose_array(base_candidates, self.planning_frame, self.tfBuffer)

        # Expand to 6DOF candidates and filter by IK reachability
        expanded = self._expand_6dof(base_candidates)
        valid_poses = []
        for pose in expanded.poses:
            ps = PoseStamped()
            ps.header.frame_id = self.planning_frame
            ps.pose = pose
            if self._check_ik(ps):
                valid_poses.append(pose)
            else:
                print("6DOF candidate rejected (IK/collision)")

        if not valid_poses:
            print("No valid 6DOF candidates found, returning unfiltered base candidates")
            return base_candidates

        expanded.poses = valid_poses
        print(f"6DOF expansion: {len(expanded.poses)} reachable candidates from {len(base_candidates.poses)} detections")
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
        """Return True only if the grasp pose AND the pre-grasp/retreat poses that
        plan_movement.py actually plans to (offset along the approach vector) all
        have a collision-free IK solution. A pose can be reachable on its own while
        its pre-grasp offset is inside a wall/crate or out of reach, which otherwise
        surfaces later as a MoveIt planning TIMED_OUT during the real grasp attempt."""
        approach_vec = self._get_approach_vec(pose_stamped.pose)
        pre_grasp = self._offset_pose(pose_stamped, approach_vec, PRE_GRASP_OFFSET_M)
        sink = self._offset_pose(pose_stamped, approach_vec, -SINK_OFFSET_M)
        retreat = self._offset_pose(pose_stamped, approach_vec, RETREAT_OFFSET_M)
        return (self._check_ik_single(pose_stamped)
                and self._check_ik_single(pre_grasp)
                and self._check_ik_single(sink)
                and self._check_ik_single(retreat))

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
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.1)
        try:
            res = self.ik_service(req)
            return res.error_code.val == 1
        except rospy.ServiceException as e:
            print(f"IK service call failed: {e}")
            return False

    def generate_grasp_pose(self, projection_history, bbox, numpy_x, numpy_y, numpy_z):
        grasp_center = (int(bbox[6]), int(bbox[7]))
        grasp_orientation = np.arctan2(bbox[8], bbox[9]) + np.pi/2

        grasp_pose = Pose()
        chosen_point = np.array(grasp_center)
        available_points = np.array(projection_history)
        index = np.sum((available_points-chosen_point)**2, axis=1, keepdims=True).argmin(axis=0)
        grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z = numpy_x[index], numpy_y[index], numpy_z[index]
        grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w = quaternion_from_euler(0, 0, grasp_orientation)
        return grasp_pose

    def publish_debug_image(self, debug_image, bboxes):
        self.grasp_candidates_raw_debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8"))
        decoded_debug_image = copy.deepcopy(debug_image)
        debug_image_raw = debug_image.copy()
        color_blue = (0,0,255)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2 = bbox[:4]
            debug_image = cv2.rectangle(debug_image, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0), 2)
            x_c,y_c = int(bbox[6]), int(bbox[7])
            angle = np.arctan2(bbox[8], bbox[9])
            p1, p2 = (int(x_c + 50*np.cos(angle)), int(y_c + 50*np.sin(angle))), (int(x_c - 50*np.cos(angle)), int(y_c - 50*np.sin(angle)))
            debug_image = cv2.circle(debug_image, (x_c,y_c), 5, color_blue, -1)
            debug_image = cv2.line(debug_image, p1, p2, color_blue, 2)
            decoded_debug_image = cv2.line(decoded_debug_image, p1, p2, color_blue, 2)
            decoded_debug_image = cv2.circle(decoded_debug_image, (x_c,y_c), 5, color_blue, -1)

        rospack = rospkg.RosPack()
        grasp_pckg_dir = rospack.get_path('grasp')
        catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
        pose_dir = os.path.join(catkin_ws_dir, "experiments/pose")
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)
        cv2.imwrite(os.path.join(pose_dir, "rgb.jpg"), debug_image_raw[:,:,::-1])
        cv2.imwrite(os.path.join(pose_dir, "rgb_debug.jpg"), debug_image[:,:,::-1])
        np.save(os.path.join(pose_dir, "bboxes.npy"),np.array(bboxes))

        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        ros_decoded_image_msg = self.bridge.cv2_to_imgmsg(decoded_debug_image, encoding="rgb8")
        self.grasp_candidates_debug_pub.publish(ros_image_msg)
        self.grasp_candidates_decoded_debug_pub.publish(ros_decoded_image_msg)
