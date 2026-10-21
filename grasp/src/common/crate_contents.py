"""Crate contents (the trusses in the crate) as a MoveIt collision object, so the planned
move to the pre-grasp pose goes around the tomatoes instead of through them.

MoveIt only knows the crate walls (the .scene file); the trusses come from the gripper
camera's point cloud. Taken once at the capture pose, right before grasp() plans, the
cloud is turned into a height map: one box per grid cell, from the crate floor up to the
highest point seen in that cell. A camera looking down can't see under the top layer
anyway, so filling the columns is the honest model of what is known.

The target truss has to stay reachable: every column within carve_radius_m of the
approach corridor (the line from the grasp point back out along the approach axis,
through the pre-grasp pose) is left out. The descent, grasp and lift happen inside that
corridor, so the obstacle is removed again once the arm is at the pre-grasp pose.

carve_mode 'gripper' also leaves out every column inside the gripper's own volume
(hand + wrist camera, a box around the TCP) swept along that corridor. A thin cylinder
is enough for a straight-down approach, where the hand stays above the trusses at the
pre-grasp pose; a tilted approach brings the hand down to the trusses beside the stem,
and with the cylinder alone its pre-grasp pose is in collision.

Why not MoveIt's octomap (sensors_kinect_pointcloud.yaml)? It keeps integrating the
wrist camera while the arm moves, and the target truss itself would block the final
descent; it would have to be cleared and refilled around every grasp, without a way to
leave the approach corridor free."""

import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from shape_msgs.msg import SolidPrimitive

CRATE_CONTENTS_ID = "crate_contents"

_FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32

# Gripper fingers in the finger link frames (franka_hand.xacro): the stock finger reaches
# z = 54.5 mm, the extension below it is finger_extension_m long; x is the finger's
# width, y its thickness (screw mount up to ~17 mm from the link origin).
FINGER_HALF_WIDTH_M = 0.0075
FINGER_HALF_THICKNESS_M = 0.017
STOCK_FINGER_LENGTH_M = 0.0545
STOCK_TCP_Z_M = 0.1034  # hand -> TCP without extension (franka_robot.xacro: tcp_z = 0.1034 + extension)


def cloud_to_xyz(cloud):
    """(N, 3) float64 xyz of a PointCloud2 (finite points only), read straight from the
    byte buffer -- point_cloud2.read_points takes seconds on a full 640x480 cloud."""
    offsets = {f.name: f.offset for f in cloud.fields if f.name in "xyz" and f.datatype == _FLOAT32}
    if len(offsets) != 3:
        raise ValueError(f"Point cloud has no float32 x/y/z fields: {[f.name for f in cloud.fields]}")
    n = cloud.width * cloud.height
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': [np.float32] * 3,
                      'offsets': [offsets['x'], offsets['y'], offsets['z']], 'itemsize': cloud.point_step})
    if cloud.row_step != cloud.width * cloud.point_step:
        raise ValueError("Padded point cloud rows are not supported")
    pts = np.frombuffer(cloud.data, dtype=dtype, count=n)
    xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1).astype(np.float64)
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def remove_finger_points(xyz, finger_transforms, finger_length_m, margin_m):
    """xyz (planning frame) without the points on or near the gripper fingers.
    finger_transforms: (Rotation, translation) of each finger link in the planning frame;
    a finger is the box |x| <= FINGER_HALF_WIDTH_M, |y| <= FINGER_HALF_THICKNESS_M,
    0 <= z <= finger_length_m in its link frame, grown by margin_m on every side."""
    keep = np.ones(len(xyz), dtype=bool)
    for rot, trans in finger_transforms:
        local = rot.inv().apply(xyz - trans)
        inside = ((np.abs(local[:, 0]) <= FINGER_HALF_WIDTH_M + margin_m)
                  & (np.abs(local[:, 1]) <= FINGER_HALF_THICKNESS_M + margin_m)
                  & (local[:, 2] >= -margin_m) & (local[:, 2] <= finger_length_m + margin_m))
        keep &= ~inside
    return xyz[keep]


def height_map_columns(xyz, interior, floor_z, cell_m, top_crop_z, min_points=10,
                       top_percentile=98.0, padding_m=0.01):
    """Columns (cx, cy, top_z) of a height map over the crate interior
    (x_lo, x_hi, y_lo, y_hi). Points above top_crop_z (a second guard against the gripper's
    own fingers in view, after remove_finger_points) or outside the interior (the walls, already in the scene) are dropped. A cell needs
    min_points points to count, so single depth-noise pixels don't become columns; its
    top is the top_percentile height (robust to flying pixels) plus padding_m."""
    x_lo, x_hi, y_lo, y_hi = interior
    keep = ((xyz[:, 0] > x_lo) & (xyz[:, 0] < x_hi) & (xyz[:, 1] > y_lo) & (xyz[:, 1] < y_hi)
            & (xyz[:, 2] > floor_z) & (xyz[:, 2] < top_crop_z))
    pts = xyz[keep]
    if len(pts) == 0:
        return []
    ix = np.floor((pts[:, 0] - x_lo) / cell_m).astype(int)
    iy = np.floor((pts[:, 1] - y_lo) / cell_m).astype(int)
    columns = []
    cells = {}
    for k, key in enumerate(zip(ix, iy)):
        cells.setdefault(key, []).append(k)
    for (i, j), idx in cells.items():
        if len(idx) < min_points:
            continue
        top = float(np.percentile(pts[idx, 2], top_percentile)) + padding_m
        # cell center, clipped so the column doesn't stick into the wall boxes
        cx = min(x_lo + (i + 0.5) * cell_m, x_hi - cell_m / 2)
        cy = min(y_lo + (j + 0.5) * cell_m, y_hi - cell_m / 2)
        columns.append((cx, cy, top))
    return columns


def _point_segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def carve_corridor(columns, floor_z, cell_m, grasp_xyz, approach_vec, carve_radius_m, corridor_length_m):
    """Drop every column that comes within carve_radius_m of the segment from the grasp
    point back out along the approach axis (approach_vec points from pre-grasp towards
    the grasp, as in plan_movement.py). Distances are measured from 8 points along each
    column's center line, less half the cell diagonal, so a column counts as soon as any
    part of it can reach the corridor."""
    a = np.asarray(grasp_xyz, dtype=float)
    b = a - np.asarray(approach_vec, dtype=float) * corridor_length_m
    half_diag = cell_m / np.sqrt(2)
    kept = []
    for cx, cy, top in columns:
        zs = np.linspace(floor_z, top, 8)
        samples = np.stack([np.full(8, cx), np.full(8, cy), zs], axis=1)
        if np.min(_point_segment_distance(samples, a, b)) - half_diag > carve_radius_m:
            kept.append((cx, cy, top))
    return kept


def carve_gripper_box(columns, floor_z, cell_m, grasp_xyz, grasp_rot, half_x_m, half_y_m, corridor_length_m):
    """Drop every column that reaches into the box |x| <= half_x_m, |y| <= half_y_m,
    -corridor_length_m <= z <= 0 in the grasp (TCP) frame: the gripper's volume swept from
    the grasp point back out along the approach axis (TCP z). grasp_rot: scipy Rotation of
    the TCP in the planning frame. Columns are tested at 8 points along their center
    line, with the box grown by half the cell diagonal."""
    a = np.asarray(grasp_xyz, dtype=float)
    grow = cell_m / np.sqrt(2)
    kept = []
    for cx, cy, top in columns:
        zs = np.linspace(floor_z, top, 8)
        local = grasp_rot.inv().apply(np.stack([np.full(8, cx), np.full(8, cy), zs], axis=1) - a)
        inside = ((np.abs(local[:, 0]) <= half_x_m + grow) & (np.abs(local[:, 1]) <= half_y_m + grow)
                  & (local[:, 2] >= -corridor_length_m - grow) & (local[:, 2] <= grow))
        if not inside.any():
            kept.append((cx, cy, top))
    return kept


def columns_to_collision_object(columns, floor_z, cell_m, frame_id):
    obj = CollisionObject()
    obj.id = CRATE_CONTENTS_ID
    obj.header.frame_id = frame_id
    obj.pose.orientation.w = 1.0
    obj.operation = CollisionObject.ADD
    for cx, cy, top in columns:
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[cell_m, cell_m, top - floor_z])
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = cx, cy, (floor_z + top) / 2
        pose.orientation.w = 1.0
        obj.primitives.append(box)
        obj.primitive_poses.append(pose)
    return obj


class CrateContentsObstacle(object):
    """Adds/removes the crate-contents height map in the MoveIt planning scene.
    scene: moveit_commander PlanningSceneInterface; crate_interior: (x_lo, x_hi, y_lo,
    y_hi, floor_z, rim_z) in planning_frame, see from_planning_scene_boxes()."""

    def __init__(self, scene, tf_buffer, planning_frame, crate_interior, cloud_topic='camera/depth/color/points',
                 cell_m=0.02, carve_radius_m=0.04, corridor_length_m=0.30, above_rim_m=0.08,
                 wall_margin_m=0.005, finger_margin_m=0.015, arm_id=None, carve_mode='cylinder',
                 gripper_half_x_m=0.06, gripper_half_y_m=0.11):
        self.scene = scene
        self.tf_buffer = tf_buffer
        self.planning_frame = planning_frame
        x_lo, x_hi, y_lo, y_hi, self.floor_z, rim_z = crate_interior
        # Shrink the interior a bit so points on the inside of the walls aren't counted
        self.interior = (x_lo + wall_margin_m, x_hi - wall_margin_m, y_lo + wall_margin_m, y_hi - wall_margin_m)
        self.top_crop_z = rim_z + above_rim_m
        self.cloud_topic = cloud_topic
        self.cell_m = cell_m
        self.carve_radius_m = carve_radius_m
        self.corridor_length_m = corridor_length_m
        self.finger_margin_m = finger_margin_m
        self.arm_id = arm_id or planning_frame.split('_link0')[0]
        if carve_mode not in ('cylinder', 'gripper'):
            raise ValueError(f"carve_mode must be 'cylinder' or 'gripper', not {carve_mode!r}")
        self.carve_mode = carve_mode
        # Hand (~20 x 6 cm, long side along the closing axis TCP y) plus the wrist camera
        # beside it (TCP x)
        self.gripper_half_x_m = gripper_half_x_m
        self.gripper_half_y_m = gripper_half_y_m
        self.columns = []

    def _lookup(self, target, source, stamp):
        tf = self.tf_buffer.lookup_transform(target, source, stamp, rospy.Duration(1.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return R.from_quat([q.x, q.y, q.z, q.w]), np.array([t.x, t.y, t.z])

    def capture_xyz(self, timeout_s=5.0):
        """Latest cloud from cloud_topic, in the planning frame, without the gripper's own
        fingers (located with TF at the cloud's time stamp, so the gripper opening and the
        finger extension are taken into account)."""
        cloud = rospy.wait_for_message(self.cloud_topic, PointCloud2, timeout=timeout_s)
        rot, trans = self._lookup(self.planning_frame, cloud.header.frame_id, cloud.header.stamp)
        xyz = rot.apply(cloud_to_xyz(cloud)) + trans
        try:
            fingers = [self._lookup(self.planning_frame, f"{self.arm_id}_{side}finger", cloud.header.stamp)
                       for side in ("left", "right")]
            _, tcp = self._lookup(f"{self.arm_id}_hand", f"{self.arm_id}_hand_tcp", cloud.header.stamp)
            finger_length = STOCK_FINGER_LENGTH_M + max(tcp[2] - STOCK_TCP_Z_M, 0.0)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Crate contents: no finger transforms ({e}); only the height crop removes the fingers")
            return xyz
        n = len(xyz)
        xyz = remove_finger_points(xyz, fingers, finger_length, self.finger_margin_m)
        rospy.loginfo(f"Crate contents: removed {n - len(xyz)} finger points "
                      f"(fingers {finger_length * 100:.1f} cm long, margin {self.finger_margin_m * 100:.1f} cm)")
        return xyz

    def add(self, grasp_xyz, approach_vec, xyz=None, grasp_quat=None):
        """Capture (unless xyz is given), build, carve and add the obstacle; returns the
        number of columns added. Blocks until the planning scene contains it. grasp_quat
        (x, y, z, w of the grasp pose) is needed for carve_mode 'gripper'."""
        start = time.time()
        if xyz is None:
            xyz = self.capture_xyz()
        columns = height_map_columns(xyz, self.interior, self.floor_z, self.cell_m, self.top_crop_z)
        n_all = len(columns)
        columns = carve_corridor(columns, self.floor_z, self.cell_m, grasp_xyz, approach_vec,
                                 self.carve_radius_m, self.corridor_length_m)
        if self.carve_mode == 'gripper':
            if grasp_quat is None:
                rospy.logwarn("Crate contents: carve_mode 'gripper' needs the grasp orientation; "
                              "only the corridor cylinder is carved")
            else:
                columns = carve_gripper_box(columns, self.floor_z, self.cell_m, grasp_xyz, R.from_quat(grasp_quat),
                                            self.gripper_half_x_m, self.gripper_half_y_m, self.corridor_length_m)
        self.columns = columns  # (cx, cy, top) of the added boxes, e.g. for clearance metrics
        if not columns:
            rospy.logwarn("Crate contents: no columns left to add (empty cloud/crate?)")
            return 0
        self.scene.add_object(columns_to_collision_object(columns, self.floor_z, self.cell_m, self.planning_frame))
        self._wait_for(present=True)
        rospy.loginfo(f"Crate contents: added {len(columns)} columns ({n_all - len(columns)} carved out around "
                      f"the approach) in {time.time() - start:.2f}s")
        return len(columns)

    def remove(self):
        if CRATE_CONTENTS_ID in self.scene.get_known_object_names():
            self.scene.remove_world_object(CRATE_CONTENTS_ID)
            self._wait_for(present=False)
        self.columns = []

    def _wait_for(self, present, timeout_s=3.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if (CRATE_CONTENTS_ID in self.scene.get_known_object_names()) == present:
                return
            rospy.sleep(0.05)
        rospy.logwarn(f"Crate contents: planning scene did not {'add' if present else 'remove'} "
                      f"'{CRATE_CONTENTS_ID}' within {timeout_s}s")


def crate_interior_from_boxes(boxes, floor_names=("floor",)):
    """(x_lo, x_hi, y_lo, y_hi, floor_z, rim_z) of the crate from the scene's wall boxes
    ((name, center, Rotation, dims), axis-aligned walls as in the .scene files): the
    interior is bounded by the inner faces of the walls around the walls' centroid."""
    walls = [(c, d) for name, c, _, d in boxes
             if not any(f in name.lower() for f in floor_names) and name != CRATE_CONTENTS_ID]
    if not walls:
        raise RuntimeError("No crate wall boxes in the planning scene")
    centroid = np.mean([c for c, _ in walls], axis=0)
    x_lo, x_hi, y_lo, y_hi = -np.inf, np.inf, -np.inf, np.inf
    for c, d in walls:
        lo, hi = c - d / 2, c + d / 2
        # A wall bounds the interior on the side of the centroid it doesn't cover
        if hi[0] < centroid[0]:
            x_lo = max(x_lo, hi[0])
        elif lo[0] > centroid[0]:
            x_hi = min(x_hi, lo[0])
        if hi[1] < centroid[1]:
            y_lo = max(y_lo, hi[1])
        elif lo[1] > centroid[1]:
            y_hi = min(y_hi, lo[1])
    floor_z = min(c[2] - d[2] / 2 for c, d in walls)
    # A crate floor box (e.g. samla_crate.scene's Crate_floor) raises the floor
    for name, c, _, d in boxes:
        if 'floor' in name.lower() and c[0] - d[0] / 2 > x_lo - 0.05 and c[0] + d[0] / 2 < x_hi + 0.05:
            floor_z = max(floor_z, c[2] + d[2] / 2)
    rim_z = max(c[2] + d[2] / 2 for c, d in walls)
    if not np.all(np.isfinite([x_lo, x_hi, y_lo, y_hi])):
        raise RuntimeError(f"Could not find walls on all four sides of the crate: {x_lo, x_hi, y_lo, y_hi}")
    return x_lo, x_hi, y_lo, y_hi, floor_z, rim_z
