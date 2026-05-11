#!/usr/bin/env python3
"""
Project 3 deterministic mapping + delivery workflow.

Coordinate / yaw convention (all code in this file uses this consistently):
  - World origin: top-left corner of the 10 ft x 10 ft workspace.
  - World +x: rightward along the top edge.
  - World +y: downward along the left edge.
  - Yaw = 0  : robot faces +x (right).
  - Yaw = π/2: robot faces +y (down).
  - Yaw increases for a clockwise (CW) turn when viewed from above
    (turning right  → yaw increases).
  - Yaw decreases for a counter-clockwise (CCW) turn when viewed from above
    (turning left   → yaw decreases).

RoboMaster chassis.move / drive_speed frame:
  - chassis.move(x=+d)  : robot moves forward  d metres.
  - chassis.move(y=+d)  : robot moves rightward d metres (body +y = robot right).
  - chassis.move(z=+θ)  : robot rotates CCW by θ degrees → yaw DECREASES by θ.
  - drive_speed(z=+ω)   : CCW angular rate     → yaw DECREASES at rate ω deg/s.

Camera geometry:
  - Pinhole model; intrinsics K_CAM from config.py.
  - Camera is mounted at CAM_OFFSET_Z_M = 0.32 m above the ground plane,
    tilted CAM_PITCH_DEG = -19.6° (downward).
  - T_ROBOT_FROM_CAMERA (config.py) transforms a point from the OpenCV camera
    frame (z-forward, x-right, y-down) into the robot body frame
    (x-forward, y-left, z-up), including the pitch tilt and mount offsets.
  - YOLO bounding-box distance uses the bottom edge of the bbox together with
    the known camera height and pitch to compute the ground-plane distance to
    the base of the object (which sits on the ground). No empirical scale
    factor is required; the calibrated linear correction (slope + offset) is
    retained to absorb any remaining systematic error.
  - AprilTag landmarks use the full 3-D pose returned by pupil_apriltags
    (pose_R, pose_t) transformed through T_ROBOT_FROM_CAMERA and then into the
    world frame.  Because the tag is mounted on the side of a box (elevated),
    the code projects the tag position onto the ground plane (z_robot = 0,
    i.e. sets the robot-frame z contribution to zero) to recover the ground XY
    position of the object rather than the tag face center.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pupil_apriltags
from ultralytics import YOLO

import robomaster
from config import (
    BATTERY_LARGE_BRICK_COST,
    BATTERY_RECHARGE_LEVEL,
    BATTERY_SMALL_BRICK_COST,
    BATTERY_START_PCT,
    CAM_OFFSET_Z_M,
    CAM_PITCH_DEG,
    CLASS_BOX,
    CLASS_LARGE_BRICK,
    CLASS_SMALL_BRICK,
    DEFAULT_ARM_X,
    DEFAULT_ARM_Y,
    DEFAULT_MODEL_PATH,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
    K_CAM,
    LARGE_GOAL_TAG_IDS,
    MODEL_PATH,
    MOVE_SPEED_MPS,
    RECHARGE_TAG_IDS,
    ROBOT_IP,
    ROBOT_SN,
    SMALL_GOAL_TAG_IDS,
    T_ROBOT_FROM_CAMERA,
    TAG_FAMILY,
    TAG_SIZE_M,
    TURN_SPEED_DPS,
)
from robomaster import camera as rm_camera
from robomaster import robot

from tower_utils import Detection, get_detections, pick_up_tower, place_down_tower


FT_TO_M = 0.3048
WORKSPACE_W_M = 10.0 * FT_TO_M   # metres, x-axis
WORKSPACE_H_M = 10.0 * FT_TO_M   # metres, y-axis

# Robot starts near the top-left region, facing +x (rightward).
# START_YAW_RAD = 0 means the robot points toward the recharge block that is
# directly in front of it at startup.
START_X_M = 0.20
START_Y_M = 0.20
START_YAW_RAD = 0.0              # facing +x (rightward)

INITIAL_FORWARD_STEP_M = 2.0 * FT_TO_M

# Pixel tolerances.
CENTER_TOL_PX = 35.0
LEFT_SCAN_STEP_M = 0.18

# Obstacle geometry.
OBSTACLE_MERGE_RADIUS_M = 0.30
OBSTACLE_PATH_CLEARANCE_M = 0.38
OBSTACLE_DETOUR_MARGIN_M = 0.28

# Navigation stop distances.
LANDMARK_STOP_DIST_M = 0.55
GOAL_SERVO_DIST_M = 0.30
RECHARGE_SERVO_DIST_M = 0.22
DOCK_SEARCH_STEP_DEG = 15.0
MAX_TAG_SEARCH_STEPS = 6

# Re-localisation tolerances.
REFERENCE_CENTER_TOL_PX = 25.0
REFERENCE_DIST_TOL_M = 0.18
INTERMEDIATE_SNAP_DIST_M = 0.12

DEBUG_WINDOW_NAME = "Project 3 Feed"

# ---------------------------------------------------------------------------
# Camera-geometry constants derived from config.
# ---------------------------------------------------------------------------
# Camera mount pitch in radians (negative = pitched downward).
_CAM_PITCH_RAD: float = math.radians(CAM_PITCH_DEG)   # ≈ -0.342 rad

# Camera height above ground (robot body frame z of the camera origin).
_CAM_HEIGHT_M: float = float(CAM_OFFSET_Z_M)           # 0.32 m

# Focal lengths and principal point from K_CAM.
_FX: float = float(K_CAM[0, 0])   # 314 px
_FY: float = float(K_CAM[1, 1])   # 314 px
_CX: float = float(K_CAM[0, 2])   # 320 px  (horizontal principal point)
_CY: float = float(K_CAM[1, 2])   # 180 px  (vertical   principal point)

# Calibrated linear correction fitted to real measurements.
# Applied as: corrected_dist = SLOPE * raw_dist + OFFSET
# The raw_dist is now the geometrically correct ground-plane distance, so
# the slope should be close to 1.0 and the offset close to 0.0 if the
# camera parameters are well-calibrated.  Keep the empirical values until
# new calibration data are collected.
DISTANCE_CORRECTION_SLOPE: float = 1
DISTANCE_CORRECTION_OFFSET: float = 0

# Known real-world heights of YOLO-detected object classes (metres).
# Used only to estimate depth from bounding-box apparent height when the
# ground-plane geometry alone is insufficient (e.g. object partially cut off).
OBJECT_HEIGHTS_M = {
    CLASS_BOX:         0.28,
    CLASS_SMALL_BRICK: 0.10,
    CLASS_LARGE_BRICK: 0.19,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Pose2D:
    """Robot pose in the global workspace frame.

    x, y : position in metres.
    yaw  : heading in radians, CW-positive, 0 = facing world +x (right).
    """
    x: float
    y: float
    yaw: float


@dataclass
class Landmark:
    """A mapped world object represented by a single 2-D ground-plane point."""
    kind: str
    x: float
    y: float
    tag_id: Optional[int] = None


@dataclass
class TagReference:
    """Reference drop-off tag view used to re-localise from the relay point."""
    tag_id: int
    goal_kind: str
    world_x: float
    world_y: float
    world_yaw: float
    reference_pose: Pose2D
    reference_distance_m: float
    reference_center_x_px: float


@dataclass
class WorldMap:
    """Persistent map state for the deterministic mission."""
    recharge: Optional[Landmark] = None
    small_goal: Optional[Landmark] = None
    large_goal: Optional[Landmark] = None
    dock: Optional[Landmark] = None
    intermediate: Optional[Landmark] = None
    dropoff_tag_ref: Optional[TagReference] = None
    obstacles: List[Landmark] = field(default_factory=list)

    def goal_for_kind(self, kind: str) -> Optional[Landmark]:
        if kind == "small_goal":
            return self.small_goal
        if kind == "large_goal":
            return self.large_goal
        return None

    def set_goal(self, kind: str, x: float, y: float, tag_id: int) -> Landmark:
        existing = self.goal_for_kind(kind)
        if existing is not None:
            return existing
        landmark = Landmark(kind=kind, x=x, y=y, tag_id=tag_id)
        if kind == "small_goal":
            self.small_goal = landmark
        else:
            self.large_goal = landmark
        return landmark

    def add_or_update_obstacle(self, x: float, y: float, tag_id: Optional[int] = None) -> Landmark:
        if tag_id is not None:
            for obstacle in self.obstacles:
                if obstacle.tag_id == tag_id:
                    return obstacle
        for obstacle in self.obstacles:
            if math.hypot(obstacle.x - x, obstacle.y - y) <= OBSTACLE_MERGE_RADIUS_M:
                if tag_id is not None:
                    obstacle.tag_id = tag_id
                return obstacle
        obstacle = Landmark(kind="obstacle", x=x, y=y, tag_id=tag_id)
        self.obstacles.append(obstacle)
        return obstacle

    def right_side_goal(self) -> Optional[Landmark]:
        # "Right side" = largest world-x (rightward in the workspace).
        goals = [g for g in [self.small_goal, self.large_goal] if g is not None]
        if not goals:
            return None
        return max(goals, key=lambda g: g.x)

    def mapped_block_count(self) -> int:
        count = len(self.obstacles)
        count += 1 if self.recharge is not None else 0
        count += 1 if self.small_goal is not None else 0
        count += 1 if self.large_goal is not None else 0
        return count

    def summary(self) -> str:
        def fmt(lm: Optional[Landmark]) -> str:
            return "NOT FOUND" if lm is None else f"({lm.x:.2f}, {lm.y:.2f})"
        return "\n".join([
            "=== WorldMap ===",
            f"  recharge:   {fmt(self.recharge)}",
            f"  small_goal: {fmt(self.small_goal)}",
            f"  large_goal: {fmt(self.large_goal)}",
            f"  dock:       {fmt(self.dock)}",
            f"  intermediate: {fmt(self.intermediate)}",
            f"  obstacles:  {len(self.obstacles)}",
        ])


class BatteryManager:
    """Minimal simulated battery model driven by the brick class carried."""
    def __init__(self, start_pct: float = BATTERY_START_PCT):
        self.level = float(start_pct)

    def cost_for_class(self, brick_class: int) -> float:
        return BATTERY_LARGE_BRICK_COST if brick_class == CLASS_LARGE_BRICK else BATTERY_SMALL_BRICK_COST

    def can_pick(self, brick_class: int) -> bool:
        return self.level - self.cost_for_class(brick_class) > 0.0

    def consume(self, brick_class: int) -> None:
        self.level = max(0.0, self.level - self.cost_for_class(brick_class))

    def recharge(self) -> None:
        self.level = float(BATTERY_RECHARGE_LEVEL)


class AprilTagDetector:
    """Thin wrapper so the rest of the file stays agnostic to detector details."""
    def __init__(
        self,
        K: np.ndarray = K_CAM,
        family: str = TAG_FAMILY,
        marker_size_m: float = TAG_SIZE_M,
    ):
        self.camera_params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        self.detector = pupil_apriltags.Detector(
            families=family,
            nthreads=2,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.marker_size_m = marker_size_m

    def find_tags(self, gray: np.ndarray) -> list:
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )

    @staticmethod
    def tag_distance_m(detection) -> float:
        """3-D Euclidean distance from camera to the tag face centre (metres)."""
        return float(np.linalg.norm(np.array(detection.pose_t, dtype=float).reshape(3)))


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def wrap_to_pi(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def goal_kind_from_tag(tag_id: int) -> Optional[str]:
    if tag_id in SMALL_GOAL_TAG_IDS:
        return "small_goal"
    if tag_id in LARGE_GOAL_TAG_IDS:
        return "large_goal"
    return None


def is_obstacle_tag(tag_id: int) -> bool:
    return (
        tag_id not in SMALL_GOAL_TAG_IDS
        and tag_id not in LARGE_GOAL_TAG_IDS
        and tag_id not in RECHARGE_TAG_IDS
    )


def brick_class_for_goal(goal: Landmark) -> int:
    return CLASS_SMALL_BRICK if goal.kind == "small_goal" else CLASS_LARGE_BRICK


def rotz_cw(yaw_rad: float) -> np.ndarray:
    """3×3 rotation matrix for a clockwise-positive yaw in the world XY plane.

    Convention: yaw increases CW when viewed from above (i.e. turning right).
    The matrix maps a robot-forward unit vector at the given yaw to world XY:
        world_x = cos(yaw) * robot_fwd_x  +  sin(yaw) * robot_right_x
        world_y = sin(yaw) * robot_fwd_x  -  cos(yaw) * robot_right_x  ... etc.

    Written out as a 3×3 (Z unused at world level, kept for homogeneous math):
        R = [[ c,  s,  0],
             [-s,  c,  0],   ← standard CW rotation in 2-D
             [ 0,  0,  1]]
    """
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [[c,  s, 0.0],
         [-s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def transform_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.array(t, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def yaw_from_rotation_cw(R: np.ndarray) -> float:
    """Extract the CW-positive yaw angle from a rotation matrix built with
    rotz_cw.  R[0,0]=cos(yaw), R[0,1]=sin(yaw)  ⟹  yaw = atan2(R[0,1], R[0,0]).
    """
    return math.atan2(float(R[0, 1]), float(R[0, 0]))


def read_frame(ep_camera, timeout: float = 1.0) -> Optional[np.ndarray]:
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=timeout)
    except Empty:
        return None
    return frame


def copy_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(x=pose.x, y=pose.y, yaw=pose.yaw)


def clamp_to_workspace(x: float, y: float) -> Tuple[float, float]:
    return (
        min(max(0.0, x), WORKSPACE_W_M),
        min(max(0.0, y), WORKSPACE_H_M),
    )


# ---------------------------------------------------------------------------
# Camera-geometry helpers
# ---------------------------------------------------------------------------

def pixel_bearing_rad(cx_px: float) -> float:
    """Horizontal bearing angle to a pixel column, positive = right of centre.

    In the CW-positive yaw convention, an object to the right of the image
    centre has a positive bearing, meaning the robot would need to rotate CW
    (increase yaw) to face it directly.  atan2(Δx_px, fx) gives exactly that.
    """
    return math.atan2(float(cx_px) - _CX, _FX)


def ground_distance_from_bottom_pixel(v_bottom_px: float) -> Optional[float]:
    """Compute the ground-plane distance to the base of an object detected by
    YOLO, using the bottom edge of its bounding box and camera geometry.

    Derivation
    ----------
    The camera is at height h_c above the ground, pitched downward by α
    (α > 0 means optical axis tilts toward the ground, so CAM_PITCH_DEG is
    negative and _CAM_PITCH_RAD is negative; we use α = -_CAM_PITCH_RAD).

    The optical axis points in the direction (0, sin(-α), cos(-α)) in robot
    body frame (x-forward, y-left, z-up).  For a pixel at row v in the image,
    the ray through that pixel in the camera frame is:

        ray_cam = (0,  (v - cy) / fy,  1)   [normalised so z=1]

    Transforming to robot body frame via T_ROBOT_FROM_CAMERA (which already
    encodes the pitch rotation):

        ray_robot = R_robot_from_camera @ ray_cam

    The ray hits the ground (z_robot = 0) at a scale factor t such that:

        (camera_origin_in_robot)_z  +  t * ray_robot_z  =  0
        ⟹  t = h_c / (-ray_robot_z)     [h_c > 0, ray_robot_z < 0 for down-
                                           looking pixels]

    The forward (x) distance along the ground is then:

        d_forward = (camera_origin_in_robot)_x  +  t * ray_robot_x

    For a zero camera x-offset (CAM_OFFSET_X_M = 0):

        d_forward = t * ray_robot_x

    This is the distance in the robot's forward direction, which equals the
    ground-plane range for flat terrain.

    Returns None if the pixel ray is parallel to or above the ground plane.
    """
    # Normalised image-plane ray in camera frame (z=1 convention).
    ray_cam = np.array([0.0, (v_bottom_px - _CY) / _FY, 1.0], dtype=float)

    # Transform ray direction to robot body frame.
    R_rc = T_ROBOT_FROM_CAMERA[:3, :3]
    t_rc = T_ROBOT_FROM_CAMERA[:3, 3]   # camera origin in robot frame
    ray_robot = R_rc @ ray_cam           # direction (not a point)

    # ray_robot[2] is the z-component in robot body frame (z-up).
    # For the ray to intersect the ground (z=0) the z-component of the ray
    # must be negative (pointing downward).
    if ray_robot[2] >= 0.0:
        return None

    # Scale factor t so that camera_origin_z + t * ray_robot_z = 0.
    t_scale = -t_rc[2] / ray_robot[2]   # t_rc[2] = h_c > 0, ray_robot[2] < 0

    # Ground-plane x coordinate in robot body frame (forward direction).
    d_forward = t_rc[0] + t_scale * ray_robot[0]

    if d_forward <= 0.0:
        return None

    return float(d_forward)


def estimate_bbox_distance_m(detection: Detection, cls: Optional[int] = None) -> Optional[float]:
    """Ground-plane distance to the base of a YOLO-detected object.

    Uses the bottom edge of the bounding box and the camera geometry (height +
    pitch) to compute a geometrically correct range.  A calibrated linear
    correction is applied to absorb any remaining systematic error.

    The object is assumed to sit on the ground plane (e.g. bricks, boxes).
    The bottom row of the bounding box corresponds to the point where the
    object meets the floor, which is the ground-plane target point.
    """
    v_bottom = float(detection.cy) + 0.5 * float(detection.h)
    raw_dist = ground_distance_from_bottom_pixel(v_bottom)

    if raw_dist is None:
        # Fallback: apparent-height model (less accurate).
        object_class = detection.cls if cls is None else cls
        H_real = OBJECT_HEIGHTS_M.get(object_class)
        if H_real is None:
            return None
        h_px = max(1.0, float(detection.h))
        raw_dist = _FY * H_real / h_px

    corrected = DISTANCE_CORRECTION_SLOPE * raw_dist + DISTANCE_CORRECTION_OFFSET
    return max(0.0, corrected)


def world_from_range_and_bearing(
    pose: Pose2D,
    range_m: float,
    bearing_rad: float,
) -> Tuple[float, float]:
    """Convert a (range, bearing) observation to world (x, y).

    bearing_rad is the horizontal angle from the robot's forward direction to
    the object, CW-positive (matching the yaw convention).

    heading = pose.yaw + bearing_rad  gives the absolute world angle to the
    object, where yaw=0 → facing +x, yaw=π/2 → facing +y (downward).

    With CW-positive angles and world +y downward:
        world_x += range * cos(heading)   [rightward component]
        world_y += range * sin(heading)   [downward  component]

    Verification:
      yaw=π/2, bearing=0 → heading=π/2 → cos=0, sin=1 → (Δx=0, Δy=+range) ✓
      yaw=0,   bearing=0 → heading=0   → cos=1, sin=0 → (Δx=+range, Δy=0) ✓
      yaw=π/2, bearing=+π/4 (object 45° to robot's right when facing down)
                           → heading=3π/4 → cos<0, sin>0 → object up-right ✓
    """
    heading = wrap_to_pi(pose.yaw + bearing_rad)
    world_x = pose.x + range_m * math.cos(heading)
    world_y = pose.y + range_m * math.sin(heading)
    return world_x, world_y


def detection_world_position(detection: Detection, pose: Pose2D) -> Optional[Tuple[float, float]]:
    """Convert a YOLO detection (ground-plane object) to world (x, y)."""
    distance_m = estimate_bbox_distance_m(detection)
    if distance_m is None:
        return None
    bearing_rad = pixel_bearing_rad(detection.cx)
    return world_from_range_and_bearing(pose, distance_m, bearing_rad)


def tag_world_position_from_pose(tag, pose: Pose2D) -> Tuple[float, float]:
    """Compute the world (x, y) of an AprilTag's ground-plane footprint.

    Unlike YOLO objects, AprilTags are mounted on the side of a box (elevated
    above the ground).  We therefore use the full 3-D pose_t returned by
    pupil_apriltags, transform it to robot body frame via T_ROBOT_FROM_CAMERA,
    then project the result onto the ground plane (zeroing the robot-frame Z)
    before rotating into the world frame.

    Steps:
      1. p_cam = pose_t  (3-D position of the tag centre in camera frame, m).
      2. p_robot = R_robot_from_camera @ p_cam + t_robot_from_camera
                 (position in robot body frame; x-forward, y-left, z-up).
      3. Ground-plane projection: keep only robot-frame x and y
         (i.e. pretend the tag is at z_robot = 0).  This gives the horizontal
         offset from the robot to directly below the tag.
      4. Rotate the horizontal offset by the robot's world yaw and add the
         robot's world position.

    Robot body frame: +x forward, +y left, +z up.
    World frame: +x right, +y down, yaw CW-positive.
    The rotation from robot body to world (for the horizontal plane) is:
        world_dx = p_robot_x * cos(yaw) + p_robot_y * (-sin(yaw))
                 = p_robot_x * cos(yaw) - p_robot_y * sin(yaw)
        world_dy = p_robot_x * sin(yaw) + p_robot_y * cos(yaw)
                 [NB: robot +y is LEFT, world +y is DOWN, CW yaw ⟹ see below]

    Verification (yaw=π/2, robot facing +y/down):
      Object directly ahead (p_robot_x=d, p_robot_y=0):
        world_dx = d*cos(π/2) - 0 = 0        world_dy = d*sin(π/2) + 0 = d  ✓
      Object to robot's left (p_robot_y=+L, p_robot_x=0, robot facing down):
        robot left when facing down = world +x direction.
        world_dx = 0 - L*sin(π/2) = -L  ✗ … wait, robot +y is LEFT.
        Robot facing down (+y world): robot's left is world +x.  p_robot_y=+L.
        world_dx = 0*cos(π/2) - L*sin(π/2) = -L  ← this is wrong sign.
        → Because robot +y is LEFT but world +x is RIGHT, a left-offset in
          robot frame is a NEGATIVE x in world frame only if robot faces down.
          Let's re-examine: robot facing down (yaw=π/2):
            robot_forward (x̂_r) → world +y  ✓ (east/south)
            robot_left    (ŷ_r) → world -x  ✓ (northward in screen coords)
          So robot +y_left = world -x.  An object at p_robot_y=+L (to robot's
          left when facing down) is at world_x = pose.x - L.
        world_dx = p_robot_x*cos(yaw) - p_robot_y*sin(yaw)
                 = 0*0             - L*1              = -L   ✓
        world_dy = p_robot_x*sin(yaw) + p_robot_y*cos(yaw)   — wait, sign?

    Full derivation:
      The world-frame basis vectors in terms of robot body frame, for CW yaw:
          x̂_world = cos(yaw)*x̂_robot - sin(yaw)*ŷ_robot
          ŷ_world = sin(yaw)*x̂_robot + cos(yaw)*ŷ_robot  ← but ŷ_robot=LEFT
      World vector components of a robot-frame offset (px, py):
          Δworld_x = px*cos(yaw) - py*sin(yaw)
          Δworld_y = px*sin(yaw) + py*cos(yaw)
      Double-check yaw=0 (robot facing right):
          (px,py) = (d,0) → Δx=d, Δy=0  ✓ (forward = right)
          (px,py) = (0,L) → Δx=-L·0=0... wait sin(0)=0, so Δx=0, Δy=L.
          But at yaw=0 (facing right), robot LEFT is world +y (downward).
          So an object to robot's left should be at world_y += L.  Δy = L ✓.
      Double-check yaw=π/2 (facing down):
          (px,py)=(d,0)→ Δx=d*0=0, Δy=d*1=d ✓ (ahead = downward)
          (px,py)=(0,L)→ Δx=-L*1=-L, Δy=L*0=0 ✓ (robot left=world leftward=-x)
    """
    pose_t = np.array(tag.pose_t, dtype=float).reshape(3)

    # Step 1–2: camera frame → robot body frame.
    R_rc = T_ROBOT_FROM_CAMERA[:3, :3]
    t_rc = T_ROBOT_FROM_CAMERA[:3, 3]
    p_robot = R_rc @ pose_t + t_rc    # (x_fwd, y_left, z_up) in robot frame

    # Step 3: project onto ground plane — keep only horizontal components.
    px = float(p_robot[0])   # forward
    py = float(p_robot[1])   # left

    # Step 4: rotate into world frame and add robot world position.
    yaw = pose.yaw
    world_x = pose.x + px * math.cos(yaw) - py * math.sin(yaw)
    world_y = pose.y + px * math.sin(yaw) + py * math.cos(yaw)
    return world_x, world_y


# ---------------------------------------------------------------------------
# Pose-tracking motion primitives
# ---------------------------------------------------------------------------

def move_robot(
    ep_chassis,
    pose: Pose2D,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    z_deg: float = 0.0,
    xy_speed: float = MOVE_SPEED_MPS,
    z_speed: float = TURN_SPEED_DPS,
) -> None:
    """Execute a chassis.move(...) command and update the dead-reckoned pose.

    RoboMaster chassis.move body-frame convention:
      x_m  > 0 : move forward.
      y_m  > 0 : move rightward (body +y = robot right).
      z_deg> 0 : rotate CCW (counter-clockwise) → world yaw DECREASES.

    World-frame update for translation (x_m, y_m in robot body frame):
      Robot body frame: +x_b forward, +y_b right.
      At world yaw θ (CW, 0=right, π/2=down):
          x̂_b → (cos θ, sin θ) in world
          ŷ_b → (-sin θ, cos θ) in world   [+y_b=right; right at θ=0 is
                                               world +y, i.e. (0,+1), so
                                               -sin(0)=0, cos(0)=1 ✓;
                                               at θ=π/2 right is world −x:
                                               -sin(π/2)=-1, cos(π/2)=0 ✓]
      world_dx = x_m * cos(yaw) - y_m * sin(yaw)
      world_dy = x_m * sin(yaw) + y_m * cos(yaw)

    Verification:
      yaw=π/2, x_m=d (forward, i.e. downward in world):
          world_dx = d*0 + 0 = 0  ✓
          world_dy = d*1 + 0 = d  ✓
      yaw=π/2, y_m=r (move right in body, i.e. world −x when facing down):
          world_dx = 0 - r*1 = -r ✓
          world_dy = 0 + r*0 = 0  ✓
      yaw=0, x_m=d (forward = right in world):
          world_dx = d*1 + 0 = d  ✓, world_dy = 0  ✓
      yaw=0, y_m=r (move right = downward in world at yaw=0):
          world_dx = 0 - r*0 = 0  ✓
          world_dy = 0 + r*1 = r  ✓

    Yaw update:
      chassis.move(z=+θ) rotates CCW → yaw decreases:
          pose.yaw -= radians(z_deg)
    """
    ep_chassis.move(
        x=x_m, y=y_m, z=z_deg,
        xy_speed=xy_speed, z_speed=z_speed,
    ).wait_for_completed()

    world_dx = x_m * math.cos(pose.yaw) - y_m * math.sin(pose.yaw)
    world_dy = x_m * math.sin(pose.yaw) + y_m * math.cos(pose.yaw)
    pose.x += world_dx
    pose.y += world_dy
    pose.yaw = wrap_to_pi(pose.yaw - math.radians(z_deg))
    print(
        f"[Pose] x={pose.x:.3f} y={pose.y:.3f} "
        f"yaw={math.degrees(pose.yaw):.1f}° "
        f"(move x={x_m:.3f} y={y_m:.3f} z={z_deg:.1f}°)"
    )


def integrate_drive_speed(
    pose: Pose2D,
    vx: float,
    vy: float,
    wz_deg_s: float,
    dt_s: float,
) -> None:
    """Update pose after a short drive_speed burst.

    Same body-frame / yaw convention as move_robot.
    drive_speed(z=+ω) is CCW → yaw decreases at rate ω deg/s.
    """
    world_dx = (vx * math.cos(pose.yaw) - vy * math.sin(pose.yaw)) * dt_s
    world_dy = (vx * math.sin(pose.yaw) + vy * math.cos(pose.yaw)) * dt_s
    pose.x += world_dx
    pose.y += world_dy
    pose.yaw = wrap_to_pi(pose.yaw - math.radians(wz_deg_s * dt_s))


def turn_to_yaw(ep_chassis, pose: Pose2D, target_yaw_rad: float) -> None:
    """Rotate to an absolute world yaw using chassis.move(z=...).

    We want: pose.yaw - radians(z_deg) = target_yaw_rad
    ⟹ z_deg = degrees(pose.yaw - target_yaw_rad)

    Positive z_deg → CCW turn → yaw decreases (correct for CW convention).
    Negative z_deg → CW  turn → yaw increases.

    Example: current yaw=π/2 (down), target yaw=0 (right).
      delta = π/2 → z_deg = +90° → CCW → turns left (yaw decreases to 0) ✓

    Example: current yaw=0 (right), target yaw=π/2 (down).
      delta = -π/2 → z_deg = -90° → CW → turns right (yaw increases to π/2) ✓
    """
    delta_deg = math.degrees(wrap_to_pi(pose.yaw - target_yaw_rad))
    if abs(delta_deg) < 1.0:
        return
    move_robot(ep_chassis, pose, z_deg=delta_deg)


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def path_blocked_by_known_obstacle(
    start: Pose2D,
    target_x: float,
    target_y: float,
    world_map: Optional[WorldMap],
) -> bool:
    if world_map is None:
        return False
    for obs in world_map.obstacles:
        if point_to_segment_distance(obs.x, obs.y, start.x, start.y, target_x, target_y) < OBSTACLE_PATH_CLEARANCE_M:
            return True
    return False


def plan_navigation_points(
    start: Pose2D,
    target_x: float,
    target_y: float,
    world_map: Optional[WorldMap],
) -> List[Tuple[float, float]]:
    if world_map is None or not world_map.obstacles:
        return [(target_x, target_y)]
    if not path_blocked_by_known_obstacle(start, target_x, target_y, world_map):
        return [(target_x, target_y)]

    sx, sy = start.x, start.y
    dx, dy = target_x - sx, target_y - sy
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return [(target_x, target_y)]

    perp_x = -dy / distance
    perp_y = dx / distance
    blocking = sorted(
        world_map.obstacles,
        key=lambda obs: point_to_segment_distance(obs.x, obs.y, sx, sy, target_x, target_y),
    )
    obstacle = blocking[0]
    detour_radius = OBSTACLE_PATH_CLEARANCE_M + OBSTACLE_DETOUR_MARGIN_M

    best_path: Optional[List[Tuple[float, float]]] = None
    best_cost = float("inf")
    for sign in (1.0, -1.0):
        detour_x, detour_y = clamp_to_workspace(
            obstacle.x + sign * perp_x * detour_radius,
            obstacle.y + sign * perp_y * detour_radius,
        )
        detour_pose = Pose2D(x=detour_x, y=detour_y, yaw=start.yaw)
        if path_blocked_by_known_obstacle(start, detour_x, detour_y, world_map):
            continue
        if path_blocked_by_known_obstacle(detour_pose, target_x, target_y, world_map):
            continue
        cost = math.hypot(detour_x - sx, detour_y - sy) + math.hypot(target_x - detour_x, target_y - detour_y)
        if cost < best_cost:
            best_cost = cost
            best_path = [(detour_x, detour_y), (target_x, target_y)]

    if best_path is not None:
        print(f"[Nav] obstacle detour via ({best_path[0][0]:.2f}, {best_path[0][1]:.2f})")
        return best_path
    return [(target_x, target_y)]


def navigate_to_point(
    ep_chassis,
    pose: Pose2D,
    target_x: float,
    target_y: float,
    stop_dist_m: float = 0.0,
) -> None:
    navigate_to_point_with_map(ep_chassis, pose, None, target_x, target_y, stop_dist_m=stop_dist_m)


def navigate_to_point_with_map(
    ep_chassis,
    pose: Pose2D,
    world_map: Optional[WorldMap],
    target_x: float,
    target_y: float,
    stop_dist_m: float = 0.0,
) -> None:
    """Turn to face each waypoint then drive straight to it."""
    for waypoint_x, waypoint_y in plan_navigation_points(pose, target_x, target_y, world_map):
        dx = waypoint_x - pose.x
        dy = waypoint_y - pose.y
        distance_m = math.hypot(dx, dy)
        waypoint_stop = stop_dist_m if (waypoint_x, waypoint_y) == (target_x, target_y) else 0.0
        if distance_m <= waypoint_stop:
            continue

        # Heading angle to the waypoint in the CW-positive world convention:
        #   atan2(dy, dx) gives the standard CCW angle, but since our world
        #   has +y downward the atan2 interpretation is already correct:
        #   (dx>0, dy=0) → 0 (right) ✓, (dx=0, dy>0) → π/2 (down) ✓.
        heading_rad = math.atan2(dy, dx)
        turn_to_yaw(ep_chassis, pose, heading_rad)
        move_robot(ep_chassis, pose, x_m=max(0.0, distance_m - waypoint_stop))


# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------

def detect_tags_and_objects(
    frame: np.ndarray,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
) -> Tuple[list, List[Detection]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    try:
        tags = tag_detector.find_tags(gray)
    except Exception:
        tags = []
    detections = get_detections(yolo_model, frame, conf_thresh=0.40)
    show_debug_overlay(frame, tags, detections)
    return tags, detections


def show_debug_overlay(
    frame: np.ndarray,
    tags: Sequence,
    detections: Sequence[Detection],
) -> None:
    overlay = frame.copy()

    for det in detections:
        x1 = int(det.cx - 0.5 * det.w)
        y1 = int(det.cy - 0.5 * det.h)
        x2 = int(det.cx + 0.5 * det.w)
        y2 = int(det.cy + 0.5 * det.h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        distance_m = estimate_bbox_distance_m(det)
        label = f"cls:{det.cls} conf:{det.conf:.2f}"
        if distance_m is not None:
            label += f" d:{distance_m:.2f}m"
        cv2.putText(overlay, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    for tag in tags:
        corners = np.array(tag.corners, dtype=np.int32).reshape((-1, 2))
        cv2.polylines(overlay, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        cx_i = int(tag.center[0])
        cy_i = int(tag.center[1])
        d = AprilTagDetector.tag_distance_m(tag)
        cv2.putText(
            overlay,
            f"id:{int(tag.tag_id)} d:{d:.2f}m",
            (cx_i + 6, cy_i + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

    cv2.imshow(DEBUG_WINDOW_NAME, overlay)
    cv2.waitKey(1)


def find_best_tag(tags: Sequence, valid_ids: Iterable[int]):
    valid_set = set(valid_ids)
    matches = [tag for tag in tags if int(tag.tag_id) in valid_set]
    if not matches:
        return None
    return min(matches, key=AprilTagDetector.tag_distance_m)


def center_error_px(cx_px: float) -> float:
    """Signed pixel error from image centre (+ve = object is to the right)."""
    return float(cx_px) - _CX


# ---------------------------------------------------------------------------
# Landmark mapping via AprilTags
# ---------------------------------------------------------------------------

def landmark_from_tag_detection(tag, kind: str, pose: Pose2D) -> Landmark:
    """Map an AprilTag to a world-frame ground-plane landmark.

    Uses the full 3-D tag pose from pupil_apriltags and projects to the ground
    plane via tag_world_position_from_pose (see that function's docstring for
    the full derivation).  This correctly handles tags that are mounted at an
    elevation above the ground (e.g. stuck to the side of a box).
    """
    world_x, world_y = tag_world_position_from_pose(tag, pose)
    return Landmark(kind=kind, x=world_x, y=world_y, tag_id=int(tag.tag_id))


def compute_tag_reference(tag, goal_kind: str, pose: Pose2D) -> TagReference:
    """Build a TagReference for later re-localisation.

    T_wt is the world-frame pose of the tag, computed by chaining:
        T_world_robot  (from robot dead-reckoned pose, CW yaw)
        T_robot_camera (extrinsics from config.py)
        T_camera_tag   (from pupil_apriltags pose_R / pose_t)
    """
    T_ct = transform_from_rt(
        np.array(tag.pose_R, dtype=float),
        np.array(tag.pose_t, dtype=float),
    )
    T_wr = transform_from_rt(
        rotz_cw(pose.yaw),
        np.array([pose.x, pose.y, 0.0], dtype=float),
    )
    T_wt = T_wr @ T_ROBOT_FROM_CAMERA @ T_ct
    return TagReference(
        tag_id=int(tag.tag_id),
        goal_kind=goal_kind,
        world_x=float(T_wt[0, 3]),
        world_y=float(T_wt[1, 3]),
        world_yaw=yaw_from_rotation_cw(T_wt[:3, :3]),
        reference_pose=copy_pose(pose),
        reference_distance_m=AprilTagDetector.tag_distance_m(tag),
        reference_center_x_px=float(tag.center[0]),
    )


def estimate_pose_from_tag_reference(tag, tag_ref: TagReference) -> Pose2D:
    """Re-localise from a known tag reference.

    Inverts the chain:
        T_world_tag (known) = T_world_robot @ T_robot_camera @ T_camera_tag
    to recover T_world_robot (and hence pose).
    """
    T_ct = transform_from_rt(
        np.array(tag.pose_R, dtype=float),
        np.array(tag.pose_t, dtype=float),
    )
    T_wt = transform_from_rt(
        rotz_cw(tag_ref.world_yaw),
        np.array([tag_ref.world_x, tag_ref.world_y, 0.0], dtype=float),
    )
    T_wc = T_wt @ invert_transform(T_ct)
    T_wr = T_wc @ invert_transform(T_ROBOT_FROM_CAMERA)
    return Pose2D(
        x=float(T_wr[0, 3]),
        y=float(T_wr[1, 3]),
        yaw=wrap_to_pi(yaw_from_rotation_cw(T_wr[:3, :3])),
    )


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def debug_log_tag_mapping(tag, pose: Pose2D, label: str) -> None:
    world_x, world_y = tag_world_position_from_pose(tag, pose)
    t3d = np.array(tag.pose_t, dtype=float).reshape(3)
    R_rc = T_ROBOT_FROM_CAMERA[:3, :3]
    t_rc = T_ROBOT_FROM_CAMERA[:3, 3]
    p_robot = R_rc @ t3d + t_rc
    print(
        f"[Debug][{label}] tag={int(tag.tag_id)} "
        f"robot=({pose.x:.2f},{pose.y:.2f},{math.degrees(pose.yaw):.1f}°) "
        f"p_cam={t3d.round(3).tolist()} "
        f"p_robot=({p_robot[0]:.3f},{p_robot[1]:.3f},{p_robot[2]:.3f}) "
        f"→ world=({world_x:.3f},{world_y:.3f})"
    )


def debug_log_box_mapping(detection: Detection, pose: Pose2D, label: str) -> None:
    distance_m = estimate_bbox_distance_m(detection)
    if distance_m is None:
        return
    bearing_rad = pixel_bearing_rad(detection.cx)
    world_x, world_y = world_from_range_and_bearing(pose, distance_m, bearing_rad)
    v_bot = float(detection.cy) + 0.5 * float(detection.h)
    print(
        f"[Debug][{label}] cls={detection.cls} "
        f"robot=({pose.x:.2f},{pose.y:.2f},{math.degrees(pose.yaw):.1f}°) "
        f"dist={distance_m:.3f}m bearing={math.degrees(bearing_rad):+.1f}° "
        f"bbox_h={float(detection.h):.1f}px v_bot={v_bot:.1f}px cx={float(detection.cx):.1f}px "
        f"→ world=({world_x:.3f},{world_y:.3f})"
    )


# ---------------------------------------------------------------------------
# Servo / localisation helpers
# ---------------------------------------------------------------------------

def capture_intermediate_reference_if_needed(
    world_map: WorldMap,
    goal: Landmark,
    tag,
    pose: Pose2D,
) -> None:
    if world_map.intermediate is not None or world_map.dropoff_tag_ref is not None:
        return
    if goal.kind not in ("small_goal", "large_goal"):
        return
    world_map.intermediate = Landmark(kind="intermediate", x=pose.x, y=pose.y)
    world_map.dropoff_tag_ref = compute_tag_reference(tag, goal.kind, pose)
    print(
        f"[Map] intermediate waypoint at ({pose.x:.2f},{pose.y:.2f}) "
        f"from {goal.kind} tag {int(tag.tag_id)}"
    )

# OLD METHOD OF RELOCALIZATION: separate loops for centering and distance correction, with a proxy pose update only after both are within tolerance.  This can lead to slow convergence if the initial pose error is large in both axes, since the centering loop doesn't correct distance at all, and the distance loop doesn't correct centering at all.  The new method below computes a full pose correction from the tag reference in one shot as soon as the tag is visible, which should be much faster.

# def relocalize_from_dropoff_tag(
#     ep_camera,
#     ep_chassis,
#     yolo_model: YOLO,
#     tag_detector: AprilTagDetector,
#     pose: Pose2D,
#     world_map: WorldMap,
#     timeout_s: float = 6.0,
# ) -> bool:
#     tag_ref = world_map.dropoff_tag_ref
#     if tag_ref is None:
#         return False

#     turn_to_yaw(ep_chassis, pose, tag_ref.reference_pose.yaw)
#     deadline = time.time() + timeout_s
#     while time.time() < deadline:
#         frame = read_frame(ep_camera, timeout=0.5)
#         if frame is None:
#             continue
#         tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
#         tag = find_best_tag(tags, {tag_ref.tag_id})
#         if tag is None:
#             # Small CW turn (yaw increases → z negative in chassis convention).
#             move_robot(ep_chassis, pose, z_deg=-8.0)
#             continue

#         center_delta_px = float(tag.center[0]) - tag_ref.reference_center_x_px
#         dist_delta_m = AprilTagDetector.tag_distance_m(tag) - tag_ref.reference_distance_m
#         if abs(center_delta_px) > REFERENCE_CENTER_TOL_PX:
#             # Positive center_delta → tag is to the right → turn CW (z negative).
#             z_correction = max(-8.0, min(8.0, 0.10 * center_delta_px))
#             move_robot(ep_chassis, pose, z_deg=-z_correction)
#             continue
#         if abs(dist_delta_m) > REFERENCE_DIST_TOL_M:
#             move_robot(ep_chassis, pose, x_m=max(-0.08, min(0.08, dist_delta_m)))
#             continue

#         refined = estimate_pose_from_tag_reference(tag, tag_ref)
#         pose.x, pose.y, pose.yaw = refined.x, refined.y, refined.yaw
#         print(
#             f"[Localize] corrected → ({pose.x:.3f},{pose.y:.3f},"
#             f"{math.degrees(pose.yaw):.1f}°)"
#         )
#         return True

#     print("[Localize] drop-off re-localisation timed out.")
#     return False

def relocalize_from_dropoff_tag(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    timeout_s: float = 6.0,
) -> bool:
    tag_ref = world_map.dropoff_tag_ref
    if tag_ref is None:
        return False

    # Turn to the heading we had when we first saw the tag — most likely
    # to make it visible immediately.
    turn_to_yaw(ep_chassis, pose, tag_ref.reference_pose.yaw)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        tag = find_best_tag(tags, {tag_ref.tag_id})
        if tag is None:
            # Sweep CW slowly until the tag appears.
            move_robot(ep_chassis, pose, z_deg=-8.0)
            continue

        # Tag is visible — compute the full corrected pose directly from
        # the stored world-frame tag pose and the current pose_R / pose_t.
        # This corrects x, y, AND yaw in one shot, with no proxy loop.
        refined = estimate_pose_from_tag_reference(tag, tag_ref)
        pose.x, pose.y, pose.yaw = refined.x, refined.y, refined.yaw
        print(
            f"[Localize] corrected → ({pose.x:.3f}, {pose.y:.3f}, "
            f"{math.degrees(pose.yaw):.1f}°)"
        )
        return True

    print("[Localize] drop-off re-localisation timed out.")
    return False


def go_to_intermediate_waypoint(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    relocalize: bool = True,
) -> None:
    if world_map.intermediate is None:
        return
    distance_m = math.hypot(world_map.intermediate.x - pose.x, world_map.intermediate.y - pose.y)
    if distance_m > INTERMEDIATE_SNAP_DIST_M:
        navigate_to_point_with_map(
            ep_chassis, pose, world_map,
            world_map.intermediate.x, world_map.intermediate.y,
            stop_dist_m=0.0,
        )
    if relocalize:
        relocalize_from_dropoff_tag(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)


def try_refine_recharge_from_tag(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    timeout_s: float = 5.0,
) -> bool:
    if world_map.recharge is not None and world_map.recharge.tag_id is not None:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        tag = find_best_tag(tags, RECHARGE_TAG_IDS)
        if tag is None:
            # Rotate CW (yaw increases) while searching: z_deg negative.
            move_robot(ep_chassis, pose, z_deg=-10.0)
            continue

        debug_log_tag_mapping(tag, pose, "recharge-tag-refine")
        landmark = landmark_from_tag_detection(tag, "recharge", pose)
        world_map.recharge = landmark
        print(f"[Recharge] refined → ({landmark.x:.3f},{landmark.y:.3f}) tag={landmark.tag_id}")
        return True

    print("[Recharge] recharge tag not found; keeping coarse position.")
    return False


def wait_for_goal_tag(
    ep_camera,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    valid_ids: Iterable[int],
    timeout_s: float = 5.0,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        tag = find_best_tag(tags, valid_ids)
        if tag is not None:
            return frame, tag
    return None, None


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def map_goal_from_view(
    ep_camera,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    valid_ids: Iterable[int],
    label: str,
) -> Landmark:
    frame, tag = wait_for_goal_tag(ep_camera, yolo_model, tag_detector, valid_ids, timeout_s=6.0)
    if tag is None:
        raise RuntimeError(f"Could not detect {label} goal tag.")
    tag_id = int(tag.tag_id)
    goal_kind = goal_kind_from_tag(tag_id)
    if goal_kind is None:
        raise RuntimeError(f"Tag {tag_id} is not configured as a goal tag.")
    debug_log_tag_mapping(tag, pose, label)
    landmark = landmark_from_tag_detection(tag, goal_kind, pose)
    landmark = world_map.set_goal(goal_kind, landmark.x, landmark.y, tag_id)
    print(f"[Map] {label}: {goal_kind} at ({landmark.x:.3f},{landmark.y:.3f}) tag={tag_id}")
    return landmark


def map_recharge_from_box(
    ep_camera,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    """Map the recharge station by detecting the visible box face with YOLO."""
    deadline = time.time() + 6.0
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        _, detections = detect_tags_and_objects(frame, yolo_model, tag_detector)
        boxes = [det for det in detections if det.cls == CLASS_BOX]
        if not boxes:
            continue
        selected = min(boxes, key=lambda det: abs(center_error_px(det.cx)))
        if abs(center_error_px(selected.cx)) > CENTER_TOL_PX:
            continue
        debug_log_box_mapping(selected, pose, "recharge-box")
        world_pos = detection_world_position(selected, pose)
        if world_pos is None:
            continue
        world_map.recharge = Landmark(kind="recharge", x=world_pos[0], y=world_pos[1])
        print(f"[Map] recharge box at ({world_pos[0]:.3f},{world_pos[1]:.3f})")
        return world_map.recharge
    raise RuntimeError("Could not map the recharge box.")


def scan_left_and_map_world(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    required_obstacles: int = 2,
) -> None:
    """Translate leftward across the workspace, mapping both goal zones and
    the obstacles encountered during the sweep.

    'Left' in the robot's own frame (body +y = right, so body −y = left).
    chassis.move(y=−step) moves the robot leftward.

    With the robot facing +y (downward, yaw=π/2), robot-left is world +x
    (toward the right wall), which is the correct sweep direction to cover the
    workspace from the starting corner.
    """
    total_left_m = 0.0
    while total_left_m < WORKSPACE_W_M:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is not None:
            tags, detections = detect_tags_and_objects(frame, yolo_model, tag_detector)

            for tag in tags:
                tag_id = int(tag.tag_id)
                goal_kind = goal_kind_from_tag(tag_id)
                if goal_kind is not None:
                    if world_map.goal_for_kind(goal_kind) is not None:
                        continue
                    if abs(center_error_px(float(tag.center[0]))) > CENTER_TOL_PX:
                        continue
                    debug_log_tag_mapping(tag, pose, f"{goal_kind}-sweep")
                    lm = landmark_from_tag_detection(tag, goal_kind, pose)
                    lm = world_map.set_goal(goal_kind, lm.x, lm.y, tag_id)
                    capture_intermediate_reference_if_needed(world_map, lm, tag, pose)
                    print(f"[Map] {goal_kind} at ({lm.x:.3f},{lm.y:.3f}) tag={tag_id}")
                    continue

                if len(world_map.obstacles) >= required_obstacles:
                    continue
                if not is_obstacle_tag(tag_id):
                    continue
                debug_log_tag_mapping(tag, pose, "obstacle-tag")
                lm = landmark_from_tag_detection(tag, "obstacle", pose)
                obs = world_map.add_or_update_obstacle(lm.x, lm.y, tag_id=tag_id)
                print(f"[Map] obstacle tag {tag_id} at ({obs.x:.3f},{obs.y:.3f})")

            if len(world_map.obstacles) < required_obstacles:
                boxes = [det for det in detections if det.cls == CLASS_BOX]
                centered = [det for det in boxes if abs(center_error_px(det.cx)) <= CENTER_TOL_PX]
                for box in centered:
                    debug_log_box_mapping(box, pose, "obstacle-box-fallback")
                    world_pos = detection_world_position(box, pose)
                    if world_pos is None:
                        continue
                    if world_map.recharge is not None:
                        if math.hypot(world_pos[0] - world_map.recharge.x, world_pos[1] - world_map.recharge.y) < 0.40:
                            continue
                    obs = world_map.add_or_update_obstacle(world_pos[0], world_pos[1])
                    print(f"[Map] obstacle fallback at ({obs.x:.3f},{obs.y:.3f})")
                    if len(world_map.obstacles) >= required_obstacles:
                        break

            if (
                world_map.small_goal is not None
                and world_map.large_goal is not None
                and len(world_map.obstacles) >= required_obstacles
            ):
                print("[Map] left sweep complete: both goals and required obstacles mapped")
                return

        # Move left in robot body frame: chassis y_m = -step (body +y = right,
        # so −y = left).  This translates the robot toward the +x world
        # direction when facing down (yaw=π/2).
        move_robot(ep_chassis, pose, y_m=-LEFT_SCAN_STEP_M)
        total_left_m += LEFT_SCAN_STEP_M

    raise RuntimeError("Could not map both goals and the required obstacles while translating left.")


def map_loading_dock(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    """Find the loading dock by locating the tower brick clump with YOLO."""
    for _ in range(MAX_TAG_SEARCH_STEPS):
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is not None:
            _, detections = detect_tags_and_objects(frame, yolo_model, tag_detector)
            bricks = [det for det in detections if det.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
            if bricks:
                points: List[Tuple[float, float]] = []
                for brick in bricks:
                    world_pos = detection_world_position(brick, pose)
                    if world_pos is not None:
                        points.append(world_pos)
                if points:
                    dock_x = float(np.mean([p[0] for p in points]))
                    dock_y = float(np.mean([p[1] for p in points]))
                    world_map.dock = Landmark(kind="dock", x=dock_x, y=dock_y)
                    print(f"[Map] dock at ({dock_x:.3f},{dock_y:.3f}) from {len(points)} towers")
                    return world_map.dock

        # Rotate CCW (yaw decreases) to sweep: chassis z positive.
        move_robot(ep_chassis, pose, z_deg=DOCK_SEARCH_STEP_DEG)

    raise RuntimeError("Could not locate the loading dock tower clump.")


# ---------------------------------------------------------------------------
# Tag servo
# ---------------------------------------------------------------------------

def servo_to_visible_tag(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    valid_ids: Iterable[int],
    target_dist_m: float,
    timeout_s: float = 15.0,
) -> bool:
    """Close-range visual servo to an AprilTag using drive_speed.

    Yaw correction:
      A positive center error (tag right of image centre) means the robot must
      turn CW (yaw increases) to centre the tag.  drive_speed(z=+ω) is CCW
      (yaw decreases), so to turn CW we use a negative z value:
          wz = -K * err_px   (negative when tag is to the right)
    """
    valid_set = set(valid_ids)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        matches = [tag for tag in tags if int(tag.tag_id) in valid_set]
        if not matches:
            # Slowly sweep CW (yaw increases): z negative for drive_speed.
            wz = -10.0
            dt = 0.15
            ep_chassis.drive_speed(x=0.0, y=0.0, z=wz, timeout=dt)
            time.sleep(dt)
            integrate_drive_speed(pose, 0.0, 0.0, wz, dt)
            continue

        tag = min(matches, key=tag_detector.tag_distance_m)
        err_px = center_error_px(float(tag.center[0]))
        err_dist_m = tag_detector.tag_distance_m(tag) - target_dist_m
        if abs(err_px) < 18.0 and abs(err_dist_m) < 0.04:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
            time.sleep(0.1)
            return True

        vx = max(-0.16, min(0.16, 0.6 * err_dist_m))
        # Positive err_px → tag to the right → turn CW → z negative.
        wz = max(-30.0, min(30.0, -0.08 * err_px))
        dt = 0.15
        ep_chassis.drive_speed(x=vx, y=0.0, z=wz, timeout=dt)
        time.sleep(dt)
        integrate_drive_speed(pose, vx, 0.0, wz, dt)

    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    return False


# ---------------------------------------------------------------------------
# Brick approach
# ---------------------------------------------------------------------------

def approach_brick_with_move(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    pose: Pose2D,
    target_class: int,
    timeout_s: float = 20.0,
) -> bool:
    """Servo to a specific brick class at the loading dock using move()."""
    deadline = time.time() + timeout_s
    stable = 0
    desired_height_px = 160.0

    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        detections = get_detections(yolo_model, frame, conf_thresh=0.40, target_class=target_class)
        if not detections:
            # Small CW search turn: z_deg negative.
            move_robot(ep_chassis, pose, z_deg=-10.0)
            continue

        selected = max(detections, key=lambda det: det.conf)
        err_x = center_error_px(selected.cx)
        err_h = desired_height_px - float(selected.h)

        if abs(err_x) < 20.0 and abs(err_h) < 12.0:
            stable += 1
            if stable >= 3:
                pick_up_tower(ep_robot=ep_robot)
                return True
        else:
            stable = 0

        forward_step_m = max(-0.08, min(0.08, 0.0014 * err_h))
        # Positive err_x → object to robot's right → move right (y_m positive).
        lateral_step_m = max(-0.05, min(0.05, 0.0009 * err_x))
        if abs(forward_step_m) < 0.01 and abs(lateral_step_m) < 0.01:
            forward_step_m = 0.02 if err_h > 0 else -0.02
        move_robot(ep_chassis, pose, x_m=forward_step_m, y_m=lateral_step_m)

    return False


# ---------------------------------------------------------------------------
# Goal delivery
# ---------------------------------------------------------------------------

def align_to_goal_and_drop(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    goal: Landmark,
) -> None:
    goal_ids = SMALL_GOAL_TAG_IDS if goal.kind == "small_goal" else LARGE_GOAL_TAG_IDS
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(ep_chassis, pose, world_map, goal.x, goal.y, stop_dist_m=LANDMARK_STOP_DIST_M)
    success = servo_to_visible_tag(
        ep_camera, ep_chassis, yolo_model, tag_detector, pose,
        goal_ids, target_dist_m=GOAL_SERVO_DIST_M,
    )
    if not success:
        print("[Goal] Tag servo timed out; placing based on mapped position.")
    place_down_tower(ep_robot=ep_robot)
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)


# ---------------------------------------------------------------------------
# Recharge
# ---------------------------------------------------------------------------

def recharge_robot(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    battery: BatteryManager,
) -> None:
    if world_map.recharge is None:
        raise RuntimeError("Recharge requested before recharge was mapped.")
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(
        ep_chassis, pose, world_map,
        world_map.recharge.x, world_map.recharge.y,
        stop_dist_m=LANDMARK_STOP_DIST_M,
    )
    try_refine_recharge_from_tag(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(
        ep_chassis, pose, world_map,
        world_map.recharge.x, world_map.recharge.y,
        stop_dist_m=max(RECHARGE_SERVO_DIST_M + 0.10, 0.30),
    )
    success = servo_to_visible_tag(
        ep_camera, ep_chassis, yolo_model, tag_detector, pose,
        RECHARGE_TAG_IDS, target_dist_m=RECHARGE_SERVO_DIST_M,
    )
    if not success:
        print("[Recharge] Tag servo timed out; holding at mapped location.")
    print("[Recharge] Holding to simulate recharge...")
    time.sleep(5.0)
    battery.recharge()
    print(f"[Recharge] Battery now {battery.level:.0f}%")
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)


# ---------------------------------------------------------------------------
# Mission sequences
# ---------------------------------------------------------------------------

def execute_mapping_sequence(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    """Deterministic mapping path.

    Start state: robot at (0.20, 0.20), yaw = 0 (facing +x / right).

    Step 1: Without moving, map the recharge block directly ahead.
    Step 2: Turn 90° CW to face downward into the arena.
    Step 3: Move 2 ft forward.
    Step 4: Translate left while mapping both goal zones and the two obstacles.
    Step 5: Turn 180° to face upward and map the loading dock.
    """
    # Step 1 — map the recharge block directly ahead at the start pose.
    map_recharge_from_box(ep_camera, yolo_model, tag_detector, pose, world_map)

    # Step 2 — turn CW 90° so the robot faces +y (downward into the arena).
    # In the chassis convention, CW is z_deg negative.
    move_robot(ep_chassis, pose, z_deg=-90.0)

    # Step 3 — move forward 2 ft (deeper into the arena, +y world direction).
    move_robot(ep_chassis, pose, x_m=INITIAL_FORWARD_STEP_M)

    # Step 4 — translate left while scanning.
    scan_left_and_map_world(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    print(f"[Map] blocks mapped so far: {world_map.mapped_block_count()}")

    # Step 5 — turn 180° to face upward (+y → -y, i.e. yaw = π/2 → -π/2).
    # 180° CCW: chassis.move(z=+180) → yaw −= π → yaw = π/2 - π = -π/2 ✓.
    move_robot(ep_chassis, pose, z_deg=180.0)
    map_loading_dock(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)

    target_goal = world_map.right_side_goal()
    if target_goal is None:
        raise RuntimeError("No drop-off goal was mapped.")
    if world_map.intermediate is None:
        world_map.intermediate = Landmark(kind="intermediate", x=pose.x, y=pose.y)
        print(f"[Map] fallback intermediate at ({pose.x:.3f},{pose.y:.3f})")
    print(f"[Mission] right-side goal: {target_goal.kind} at ({target_goal.x:.3f},{target_goal.y:.3f})")
    return target_goal


def run_delivery_loop(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    battery: BatteryManager,
    target_goal: Landmark,
    max_deliveries: int,
) -> None:
    if world_map.dock is None:
        raise RuntimeError("Delivery loop started before loading dock was mapped.")
    target_class = brick_class_for_goal(target_goal)
    print(f"[Mission] Dock → {target_goal.kind} loop, brick class {target_class}")

    deliveries = 0
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    while deliveries < max_deliveries:
        if not battery.can_pick(target_class):
            recharge_robot(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map, battery)

        go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
        navigate_to_point_with_map(
            ep_chassis, pose, world_map,
            world_map.dock.x, world_map.dock.y,
            stop_dist_m=0.50,
        )
        success = approach_brick_with_move(ep_robot, ep_camera, ep_chassis, yolo_model, pose, target_class)
        if not success:
            raise RuntimeError("Could not approach the requested brick class at the loading dock.")

        battery.consume(target_class)
        print(f"[Battery] After pickup: {battery.level:.0f}%")
        go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
        align_to_goal_and_drop(ep_robot, ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map, target_goal)
        deliveries += 1
        print(f"[Mission] Delivery {deliveries}/{max_deliveries} complete")


# ---------------------------------------------------------------------------
# Visualisation (debug)
# ---------------------------------------------------------------------------

def visualize_map(world_map: WorldMap, robot_pose: Optional[Pose2D] = None) -> None:
    """Bird's-eye debug plot.  World +y is rendered downward to match the
    workspace convention (+x right, +y down, origin top-left).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Map] matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0.0, WORKSPACE_W_M)
    ax.set_ylim(WORKSPACE_H_M, 0.0)   # ← inverted so +y goes downward
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  →  rightward")
    ax.set_ylabel("y (m)  ↓  downward")
    ax.set_title("Project 3 Map (top-left origin, +y down)")

    for v in np.arange(0.0, WORKSPACE_W_M + 0.01, 0.10):
        ax.axvline(v, color="lightgray", linewidth=0.3)
    for v in np.arange(0.0, WORKSPACE_H_M + 0.01, 0.10):
        ax.axhline(v, color="lightgray", linewidth=0.3)

    ax.plot(
        [0, WORKSPACE_W_M, WORKSPACE_W_M, 0, 0],
        [0, 0, WORKSPACE_H_M, WORKSPACE_H_M, 0],
        "k-", linewidth=2,
    )

    for obs in world_map.obstacles:
        ax.add_patch(plt.Circle((obs.x, obs.y), 0.12, color="red", alpha=0.6))

    if world_map.recharge:
        ax.add_patch(patches.Rectangle(
            (world_map.recharge.x - 0.10, world_map.recharge.y - 0.10),
            0.20, 0.20, facecolor="black", edgecolor="black", alpha=0.8,
        ))

    if world_map.small_goal:
        ax.plot(world_map.small_goal.x, world_map.small_goal.y, "b^", markersize=14, label="small_goal")
    if world_map.large_goal:
        ax.plot(world_map.large_goal.x, world_map.large_goal.y, "g^", markersize=14, label="large_goal")
    if world_map.dock:
        ax.add_patch(patches.Rectangle(
            (world_map.dock.x - 0.15, world_map.dock.y - 0.15),
            0.30, 0.30, facecolor="yellow", edgecolor="goldenrod", alpha=0.8,
        ))
    if world_map.intermediate:
        ax.plot(world_map.intermediate.x, world_map.intermediate.y, "co", markersize=10, label="intermediate")

    if robot_pose is not None:
        ax.plot(robot_pose.x, robot_pose.y, "ms", markersize=10, label="robot")
        # Arrow in world direction of current yaw.
        # With +y downward axes, cos(yaw) gives Δx (rightward) and sin(yaw)
        # gives Δy (downward) — both consistent with matplotlib's natural axes
        # when ylim is inverted.
        dx = 0.15 * math.cos(robot_pose.yaw)
        dy = 0.15 * math.sin(robot_pose.yaw)
        ax.annotate(
            "",
            xy=(robot_pose.x + dx, robot_pose.y + dy),
            xytext=(robot_pose.x, robot_pose.y),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=2),
        )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("arena_map.png", dpi=150)
    print("[Map] Saved arena_map.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3 deterministic mapping and delivery")
    parser.add_argument("--model-path", default=str(MODEL_PATH or DEFAULT_MODEL_PATH))
    parser.add_argument("--robot-ip", default=ROBOT_IP or DEFAULT_ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN or DEFAULT_ROBOT_SN)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"])
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--show-map", action="store_true", default=True,
                        help="Show and save the generated map (default: True)")
    parser.add_argument("--max-deliveries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Project 3 Updated Workflow ===")
    print(f"[Setup] workspace = {WORKSPACE_W_M:.3f} m × {WORKSPACE_H_M:.3f} m")
    print(f"[Setup] start pose = ({START_X_M:.3f}, {START_Y_M:.3f}, {math.degrees(START_YAW_RAD):.1f}°)")
    print(f"[Setup] camera height = {_CAM_HEIGHT_M:.3f} m, pitch = {CAM_PITCH_DEG:.1f}°")

    yolo_model = YOLO(str(args.model_path))
    tag_detector = AprilTagDetector()
    pose = Pose2D(x=START_X_M, y=START_Y_M, yaw=START_YAW_RAD)
    world_map = WorldMap()
    battery = BatteryManager()

    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = str(args.robot_ip)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=str(args.sn))
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    resolution = rm_camera.STREAM_720P if args.resolution == "720p" else rm_camera.STREAM_360P
    ep_camera.start_video_stream(display=False, resolution=resolution)

    try:
        ep_robot.robotic_arm.moveto(x=DEFAULT_ARM_X, y=DEFAULT_ARM_Y).wait_for_completed()
        ep_robot.gripper.open()
        time.sleep(1.0)
        ep_robot.gripper.pause()

        target_goal = execute_mapping_sequence(
            ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map,
        )
        print(world_map.summary())

        # Generate and save the map immediately after the mapping sequence
        # completes so the saved PNG reflects the mapped obstacles/loading dock.
        if args.show_map or args.map_only:
            visualize_map(world_map, pose)

        if not args.map_only:
            run_delivery_loop(
                ep_robot, ep_camera, ep_chassis,
                yolo_model, tag_detector,
                pose, world_map, battery,
                target_goal, args.max_deliveries,
            )

    finally:
        try:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
        except Exception:
            pass
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
