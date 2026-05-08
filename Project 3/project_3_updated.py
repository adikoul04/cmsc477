#!/usr/bin/env python3
"""
Project 3 deterministic mapping + delivery workflow.

This rewrite follows the updated arena information from the user:
- Workspace is 10 ft x 10 ft => 3.048 m x 3.048 m.
- World origin is the top-left corner.
- World +x points right and world +y points down.
- Robot starts near (1 ft, 1 ft) facing downward (+y).
- Planned motion uses `chassis.move(...)` whenever possible.
- `drive_speed(...)` is only used while servoing to an AprilTag.

Notes:
- This file keeps robot pose `(x, y, yaw)` updated after every motion command.
- AprilTag IDs still come from `config.py`.
- Distance estimates for boxes / tower clumps reuse the same calibrated
  bbox-height model currently used in `live_feed.py`.
- The example setup PDFs could not be programmatically extracted in this
  environment, so this implementation follows the user's detailed setup
  description plus the existing project code.
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
# The arena dimensions and start pose are hard-coded from the updated setup.
WORKSPACE_W_M = 10.0 * FT_TO_M
WORKSPACE_H_M = 10.0 * FT_TO_M
# The robot starts in the top-left corner with a small safety margin from the
# boundary, facing downward so the lower-left goal is visible first.
START_X_M = 0.20
START_Y_M = 0.20
START_YAW_RAD = math.pi / 2.0
INITIAL_FORWARD_STEP_M = 2.0 * FT_TO_M
LEFT_SCAN_STEP_M = 0.18
CENTER_TOL_PX = 35.0
OBSTACLE_MERGE_RADIUS_M = 0.30
OBSTACLE_PATH_CLEARANCE_M = 0.38
OBSTACLE_DETOUR_MARGIN_M = 0.28
LANDMARK_STOP_DIST_M = 0.55
GOAL_SERVO_DIST_M = 0.30
RECHARGE_SERVO_DIST_M = 0.22
DOCK_SEARCH_STEP_DEG = 15.0
MAX_TAG_SEARCH_STEPS = 24
REFERENCE_CENTER_TOL_PX = 25.0
REFERENCE_DIST_TOL_M = 0.18
INTERMEDIATE_SNAP_DIST_M = 0.12

# Calibrated distance model copied from live_feed.py.
OBJECT_HEIGHTS_M = {
    CLASS_BOX: 0.28,
    CLASS_SMALL_BRICK: 0.10,
    CLASS_LARGE_BRICK: 0.19,
}
RAW_DISTANCE_SCALE = 2.0
DISTANCE_CORRECTION_SLOPE = 1.1971830985915493
DISTANCE_CORRECTION_OFFSET = -0.27464788732394363


@dataclass
class Pose2D:
    """Robot pose in the global workspace frame."""
    x: float
    y: float
    yaw: float


@dataclass
class Landmark:
    """A mapped world object represented by a single 2-D point."""
    kind: str
    x: float
    y: float
    tag_id: Optional[int] = None


@dataclass
class TagReference:
    """Reference drop-off tag view used to re-localize from the relay point."""
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
        # The course only has one small-goal zone and one large-goal zone, so
        # later observations overwrite earlier ones for that specific goal kind.
        landmark = Landmark(kind=kind, x=x, y=y, tag_id=tag_id)
        if kind == "small_goal":
            self.small_goal = landmark
        else:
            self.large_goal = landmark
        return landmark

    def add_or_update_obstacle(self, x: float, y: float, tag_id: Optional[int] = None) -> Landmark:
        # Obstacle tags provide persistent identity, so prefer matching by tag
        # before falling back to geometric merge-by-distance.
        if tag_id is not None:
            for obstacle in self.obstacles:
                if obstacle.tag_id == tag_id:
                    obstacle.x = x
                    obstacle.y = y
                    return obstacle
        # Obstacles are otherwise merged when repeated observations land nearby.
        for obstacle in self.obstacles:
            if math.hypot(obstacle.x - x, obstacle.y - y) <= OBSTACLE_MERGE_RADIUS_M:
                obstacle.x = 0.5 * (obstacle.x + x)
                obstacle.y = 0.5 * (obstacle.y + y)
                if tag_id is not None:
                    obstacle.tag_id = tag_id
                return obstacle
        obstacle = Landmark(kind="obstacle", x=x, y=y, tag_id=tag_id)
        self.obstacles.append(obstacle)
        return obstacle

    def right_side_goal(self) -> Optional[Landmark]:
        # The updated mission description uses the right-side mapped goal as the
        # repeated delivery target.
        goals = [goal for goal in [self.small_goal, self.large_goal] if goal is not None]
        if not goals:
            return None
        return max(goals, key=lambda goal: goal.x)

    def mapped_block_count(self) -> int:
        count = len(self.obstacles)
        count += 1 if self.recharge is not None else 0
        count += 1 if self.small_goal is not None else 0
        count += 1 if self.large_goal is not None else 0
        return count

    def summary(self) -> str:
        def fmt(landmark: Optional[Landmark]) -> str:
            if landmark is None:
                return "NOT FOUND"
            return f"({landmark.x:.2f}, {landmark.y:.2f})"

        return "\n".join(
            [
                "=== WorldMap ===",
                f"  recharge: {fmt(self.recharge)}",
                f"  small_goal: {fmt(self.small_goal)}",
                f"  large_goal: {fmt(self.large_goal)}",
                f"  dock: {fmt(self.dock)}",
                f"  intermediate: {fmt(self.intermediate)}",
                f"  obstacles: {len(self.obstacles)}",
            ]
        )


class BatteryManager:
    """Minimal simulated battery model driven by the brick class being carried."""
    def __init__(self, start_pct: float = BATTERY_START_PCT):
        self.level = float(start_pct)

    def cost_for_class(self, brick_class: int) -> float:
        return (
            BATTERY_LARGE_BRICK_COST
            if brick_class == CLASS_LARGE_BRICK
            else BATTERY_SMALL_BRICK_COST
        )

    def can_pick(self, brick_class: int) -> bool:
        return self.level - self.cost_for_class(brick_class) > 0.0

    def consume(self, brick_class: int) -> None:
        self.level = max(0.0, self.level - self.cost_for_class(brick_class))

    def recharge(self) -> None:
        self.level = float(BATTERY_RECHARGE_LEVEL)


class AprilTagDetector:
    """Small wrapper so the rest of the file stays agnostic to detector details."""
    def __init__(self, K: np.ndarray = K_CAM, family: str = TAG_FAMILY, marker_size_m: float = TAG_SIZE_M):
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

    def find_tags(self, gray: np.ndarray):
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )

    @staticmethod
    def tag_distance_m(detection) -> float:
        return float(np.linalg.norm(np.array(detection.pose_t, dtype=float).reshape(3)))


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


def rotz(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
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


def yaw_from_rotation(R: np.ndarray) -> float:
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def read_frame(ep_camera, timeout: float = 1.0) -> Optional[np.ndarray]:
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=timeout)
    except Empty:
        return None
    return frame


def pixel_bearing_rad(cx_px: float, camera_matrix: np.ndarray = K_CAM) -> float:
    return math.atan2(float(cx_px) - float(camera_matrix[0, 2]), float(camera_matrix[0, 0]))


def world_from_range_and_bearing(pose: Pose2D, range_m: float, bearing_rad: float) -> Tuple[float, float]:
    heading = wrap_to_pi(pose.yaw + bearing_rad)
    world_x = pose.x + range_m * math.cos(heading)
    world_y = pose.y + range_m * math.sin(heading)
    return world_x, world_y


def copy_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(x=pose.x, y=pose.y, yaw=pose.yaw)


def clamp_to_workspace(x: float, y: float) -> Tuple[float, float]:
    return (
        min(max(0.0, x), WORKSPACE_W_M),
        min(max(0.0, y), WORKSPACE_H_M),
    )


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def path_blocked_by_known_obstacle(start: Pose2D, target_x: float, target_y: float, world_map: Optional[WorldMap]) -> bool:
    if world_map is None:
        return False
    for obstacle in world_map.obstacles:
        if point_to_segment_distance(obstacle.x, obstacle.y, start.x, start.y, target_x, target_y) < OBSTACLE_PATH_CLEARANCE_M:
            return True
    return False


def plan_navigation_points(start: Pose2D, target_x: float, target_y: float, world_map: Optional[WorldMap]) -> List[Tuple[float, float]]:
    if world_map is None or not world_map.obstacles:
        return [(target_x, target_y)]
    if not path_blocked_by_known_obstacle(start, target_x, target_y, world_map):
        return [(target_x, target_y)]

    sx, sy = start.x, start.y
    dx = target_x - sx
    dy = target_y - sy
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
    candidates: List[Tuple[float, float]] = []
    for sign in (1.0, -1.0):
        detour_x = obstacle.x + sign * perp_x * detour_radius
        detour_y = obstacle.y + sign * perp_y * detour_radius
        detour_x, detour_y = clamp_to_workspace(detour_x, detour_y)
        candidates.append((detour_x, detour_y))

    best_path: Optional[List[Tuple[float, float]]] = None
    best_cost = float("inf")
    for detour_x, detour_y in candidates:
        detour_pose = Pose2D(x=detour_x, y=detour_y, yaw=start.yaw)
        if path_blocked_by_known_obstacle(start, detour_x, detour_y, world_map):
            continue
        if path_blocked_by_known_obstacle(detour_pose, target_x, target_y, world_map):
            continue
        cost = (
            math.hypot(detour_x - sx, detour_y - sy)
            + math.hypot(target_x - detour_x, target_y - detour_y)
        )
        if cost < best_cost:
            best_cost = cost
            best_path = [(detour_x, detour_y), (target_x, target_y)]

    if best_path is not None:
        print(f"[Nav] planned obstacle detour via ({best_path[0][0]:.2f}, {best_path[0][1]:.2f})")
        return best_path
    return [(target_x, target_y)]


def estimate_bbox_distance_m(detection: Detection, cls: Optional[int] = None) -> Optional[float]:
    # This mirrors the live-feed calibration so the map and the overlay use the
    # same object-to-distance relationship.
    object_class = detection.cls if cls is None else cls
    object_height_m = OBJECT_HEIGHTS_M.get(object_class)
    if object_height_m is None:
        return None
    pixel_height = max(1.0, float(detection.h))
    raw_distance_m = RAW_DISTANCE_SCALE * float(K_CAM[1, 1]) * object_height_m / pixel_height
    corrected_distance_m = DISTANCE_CORRECTION_SLOPE * raw_distance_m + DISTANCE_CORRECTION_OFFSET
    return max(0.0, corrected_distance_m)


def detection_world_position(detection: Detection, pose: Pose2D) -> Optional[Tuple[float, float]]:
    # Convert a centered camera observation into a 2-D world point using the
    # current dead-reckoned pose plus the calibrated range estimate.
    distance_m = estimate_bbox_distance_m(detection)
    if distance_m is None:
        return None
    bearing_rad = pixel_bearing_rad(detection.cx)
    return world_from_range_and_bearing(pose, distance_m, bearing_rad)


def debug_log_tag_mapping(tag, pose: Pose2D, label: str) -> None:
    distance_m = AprilTagDetector.tag_distance_m(tag)
    bearing_rad = pixel_bearing_rad(float(tag.center[0]))
    world_x, world_y = world_from_range_and_bearing(pose, distance_m, bearing_rad)
    print(
        f"[Debug][{label}] tag={int(tag.tag_id)} "
        f"pose=({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):.1f} deg) "
        f"dist={distance_m:.2f}m bearing={math.degrees(bearing_rad):+.1f} deg "
        f"-> world=({world_x:.2f}, {world_y:.2f})"
    )


def debug_log_box_mapping(detection: Detection, pose: Pose2D, label: str) -> None:
    distance_m = estimate_bbox_distance_m(detection)
    if distance_m is None:
        return
    bearing_rad = pixel_bearing_rad(detection.cx)
    world_x, world_y = world_from_range_and_bearing(pose, distance_m, bearing_rad)
    print(
        f"[Debug][{label}] cls={detection.cls} "
        f"pose=({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):.1f} deg) "
        f"dist={distance_m:.2f}m bearing={math.degrees(bearing_rad):+.1f} deg "
        f"bbox_h={float(detection.h):.1f}px center_x={float(detection.cx):.1f}px "
        f"-> world=({world_x:.2f}, {world_y:.2f})"
    )


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
    # `chassis.move(...)` is the preferred motion primitive for planned travel
    # because it drifts less than `drive_speed(...)` on this robot.
    ep_chassis.move(x=x_m, y=y_m, z=z_deg, xy_speed=xy_speed, z_speed=z_speed).wait_for_completed()

    # RoboMaster translations are commanded in the robot body frame, so we
    # rotate that motion into the global workspace frame before updating pose.
    world_dx = x_m * math.cos(pose.yaw) - y_m * math.sin(pose.yaw)
    world_dy = x_m * math.sin(pose.yaw) + y_m * math.cos(pose.yaw)
    pose.x += world_dx
    pose.y += world_dy
    pose.yaw = wrap_to_pi(pose.yaw + math.radians(z_deg))
    print(f"[Pose] x={pose.x:.2f} y={pose.y:.2f} yaw={math.degrees(pose.yaw):.1f} deg")


def integrate_drive_speed(pose: Pose2D, vx: float, vy: float, wz_deg_s: float, dt_s: float) -> None:
    # Only used during tag servo, where motion is applied as short velocity
    # bursts instead of a single `move(...)` command.
    world_dx = (vx * math.cos(pose.yaw) - vy * math.sin(pose.yaw)) * dt_s
    world_dy = (vx * math.sin(pose.yaw) + vy * math.cos(pose.yaw)) * dt_s
    pose.x += world_dx
    pose.y += world_dy
    pose.yaw = wrap_to_pi(pose.yaw + math.radians(wz_deg_s * dt_s))


def turn_to_yaw(ep_chassis, pose: Pose2D, target_yaw_rad: float) -> None:
    # Turn with the high-level move command so the yaw update stays consistent
    # with the global pose tracker.
    delta_deg = math.degrees(wrap_to_pi(target_yaw_rad - pose.yaw))
    if abs(delta_deg) < 1.0:
        return
    move_robot(ep_chassis, pose, z_deg=delta_deg)


def navigate_to_point(ep_chassis, pose: Pose2D, target_x: float, target_y: float, stop_dist_m: float = 0.0) -> None:
    navigate_to_point_with_map(ep_chassis, pose, None, target_x, target_y, stop_dist_m=stop_dist_m)


def navigate_to_point_with_map(
    ep_chassis,
    pose: Pose2D,
    world_map: Optional[WorldMap],
    target_x: float,
    target_y: float,
    stop_dist_m: float = 0.0,
) -> None:
    for waypoint_x, waypoint_y in plan_navigation_points(pose, target_x, target_y, world_map):
        dx = waypoint_x - pose.x
        dy = waypoint_y - pose.y
        distance_m = math.hypot(dx, dy)
        waypoint_stop = stop_dist_m if (waypoint_x, waypoint_y) == (target_x, target_y) else 0.0
        if distance_m <= waypoint_stop:
            continue
        heading_rad = math.atan2(dy, dx)
        turn_to_yaw(ep_chassis, pose, heading_rad)
        move_robot(ep_chassis, pose, x_m=max(0.0, distance_m - waypoint_stop))


def detect_tags_and_objects(frame: np.ndarray, yolo_model: YOLO, tag_detector: AprilTagDetector) -> Tuple[list, List[Detection]]:
    # Mapping and servo loops usually need both detectors on the same frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    try:
        tags = tag_detector.find_tags(gray)
    except Exception:
        tags = []
    detections = get_detections(yolo_model, frame, conf_thresh=0.40)
    return tags, detections


def find_best_tag(tags: Sequence, valid_ids: Iterable[int]):
    valid_set = set(valid_ids)
    matches = [tag for tag in tags if int(tag.tag_id) in valid_set]
    if not matches:
        return None
    return min(matches, key=AprilTagDetector.tag_distance_m)


def center_error_px(cx_px: float) -> float:
    return float(cx_px) - float(K_CAM[0, 2])


def compute_tag_reference(tag, goal_kind: str, pose: Pose2D) -> TagReference:
    T_ct = transform_from_rt(np.array(tag.pose_R, dtype=float), np.array(tag.pose_t, dtype=float))
    T_wr = transform_from_rt(rotz(pose.yaw), np.array([pose.x, pose.y, 0.0], dtype=float))
    T_wt = T_wr @ T_ROBOT_FROM_CAMERA @ T_ct
    return TagReference(
        tag_id=int(tag.tag_id),
        goal_kind=goal_kind,
        world_x=float(T_wt[0, 3]),
        world_y=float(T_wt[1, 3]),
        world_yaw=yaw_from_rotation(T_wt[:3, :3]),
        reference_pose=copy_pose(pose),
        reference_distance_m=AprilTagDetector.tag_distance_m(tag),
        reference_center_x_px=float(tag.center[0]),
    )


def tag_world_position_from_pose(tag, pose: Pose2D) -> Tuple[float, float]:
    """Map a visible tag using current robot pose plus tag distance/bearing."""
    distance_m = AprilTagDetector.tag_distance_m(tag)
    bearing_rad = pixel_bearing_rad(float(tag.center[0]))
    return world_from_range_and_bearing(pose, distance_m, bearing_rad)


def landmark_from_tag_detection(tag, kind: str, pose: Pose2D) -> Landmark:
    world_x, world_y = tag_world_position_from_pose(tag, pose)
    return Landmark(
        kind=kind,
        x=world_x,
        y=world_y,
        tag_id=int(tag.tag_id),
    )


def estimate_pose_from_tag_reference(tag, tag_ref: TagReference) -> Pose2D:
    T_ct = transform_from_rt(np.array(tag.pose_R, dtype=float), np.array(tag.pose_t, dtype=float))
    T_wt = transform_from_rt(
        rotz(tag_ref.world_yaw),
        np.array([tag_ref.world_x, tag_ref.world_y, 0.0], dtype=float),
    )
    T_wc = T_wt @ invert_transform(T_ct)
    T_wr = T_wc @ invert_transform(T_ROBOT_FROM_CAMERA)
    return Pose2D(
        x=float(T_wr[0, 3]),
        y=float(T_wr[1, 3]),
        yaw=wrap_to_pi(yaw_from_rotation(T_wr[:3, :3])),
    )


def capture_intermediate_reference_if_needed(world_map: WorldMap, goal: Landmark, tag, pose: Pose2D) -> None:
    if world_map.intermediate is not None or world_map.dropoff_tag_ref is not None:
        return
    if goal.kind not in ("small_goal", "large_goal"):
        return
    world_map.intermediate = Landmark(kind="intermediate", x=pose.x, y=pose.y)
    world_map.dropoff_tag_ref = compute_tag_reference(tag, goal.kind, pose)
    print(
        "[Map] intermediate waypoint recorded at "
        f"({pose.x:.2f}, {pose.y:.2f}) from {goal.kind} tag {int(tag.tag_id)}"
    )


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

    turn_to_yaw(ep_chassis, pose, tag_ref.reference_pose.yaw)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        tag = find_best_tag(tags, {tag_ref.tag_id})
        if tag is None:
            move_robot(ep_chassis, pose, z_deg=8.0)
            continue

        center_delta_px = float(tag.center[0]) - tag_ref.reference_center_x_px
        dist_delta_m = AprilTagDetector.tag_distance_m(tag) - tag_ref.reference_distance_m
        if abs(center_delta_px) > REFERENCE_CENTER_TOL_PX:
            move_robot(ep_chassis, pose, z_deg=max(-8.0, min(8.0, -0.10 * center_delta_px)))
            continue
        if abs(dist_delta_m) > REFERENCE_DIST_TOL_M:
            move_robot(ep_chassis, pose, x_m=max(-0.08, min(0.08, dist_delta_m)))
            continue

        refined_pose = estimate_pose_from_tag_reference(tag, tag_ref)
        pose.x = refined_pose.x
        pose.y = refined_pose.y
        pose.yaw = refined_pose.yaw
        print(f"[Localize] pose corrected from drop-off tag -> ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):.1f} deg)")
        return True

    print("[Localize] drop-off relocalization timed out.")
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
            ep_chassis,
            pose,
            world_map,
            world_map.intermediate.x,
            world_map.intermediate.y,
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
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        tag = find_best_tag(tags, RECHARGE_TAG_IDS)
        if tag is None:
            move_robot(ep_chassis, pose, z_deg=10.0)
            continue

        debug_log_tag_mapping(tag, pose, "recharge-tag-refine")
        recharge_landmark = landmark_from_tag_detection(tag, "recharge", pose)
        world_map.recharge = recharge_landmark
        print(
            "[Recharge] refined recharge landmark from tag "
            f"{recharge_landmark.tag_id} -> ({recharge_landmark.x:.2f}, {recharge_landmark.y:.2f})"
        )
        return True

    print("[Recharge] recharge tag not found during refinement window; keeping coarse recharge position.")
    return False


def wait_for_goal_tag(ep_camera, yolo_model: YOLO, tag_detector: AprilTagDetector, valid_ids: Iterable[int], timeout_s: float = 5.0):
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


def map_goal_from_view(
    ep_camera,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    valid_ids: Iterable[int],
    label: str,
) -> Landmark:
    # Goals are mapped from the current robot pose plus tag distance/bearing.
    frame, tag = wait_for_goal_tag(ep_camera, yolo_model, tag_detector, valid_ids, timeout_s=6.0)
    if tag is None:
        raise RuntimeError(f"Could not detect {label} goal tag.")
    tag_id = int(tag.tag_id)
    goal_kind = goal_kind_from_tag(tag_id)
    if goal_kind is None:
        raise RuntimeError(f"Detected tag {tag_id}, but it is not configured as a goal tag.")
    debug_log_tag_mapping(tag, pose, label)
    landmark = landmark_from_tag_detection(tag, goal_kind, pose)
    landmark = world_map.set_goal(goal_kind, landmark.x, landmark.y, tag_id)
    print(f"[Map] {label}: {goal_kind} at ({landmark.x:.2f}, {landmark.y:.2f}) tag={tag_id}")
    return landmark


def map_recharge_from_box(
    ep_camera,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    # In the described setup the recharge station is first observed as the left
    # face of a box, so this step uses YOLO + calibrated box distance instead
    # of requiring the recharge tag to be visible immediately.
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
        print(f"[Map] recharge box at ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
        return world_map.recharge
    raise RuntimeError("Could not map the recharge box from the left-face view.")


def scan_left_and_map_world(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    initial_goal_kind: str,
) -> Landmark:
    # After moving forward 2 ft, the robot translates left and opportunistically
    # maps known obstacle tags plus the opposite goal zone.
    opposite_goal_kind = "large_goal" if initial_goal_kind == "small_goal" else "small_goal"
    opposite_ids = LARGE_GOAL_TAG_IDS if opposite_goal_kind == "large_goal" else SMALL_GOAL_TAG_IDS

    total_left_m = 0.0
    while total_left_m < WORKSPACE_W_M:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is not None:
            tags, detections = detect_tags_and_objects(frame, yolo_model, tag_detector)

            if len(world_map.obstacles) < 2:
                mapped_from_tags = False
                for tag in tags:
                    tag_id = int(tag.tag_id)
                    if not is_obstacle_tag(tag_id):
                        continue
                    debug_log_tag_mapping(tag, pose, "obstacle-tag")
                    obstacle_landmark = landmark_from_tag_detection(tag, "obstacle", pose)
                    obstacle = world_map.add_or_update_obstacle(
                        obstacle_landmark.x,
                        obstacle_landmark.y,
                        tag_id=tag_id,
                    )
                    print(f"[Map] obstacle tag {tag_id} at ({obstacle.x:.2f}, {obstacle.y:.2f})")
                    mapped_from_tags = True
                    if len(world_map.obstacles) >= 2:
                        break

                if not mapped_from_tags:
                    # Fallback: if no obstacle tag is visible in the current
                    # frame, keep the old centered-box mapping behavior.
                    boxes = [det for det in detections if det.cls == CLASS_BOX]
                    centered_boxes = [det for det in boxes if abs(center_error_px(det.cx)) <= CENTER_TOL_PX]
                    for box in centered_boxes:
                        debug_log_box_mapping(box, pose, "obstacle-box-fallback")
                        world_pos = detection_world_position(box, pose)
                        if world_pos is None:
                            continue
                        if world_map.recharge is not None:
                            if math.hypot(world_pos[0] - world_map.recharge.x, world_pos[1] - world_map.recharge.y) < 0.40:
                                continue
                        obstacle = world_map.add_or_update_obstacle(world_pos[0], world_pos[1])
                        print(f"[Map] obstacle fallback at ({obstacle.x:.2f}, {obstacle.y:.2f})")
                        if len(world_map.obstacles) >= 2:
                            break

            tag = find_best_tag(tags, opposite_ids)
            if tag is not None and abs(center_error_px(float(tag.center[0]))) <= CENTER_TOL_PX:
                debug_log_tag_mapping(tag, pose, "opposite-goal")
                landmark = landmark_from_tag_detection(tag, opposite_goal_kind, pose)
                landmark = world_map.set_goal(opposite_goal_kind, landmark.x, landmark.y, int(tag.tag_id))
                capture_intermediate_reference_if_needed(world_map, landmark, tag, pose)
                print(f"[Map] opposite goal: {opposite_goal_kind} at ({landmark.x:.2f}, {landmark.y:.2f})")
                return landmark

        move_robot(ep_chassis, pose, y_m=LEFT_SCAN_STEP_M)
        total_left_m += LEFT_SCAN_STEP_M

    raise RuntimeError("Could not map the opposite goal while translating left.")


def map_loading_dock(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    # The loading dock has no tag in this workflow, so we treat the visible
    # tower clump as a set of detections whose centroid becomes the dock point.
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
                    dock_x = float(np.mean([pt[0] for pt in points]))
                    dock_y = float(np.mean([pt[1] for pt in points]))
                    world_map.dock = Landmark(kind="dock", x=dock_x, y=dock_y)
                    print(f"[Map] dock at ({dock_x:.2f}, {dock_y:.2f}) from {len(points)} towers")
                    return world_map.dock
        move_robot(ep_chassis, pose, z_deg=DOCK_SEARCH_STEP_DEG)

    raise RuntimeError("Could not locate the loading dock tower clump.")


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
    # `drive_speed(...)` is intentionally limited to this close-range visual
    # servo loop where we need fast incremental corrections.
    valid_set = set(valid_ids)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue
        tags, _ = detect_tags_and_objects(frame, yolo_model, tag_detector)
        matches = [tag for tag in tags if int(tag.tag_id) in valid_set]
        if not matches:
            # If the tag is not visible, rotate slowly in place until it appears.
            wz = 10.0
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
        wz = max(-30.0, min(30.0, -0.08 * err_px))
        dt = 0.15
        ep_chassis.drive_speed(x=vx, y=0.0, z=wz, timeout=dt)
        time.sleep(dt)
        integrate_drive_speed(pose, vx, 0.0, wz, dt)

    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    return False


def approach_brick_with_move(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    pose: Pose2D,
    target_class: int,
    timeout_s: float = 20.0,
) -> bool:
    # Brick pickup stays in the higher-level `move(...)` regime. The robot uses
    # bbox center and bbox height to nudge itself into a repeatable pickup pose.
    deadline = time.time() + timeout_s
    stable = 0
    desired_height_px = 160.0

    while time.time() < deadline:
        frame = read_frame(ep_camera, timeout=0.5)
        if frame is None:
            continue

        detections = get_detections(yolo_model, frame, conf_thresh=0.40, target_class=target_class)
        if not detections:
            # Small search turns are enough here because the dock should already
            # be roughly in front of the robot once navigation completes.
            move_robot(ep_chassis, pose, z_deg=10.0)
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
        lateral_step_m = max(-0.05, min(0.05, -0.0009 * err_x))
        if abs(forward_step_m) < 0.01 and abs(lateral_step_m) < 0.01:
            forward_step_m = 0.02 if err_h > 0 else -0.02
        move_robot(ep_chassis, pose, x_m=forward_step_m, y_m=lateral_step_m)

    return False


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
    # Navigate to the mapped goal first, then refine with a tag servo so the
    # final placement lines up with the actual drop-off face.
    goal_ids = SMALL_GOAL_TAG_IDS if goal.kind == "small_goal" else LARGE_GOAL_TAG_IDS
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(ep_chassis, pose, world_map, goal.x, goal.y, stop_dist_m=LANDMARK_STOP_DIST_M)
    success = servo_to_visible_tag(
        ep_camera,
        ep_chassis,
        yolo_model,
        tag_detector,
        pose,
        goal_ids,
        target_dist_m=GOAL_SERVO_DIST_M,
    )
    if not success:
        print("[Goal] Tag servo timed out; placing based on mapped position.")
    place_down_tower(ep_robot=ep_robot)
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)


def recharge_robot(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
    battery: BatteryManager,
) -> None:
    # Recharge uses the same pattern as delivery: coarse move to the mapped
    # landmark, then fine servo to the visible recharge tag(s).
    if world_map.recharge is None:
        raise RuntimeError("Recharge requested before recharge landmark was mapped.")

    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(
        ep_chassis,
        pose,
        world_map,
        world_map.recharge.x,
        world_map.recharge.y,
        stop_dist_m=LANDMARK_STOP_DIST_M,
    )
    try_refine_recharge_from_tag(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    navigate_to_point_with_map(
        ep_chassis,
        pose,
        world_map,
        world_map.recharge.x,
        world_map.recharge.y,
        stop_dist_m=max(RECHARGE_SERVO_DIST_M + 0.10, 0.30),
    )
    success = servo_to_visible_tag(
        ep_camera,
        ep_chassis,
        yolo_model,
        tag_detector,
        pose,
        RECHARGE_TAG_IDS,
        target_dist_m=RECHARGE_SERVO_DIST_M,
    )
    if not success:
        print("[Recharge] Tag servo timed out; holding at mapped recharge location.")
    print("[Recharge] Holding to simulate recharge...")
    time.sleep(5.0)
    battery.recharge()
    print(f"[Recharge] Battery now {battery.level:.0f}%")
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)


def execute_mapping_sequence(
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    pose: Pose2D,
    world_map: WorldMap,
) -> Landmark:
    # This function encodes the exact deterministic startup path described by
    # the user rather than performing a generic exploratory sweep.
    initial_goal = map_goal_from_view(
        ep_camera,
        yolo_model,
        tag_detector,
        pose,
        world_map,
        SMALL_GOAL_TAG_IDS | LARGE_GOAL_TAG_IDS,
        "initial front goal",
    )

    move_robot(ep_chassis, pose, z_deg=90.0)
    map_recharge_from_box(ep_camera, yolo_model, tag_detector, pose, world_map)

    # Return to the original heading, move 2 ft forward, then begin the leftward
    # scan that maps the two obstacles and the opposite goal.
    move_robot(ep_chassis, pose, z_deg=-90.0)
    move_robot(ep_chassis, pose, x_m=INITIAL_FORWARD_STEP_M)

    scan_left_and_map_world(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map, initial_goal.kind)
    print(f"[Map] mapped blocks so far: {world_map.mapped_block_count()}")

    move_robot(ep_chassis, pose, z_deg=180.0)
    map_loading_dock(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)

    # Assumption from the updated setup description:
    # the delivery destination is the mapped goal zone on the right side
    # of the workspace (largest x among the two goal landmarks).
    target_goal = world_map.right_side_goal()
    if target_goal is None:
        raise RuntimeError("No drop-off goal was mapped.")
    if world_map.intermediate is None:
        world_map.intermediate = Landmark(kind="intermediate", x=pose.x, y=pose.y)
        print(f"[Map] fallback intermediate waypoint at ({pose.x:.2f}, {pose.y:.2f})")
    print(f"[Mission] Using right-side goal `{target_goal.kind}` at ({target_goal.x:.2f}, {target_goal.y:.2f})")
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
    # Once mapping is complete, the mission collapses into a simple repeated
    # route: dock -> target goal -> dock, with recharge inserted as needed.
    if world_map.dock is None:
        raise RuntimeError("Delivery loop started before mapping the loading dock.")

    target_class = brick_class_for_goal(target_goal)
    print(f"[Mission] Dock -> {target_goal.kind} loop for class {target_class}")

    deliveries = 0
    go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
    while deliveries < max_deliveries:
        # Recharge before pickup if this battery level cannot support one more
        # brick of the required class.
        if not battery.can_pick(target_class):
            recharge_robot(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map, battery)

        go_to_intermediate_waypoint(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
        navigate_to_point_with_map(
            ep_chassis,
            pose,
            world_map,
            world_map.dock.x,
            world_map.dock.y,
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


def visualize_map(world_map: WorldMap, robot_pose: Optional[Pose2D] = None) -> None:
    """Render the mapped landmarks as a simple bird's-eye debugging plot."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Map] matplotlib not available. Skipping visualisation.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0.0, WORKSPACE_W_M)
    ax.set_ylim(0.0, WORKSPACE_H_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Project 3 Updated Map")

    for value in np.arange(0.0, WORKSPACE_W_M + 0.01, 0.10):
        ax.axvline(value, color="lightgray", linewidth=0.3)
    for value in np.arange(0.0, WORKSPACE_H_M + 0.01, 0.10):
        ax.axhline(value, color="lightgray", linewidth=0.3)

    ax.plot([0, WORKSPACE_W_M, WORKSPACE_W_M, 0, 0], [0, 0, WORKSPACE_H_M, WORKSPACE_H_M, 0], "k-", linewidth=2)

    for obstacle in world_map.obstacles:
        ax.add_patch(plt.Circle((obstacle.x, obstacle.y), 0.12, color="red", alpha=0.6))

    if world_map.recharge:
        ax.add_patch(
            patches.Rectangle(
                (world_map.recharge.x - 0.10, world_map.recharge.y - 0.10),
                0.20,
                0.20,
                facecolor="black",
                edgecolor="black",
                alpha=0.8,
            )
        )

    if world_map.small_goal:
        ax.plot(world_map.small_goal.x, world_map.small_goal.y, "b^", markersize=14)
    if world_map.large_goal:
        ax.plot(world_map.large_goal.x, world_map.large_goal.y, "g^", markersize=14)
    if world_map.dock:
        ax.add_patch(
            patches.Rectangle(
                (world_map.dock.x - 0.15, world_map.dock.y - 0.15),
                0.30,
                0.30,
                facecolor="yellow",
                edgecolor="goldenrod",
                alpha=0.8,
            )
        )
    if world_map.intermediate:
        ax.plot(world_map.intermediate.x, world_map.intermediate.y, "co", markersize=10)

    if robot_pose is not None:
        ax.plot(robot_pose.x, robot_pose.y, "ms", markersize=10)
        dx = 0.15 * math.cos(robot_pose.yaw)
        dy = 0.15 * math.sin(robot_pose.yaw)
        ax.annotate("", xy=(robot_pose.x + dx, robot_pose.y + dy), xytext=(robot_pose.x, robot_pose.y), arrowprops=dict(arrowstyle="->", color="magenta", lw=2))

    plt.tight_layout()
    plt.savefig("arena_map.png", dpi=150)
    print("[Map] Saved to arena_map.png")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3 deterministic mapping and delivery")
    parser.add_argument("--model-path", default=str(MODEL_PATH or DEFAULT_MODEL_PATH))
    parser.add_argument("--robot-ip", default=ROBOT_IP or DEFAULT_ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN or DEFAULT_ROBOT_SN)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--resolution", default="720p", choices=["360p", "720p"])
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--show-map", action="store_true")
    parser.add_argument("--max-deliveries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    # Main boot sequence:
    # 1. Connect hardware.
    # 2. Move arm to default pose.
    # 3. Run the deterministic mapping path.
    # 4. Optionally start repeated deliveries.
    args = parse_args()

    print("=== Project 3 Updated Workflow ===")
    print(f"[Setup] workspace = {WORKSPACE_W_M:.3f}m x {WORKSPACE_H_M:.3f}m")
    print(f"[Setup] start pose = ({START_X_M:.3f}, {START_Y_M:.3f}, {math.degrees(START_YAW_RAD):.1f} deg)")

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
    ep_camera.start_video_stream(display=True, resolution=resolution)

    try:
        ep_robot.robotic_arm.moveto(x=DEFAULT_ARM_X, y=DEFAULT_ARM_Y).wait_for_completed()
        ep_robot.gripper.open()
        time.sleep(1.0)
        ep_robot.gripper.pause()

        target_goal = execute_mapping_sequence(ep_camera, ep_chassis, yolo_model, tag_detector, pose, world_map)
        print(world_map.summary())

        if not args.map_only:
            run_delivery_loop(
                ep_robot,
                ep_camera,
                ep_chassis,
                yolo_model,
                tag_detector,
                pose,
                world_map,
                battery,
                target_goal,
                args.max_deliveries,
            )

        if args.show_map or args.map_only:
            visualize_map(world_map, pose)

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
