#!/usr/bin/env python3
"""
Project 3: Energy-Aware Logistics Challenge
CMSC477 - Robotics Perception and Planning

High-level sequence:
  1. Startup sweep: spin 360° to map ArUco tags (recharge, goals) and
     YOLO obstacles/cones. Build a live WorldMap.
  2. Locate loading dock via purple-tape color segmentation or YOLO brick detections.
  3. Battery-aware delivery loop:
       a. Check battery budget for next brick (factoring in pick+deliver+return).
       b. If budget too low → recharge first.
       c. Navigate to loading dock → approach brick → pick up → navigate to
          correct goal → place down → navigate back to loading dock.
  4. Recharging protocol: approach recharge tag head-on from ≥30 cm, stop
     within 5 cm for 5 seconds.
  5. Map visualisation via matplotlib at the end.

Coordinate system: world frame with origin at arena corner (Side-1 bottom-left).
  x → right (across arena width), y → into arena (Side-1 to Side-2).
  Arena is 3.0 × 3.0 m (≈10 × 10 ft).

ArUco tag IDs:
  Recharge station : 8, 10
  Small brick goal : 27, 30
  Large brick goal : 34, 38

YOLO class indices:
  0 = cone, 1 = box (obstacle), 2 = small_brick, 3 = large_brick
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pupil_apriltags
from ultralytics import YOLO

import robomaster
from config import (
    ALL_LANDMARK_TAG_IDS,
    ARENA_H_M,
    ARENA_W_M,
    BATTERY_LARGE_BRICK_COST,
    BATTERY_RECHARGE_LEVEL,
    BATTERY_RESERVE_PCT,
    BATTERY_SMALL_BRICK_COST,
    BATTERY_START_PCT,
    BRICK_SERVO_CENTER_TOL_PX,
    BRICK_SERVO_K_FWD,
    BRICK_SERVO_K_LAT,
    BRICK_SERVO_MAX_V,
    BRICK_SERVO_STABLE_THRESH,
    BRICK_SERVO_STEP_S,
    BRICK_SERVO_TOP_TOL_PX,
    BRICK_SERVO_TOP_Y_RATIO,
    CLASS_BOX,
    CLASS_CONE,
    CLASS_LARGE_BRICK,
    CLASS_SMALL_BRICK,
    K_CAM,
    LARGE_GOAL_TAG_IDS,
    MODEL_PATH,
    MOVE_SPEED_MPS,
    OBS_APPROACH_DIST_M,
    OBS_CLEAR_DIST_M,
    OBS_FWD_SPEED_MPS,
    OBS_SLIDE_SPEED_MPS,
    PURPLE_HSV_HI,
    PURPLE_HSV_LO,
    PURPLE_MIN_AREA_PX,
    RECHARGE_APPROACH_DIST_M,
    RECHARGE_HOLD_S,
    RECHARGE_STOP_DIST_M,
    RECHARGE_TAG_IDS,
    ROBOT_IP,
    ROBOT_SN,
    SIDE1_Y_LIMIT,
    SMALL_GOAL_TAG_IDS,
    SWEEP_SETTLE_S,
    SWEEP_STEP_DEG,
    TAG_DIST_TOL_M,
    TAG_FAMILY,
    TAG_SERVO_CENTER_TOL_PX,
    TAG_SERVO_DIST_TOL_M,
    TAG_SERVO_K_FWD,
    TAG_SERVO_K_YAW,
    TAG_SERVO_MAX_V,
    TAG_SERVO_MAX_YAW_DPS,
    TAG_SERVO_STEP_S,
    TAG_SIZE_M,
    TURN_SPEED_DPS,
)
from robomaster import camera as rm_camera
from robomaster import robot

from tower_utils import (
    clamp,
    get_detections,
    move_arm_to_default,
    move_arm_to_top,
    pick_up_tower,
    place_down_tower,
    select_detection,
    Detection,
)

# ─────────────────────────────────────────────────────────────────────────────
# Robot / model connection constants
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Arena geometry
# ─────────────────────────────────────────────────────────────────────────────


# Approximate y-boundary that separates Side-1 from the obstacle field.
# Side-1 occupies y ∈ [0, SIDE1_Y_LIMIT]. Tune after measuring.

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics (from project 1)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ArUco tag ID groups
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# YOLO class IDs
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Battery constants
# ─────────────────────────────────────────────────────────────────────────────


# Safety margin: always keep at least this much battery in reserve so we
# can still drive to the recharge station after an unexpected delay.

# ─────────────────────────────────────────────────────────────────────────────
# Navigation / servo constants
# ─────────────────────────────────────────────────────────────────────────────


# Obstacle avoidance

# Recharge approach

# AprilTag servo

# YOLO brick servo (mirrors project 2 go_to_tower_recorded style)

# Purple tape HSV bounds for loading dock segmentation

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose2D:
    """Robot pose in world frame."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0   # radians, CCW positive


@dataclass
class Landmark:
    """A detected landmark in the world map."""
    kind: str            # "recharge" | "small_goal" | "large_goal" | "obstacle" | "cone" | "dock"
    x: float             # world x (m)
    y: float             # world y (m)
    tag_id: Optional[int] = None


@dataclass
class WorldMap:
    """Incrementally-built map of the arena."""
    recharge: Optional[Landmark]       = None
    small_goal: Optional[Landmark]     = None
    large_goal: Optional[Landmark]     = None
    dock: Optional[Landmark]           = None
    obstacles: List[Landmark]          = field(default_factory=list)

    def is_fully_mapped(self) -> bool:
        return all([
            self.recharge is not None,
            self.small_goal is not None,
            self.large_goal is not None,
            self.dock is not None,
        ])

    def summary(self) -> str:
        lines = ["=== WorldMap ==="]
        for attr in ["recharge", "small_goal", "large_goal", "dock"]:
            lm = getattr(self, attr)
            if lm:
                lines.append(f"  {attr}: ({lm.x:.2f}, {lm.y:.2f})")
            else:
                lines.append(f"  {attr}: NOT FOUND")
        lines.append(f"  obstacles: {len(self.obstacles)}")
        return "\n".join(lines)


@dataclass
class DriveAction:
    vx: float
    vy: float
    vz: float
    dt: float


# ─────────────────────────────────────────────────────────────────────────────
# Battery manager
# ─────────────────────────────────────────────────────────────────────────────

class BatteryManager:
    def __init__(self, start_pct: float = BATTERY_START_PCT):
        self.level = start_pct
        print(f"[Battery] Initialized at {self.level:.1f}%")

    def consume(self, brick_class: int) -> None:
        cost = BATTERY_LARGE_BRICK_COST if brick_class == CLASS_LARGE_BRICK else BATTERY_SMALL_BRICK_COST
        self.level = max(0.0, self.level - cost)
        print(f"[Battery] Consumed {cost}% for class {brick_class}. Level={self.level:.1f}%")

    def recharge(self) -> None:
        self.level = BATTERY_RECHARGE_LEVEL
        print("[Battery] Recharged to 100%.")

    def can_afford(self, brick_class: int) -> bool:
        cost = BATTERY_LARGE_BRICK_COST if brick_class == CLASS_LARGE_BRICK else BATTERY_SMALL_BRICK_COST
        remaining = self.level - cost
        return remaining >= BATTERY_RESERVE_PCT

    def needs_recharge_before(self, brick_class: int) -> bool:
        return not self.can_afford(brick_class)

    @property
    def depleted(self) -> bool:
        return self.level <= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag detector (mirrors project 1)
# ─────────────────────────────────────────────────────────────────────────────

class AprilTagDetector:
    def __init__(
        self,
        K: np.ndarray = K_CAM,
        family: str = TAG_FAMILY,
        marker_size_m: float = TAG_SIZE_M,
        threads: int = 2,
    ):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(
            families=family,
            nthreads=threads,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def find_tags(self, gray: np.ndarray):
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )

    def tag_distance_m(self, detection) -> float:
        """Euclidean distance from camera to the tag centre (metres)."""
        t = np.array(detection.pose_t, dtype=float).reshape(3)
        return float(np.linalg.norm(t))

    def tag_center_px(self, detection) -> Tuple[float, float]:
        return float(detection.center[0]), float(detection.center[1])


# ─────────────────────────────────────────────────────────────────────────────
# Action stack (from project 2)
# ─────────────────────────────────────────────────────────────────────────────

class ActionStack:
    def __init__(self):
        self._stack: Deque[DriveAction] = deque()

    def push(self, a: DriveAction) -> None:
        self._stack.append(a)

    def clear(self) -> None:
        self._stack.clear()

    def snapshot(self) -> List[DriveAction]:
        return list(self._stack)

    def unwind(self, ep_chassis, pause_s: float = 0.05) -> None:
        while self._stack:
            a = self._stack.pop()
            ep_chassis.drive_speed(x=-a.vx, y=-a.vy, z=-a.vz, timeout=a.dt)
            time.sleep(a.dt + pause_s)
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
        time.sleep(0.1 + pause_s)


def replay_route(ep_chassis, route: List[DriveAction], pause_s: float = 0.05) -> None:
    for a in route:
        ep_chassis.drive_speed(x=a.vx, y=a.vy, z=a.vz, timeout=a.dt)
        time.sleep(a.dt + pause_s)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    time.sleep(0.1 + pause_s)


def reverse_route(ep_chassis, route: List[DriveAction], pause_s: float = 0.05) -> None:
    for a in reversed(route):
        ep_chassis.drive_speed(x=-a.vx, y=-a.vy, z=-a.vz, timeout=a.dt)
        time.sleep(a.dt + pause_s)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    time.sleep(0.1 + pause_s)


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(a: float) -> float:
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Localizer: compute robot world pose from a single ArUco detection,
# given that the tag's world pose was recorded during the sweep.
# ─────────────────────────────────────────────────────────────────────────────

class TagLocalizer:
    """
    Estimates robot pose from an ArUco detection, using a tag_world_map that
    is populated incrementally during the startup sweep.

    Because we don't know tag world poses at the start, we bootstrap by:
      1. During the sweep the robot is at a known (approximately dead-reckoned)
         pose. When we first see a tag we record its world position based on
         that dead-reckoned pose and the camera measurement.
      2. Once the tag's world pose is recorded, subsequent detections of that
         tag yield refined robot poses (inverting the camera-to-tag transform).
    """

    # Camera-to-robot-body transform: identity (camera is at robot centre).
    T_RC = np.eye(4, dtype=float)

    def __init__(self):
        # tag_id → (world_x, world_y, world_yaw of the tag face normal)
        self.tag_world: Dict[int, Tuple[float, float, float]] = {}
        self._last: Optional[Pose2D] = None
        self._alpha = 0.4   # EMA smoothing

    def register_tag_from_robot_pose(
        self,
        detection,
        robot_pose: Pose2D,
    ) -> None:
        """
        Given a dead-reckoned robot_pose and a fresh tag detection,
        compute and store the tag's world pose.
        """
        tag_id = int(detection.tag_id)
        if tag_id in self.tag_world:
            return  # already registered; don't overwrite

        t_ct = np.array(detection.pose_t, dtype=float).reshape(3)
        R_ct = np.array(detection.pose_R, dtype=float).reshape(3, 3)

        # tag → camera → robot-body → world
        T_CT = T_from_Rt(R_ct, t_ct)          # tag in camera frame
        T_WR = T_from_Rt(                      # robot in world frame
            rotz(robot_pose.yaw),
            np.array([robot_pose.x, robot_pose.y, 0.0])
        )
        T_WT = T_WR @ self.T_RC @ T_CT         # tag in world frame

        wx = float(T_WT[0, 3])
        wy = float(T_WT[1, 3])
        wyaw = yaw_from_R(T_WT[:3, :3])

        self.tag_world[tag_id] = (wx, wy, wyaw)
        print(f"  [Localizer] Registered tag {tag_id} at world ({wx:.2f}, {wy:.2f}, yaw={math.degrees(wyaw):.1f}°)")

    def estimate_pose(self, detection) -> Optional[Pose2D]:
        """
        Given a tag detection, return an estimated robot world pose.
        Returns None if the tag has not been registered yet.
        """
        tag_id = int(detection.tag_id)
        if tag_id not in self.tag_world:
            return None

        wx, wy, wyaw = self.tag_world[tag_id]

        t_ct = np.array(detection.pose_t, dtype=float).reshape(3)
        R_ct = np.array(detection.pose_R, dtype=float).reshape(3, 3)
        T_CT = T_from_Rt(R_ct, t_ct)

        T_WT = T_from_Rt(rotz(wyaw), np.array([wx, wy, 0.0]))
        T_WC = T_WT @ inv_T(T_CT)
        T_WR = T_WC @ inv_T(self.T_RC)

        pose = Pose2D(
            x=float(T_WR[0, 3]),
            y=float(T_WR[1, 3]),
            yaw=yaw_from_R(T_WR[:3, :3]),
        )
        return self._smooth(pose)

    def _smooth(self, pose: Pose2D) -> Pose2D:
        if self._last is None:
            self._last = pose
            return pose
        a = self._alpha
        x   = (1 - a) * self._last.x   + a * pose.x
        y   = (1 - a) * self._last.y   + a * pose.y
        u0  = np.array([math.cos(self._last.yaw), math.sin(self._last.yaw)])
        u1  = np.array([math.cos(pose.yaw),       math.sin(pose.yaw)])
        u   = (1 - a) * u0 + a * u1
        yaw = math.atan2(u[1], u[0])
        self._last = Pose2D(x=x, y=y, yaw=yaw)
        return self._last


# ─────────────────────────────────────────────────────────────────────────────
# Dead-reckoning pose tracker (fallback when no tag is visible)
# ─────────────────────────────────────────────────────────────────────────────

class DeadReckoner:
    def __init__(self, initial_pose: Pose2D = Pose2D()):
        self.pose = initial_pose

    def update_from_drive(self, vx: float, vy: float, vz_dps: float, dt: float) -> None:
        """
        Integrate a chassis command into the pose estimate.
        vx/vy are in robot body frame. vz_dps is yaw rate in degrees/second.
        """
        dyaw = math.radians(vz_dps) * dt
        # Rotate body-frame displacement to world frame (use midpoint yaw)
        mid_yaw = self.pose.yaw + dyaw / 2.0
        dx_w = vx * math.cos(mid_yaw) - vy * math.sin(mid_yaw)
        dy_w = vx * math.sin(mid_yaw) + vy * math.cos(mid_yaw)
        self.pose.x   += dx_w * dt
        self.pose.y   += dy_w * dt
        self.pose.yaw  = wrap_to_pi(self.pose.yaw + dyaw)

    def fuse_tag_pose(self, tag_pose: Pose2D, weight: float = 0.7) -> None:
        """Hard-fuse a tag-derived pose into the dead-reckoner."""
        w = weight
        self.pose.x   = (1 - w) * self.pose.x   + w * tag_pose.x
        self.pose.y   = (1 - w) * self.pose.y   + w * tag_pose.y
        # Slerp for yaw
        u0  = np.array([math.cos(self.pose.yaw), math.sin(self.pose.yaw)])
        u1  = np.array([math.cos(tag_pose.yaw),  math.sin(tag_pose.yaw)])
        u   = (1 - w) * u0 + w * u1
        self.pose.yaw = math.atan2(u[1], u[0])


# ─────────────────────────────────────────────────────────────────────────────
# Low-level chassis helpers
# ─────────────────────────────────────────────────────────────────────────────

def chassis_stop(ep_chassis, hold_s: float = 0.1) -> None:
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=hold_s)
    time.sleep(hold_s)


def open_loop_turn(ep_chassis, degrees: float, dead_reckoner: DeadReckoner) -> None:
    """Turn in place by a given angle (+ = CCW)."""
    if abs(degrees) < 0.5:
        return
    vz = TURN_SPEED_DPS if degrees > 0 else -TURN_SPEED_DPS
    dt = abs(degrees) / TURN_SPEED_DPS
    ep_chassis.drive_speed(x=0.0, y=0.0, z=vz, timeout=dt + 0.2)
    time.sleep(dt)
    chassis_stop(ep_chassis)
    dead_reckoner.update_from_drive(0.0, 0.0, vz, dt)


def open_loop_drive(
    ep_chassis,
    dist_m: float,
    dead_reckoner: DeadReckoner,
    vx: float = MOVE_SPEED_MPS,
    vy: float = 0.0,
) -> None:
    """Drive forward (or laterally) a fixed distance in metres."""
    if abs(dist_m) < 0.005:
        return
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return
    actual_vx = vx * abs(dist_m) / (dist_m if dist_m != 0 else 1) * speed / speed
    actual_vy = vy
    # If dist_m is negative we want to back up:
    if dist_m < 0:
        actual_vx = -abs(vx)
    dt = abs(dist_m) / speed
    ep_chassis.drive_speed(x=actual_vx, y=actual_vy, z=0.0, timeout=dt + 0.2)
    time.sleep(dt)
    chassis_stop(ep_chassis)
    dead_reckoner.update_from_drive(actual_vx, actual_vy, 0.0, dt)


def turn_to_heading(
    ep_chassis,
    target_yaw_rad: float,
    dead_reckoner: DeadReckoner,
) -> None:
    """Turn to face a specific world yaw."""
    delta = wrap_to_pi(target_yaw_rad - dead_reckoner.pose.yaw)
    open_loop_turn(ep_chassis, math.degrees(delta), dead_reckoner)


# ─────────────────────────────────────────────────────────────────────────────
# Color segmentation: find loading dock (purple tape)
# ─────────────────────────────────────────────────────────────────────────────

def detect_purple_dock(frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Segment the purple loading-dock tape and return (cx, cy, area) in pixel
    coordinates if found, else None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_HSV_LO, PURPLE_HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < PURPLE_MIN_AREA_PX:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy, area


# ─────────────────────────────────────────────────────────────────────────────
# Startup 360° sweep: map all landmarks
# ─────────────────────────────────────────────────────────────────────────────

def startup_sweep(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    tag_localizer: TagLocalizer,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    show: bool = False,
) -> None:
    """
    Spin the robot 360° in SWEEP_STEP_DEG increments, at each step:
      - Run AprilTag detection to find/register recharge and goal tags.
      - Run YOLO to find obstacle boxes and cones.
      - Attempt purple-tape segmentation to find the loading dock.
      - Fuse any tag-based localisation into dead_reckoner.

    After the sweep the robot is back at its starting orientation.
    """
    print("[Sweep] Starting 360° mapping sweep ...")
    total_turned = 0.0

    while total_turned < 360.0:
        step = min(SWEEP_STEP_DEG, 360.0 - total_turned)
        open_loop_turn(ep_chassis, step, dead_reckoner)
        total_turned += step
        time.sleep(SWEEP_SETTLE_S)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # ── ArUco tag detection ────────────────────────────────────────────
        tags = tag_detector.find_tags(gray)
        for det in tags:
            tid = int(det.tag_id)
            if tid not in ALL_LANDMARK_TAG_IDS:
                continue

            # Register the tag's world location using the current dead-reckoned pose
            tag_localizer.register_tag_from_robot_pose(det, dead_reckoner.pose)

            # Fuse a pose correction if the tag is already registered
            refined = tag_localizer.estimate_pose(det)
            if refined is not None:
                dead_reckoner.fuse_tag_pose(refined, weight=0.6)

            # Record into world_map
            wx, wy, _ = tag_localizer.tag_world[tid]
            if tid in RECHARGE_TAG_IDS and world_map.recharge is None:
                world_map.recharge = Landmark("recharge", wx, wy, tag_id=tid)
                print(f"  [Map] Recharge station found at ({wx:.2f}, {wy:.2f}) tag={tid}")
            elif tid in SMALL_GOAL_TAG_IDS and world_map.small_goal is None:
                world_map.small_goal = Landmark("small_goal", wx, wy, tag_id=tid)
                print(f"  [Map] Small goal found at ({wx:.2f}, {wy:.2f}) tag={tid}")
            elif tid in LARGE_GOAL_TAG_IDS and world_map.large_goal is None:
                world_map.large_goal = Landmark("large_goal", wx, wy, tag_id=tid)
                print(f"  [Map] Large goal found at ({wx:.2f}, {wy:.2f}) tag={tid}")

        # ── YOLO: obstacles and cones ──────────────────────────────────────
        dets = get_detections(yolo_model, frame, conf_thresh=0.45)
        for d in dets:
            if d.cls in (CLASS_BOX, CLASS_CONE):
                # Estimate obstacle world position from bounding-box pixel position.
                # We use a rough monocular depth estimate: actual_height_m / pixel_height * f
                # For a generic box height assumption:
                assumed_h_m = 0.30  # approximate fabric-box height
                dist_m = (assumed_h_m * K_CAM[1, 1]) / max(d.h, 1.0)
                # Bearing angle relative to camera centre
                angle_cam = math.atan2(d.cx - K_CAM[0, 2], K_CAM[0, 0])
                bearing = wrap_to_pi(dead_reckoner.pose.yaw + angle_cam)
                ox = dead_reckoner.pose.x + dist_m * math.cos(bearing)
                oy = dead_reckoner.pose.y + dist_m * math.sin(bearing)
                kind = "obstacle" if d.cls == CLASS_BOX else "cone"
                # Deduplicate: skip if a same-kind landmark is within 0.4 m
                duplicate = any(
                    math.hypot(lm.x - ox, lm.y - oy) < 0.40
                    for lm in world_map.obstacles
                )
                if not duplicate:
                    world_map.obstacles.append(Landmark(kind, ox, oy))
                    print(f"  [Map] {kind} found at ({ox:.2f}, {oy:.2f})")

        # ── Purple tape: loading dock ──────────────────────────────────────
        if world_map.dock is None:
            purple = detect_purple_dock(frame)
            if purple is not None:
                cx_px, cy_px, _ = purple
                # Rough distance based on apparent vertical position
                dist_m = 0.6  # heuristic for now; refine with monocular geometry
                angle_cam = math.atan2(cx_px - K_CAM[0, 2], K_CAM[0, 0])
                bearing = wrap_to_pi(dead_reckoner.pose.yaw + angle_cam)
                dx = dead_reckoner.pose.x + dist_m * math.cos(bearing)
                dy = dead_reckoner.pose.y + dist_m * math.sin(bearing)
                world_map.dock = Landmark("dock", dx, dy)
                print(f"  [Map] Loading dock found at ({dx:.2f}, {dy:.2f})")

        if show:
            dbg = frame.copy()
            for det in tags:
                pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(dbg, [pts], True, (0, 0, 255), 2)
                c = det.center.astype(int)
                cv2.putText(dbg, str(det.tag_id), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("sweep", dbg)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[Sweep] Complete.")
    print(world_map.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle-aware crossing: navigate from Side-1 through obstacle field to Side-2
# (and back). Strategy: face the obstacle field, scan left/right for clear
# corridor, drive forward stopping before obstacles then slide to side.
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_obstacle_dist(yolo_model: YOLO, frame: np.ndarray) -> float:
    """Return pixel-based distance estimate to nearest obstacle in frame."""
    dets = get_detections(yolo_model, frame, conf_thresh=0.40, target_class=CLASS_BOX)
    if not dets:
        return float("inf")
    best = max(dets, key=lambda d: d.h)   # tallest = closest
    assumed_h_m = 0.30
    dist_m = (assumed_h_m * K_CAM[1, 1]) / max(best.h, 1.0)
    return dist_m


def _corridor_is_clear(yolo_model: YOLO, frame: np.ndarray, min_dist_m: float = OBS_CLEAR_DIST_M) -> bool:
    return _nearest_obstacle_dist(yolo_model, frame) >= min_dist_m


def cross_obstacle_field(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    direction: str = "forward",   # "forward" (Side-1→2) or "backward" (Side-2→1)
    timeout_s: float = 60.0,
    show: bool = False,
) -> None:
    """
    Cross the obstacle field with a simple reactive strategy:
      1. Look for a clear corridor straight ahead.
      2. If blocked, slide left; if still blocked after a limit, slide right.
      3. Drive forward incrementally; repeat from step 1.
    direction="backward" simply drives in the negative-x direction (robot backs
    through or uses the same logic mirrored).
    """
    print(f"[CrossField] Starting obstacle-field crossing ({direction}) ...")
    t0 = time.time()
    fwd_sign = 1.0 if direction == "forward" else -1.0

    while time.time() - t0 < timeout_s:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        if _corridor_is_clear(yolo_model, frame):
            # Drive forward one step
            vx = fwd_sign * OBS_FWD_SPEED_MPS
            dt = 0.3
            ep_chassis.drive_speed(x=vx, y=0.0, z=0.0, timeout=dt)
            dead_reckoner.update_from_drive(vx, 0.0, 0.0, dt)
            time.sleep(dt)

            # Check if we have crossed (simplified: if y > arena_mid we are in Side-2)
            if direction == "forward" and dead_reckoner.pose.y > ARENA_H_M * 0.65:
                print("[CrossField] Reached Side-2.")
                chassis_stop(ep_chassis)
                return
            if direction == "backward" and dead_reckoner.pose.y < SIDE1_Y_LIMIT:
                print("[CrossField] Returned to Side-1.")
                chassis_stop(ep_chassis)
                return
        else:
            # Blocked — try sliding left, then right
            chassis_stop(ep_chassis)
            slid = _slide_around_obstacle(ep_chassis, ep_camera, yolo_model, dead_reckoner, fwd_sign, show)
            if not slid:
                print("[CrossField] WARNING: could not clear obstacle. Retrying ...")
                time.sleep(0.5)

        if show:
            cv2.imshow("cross_field", frame)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[CrossField] WARNING: timeout reached.")


def _slide_around_obstacle(
    ep_chassis,
    ep_camera,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    fwd_sign: float,
    show: bool,
) -> bool:
    """
    Attempt to slide left, then right, to find a clear corridor.
    Returns True if a clear path was found and the robot has moved to it.
    """
    for lateral_sign in [1.0, -1.0, -1.0]:   # try left, then right (2 steps right)
        # Slide one step
        vy = lateral_sign * OBS_SLIDE_SPEED_MPS
        dt = 0.5
        ep_chassis.drive_speed(x=0.0, y=vy, z=0.0, timeout=dt)
        dead_reckoner.update_from_drive(0.0, vy, 0.0, dt)
        time.sleep(dt)
        chassis_stop(ep_chassis)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        if _corridor_is_clear(yolo_model, frame):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag visual servo: approach a tag head-on to a target distance
# ─────────────────────────────────────────────────────────────────────────────

def servo_to_tag(
    ep_chassis,
    ep_camera,
    tag_detector: AprilTagDetector,
    target_tag_ids: set,
    target_dist_m: float,
    dead_reckoner: DeadReckoner,
    timeout_s: float = 20.0,
    show: bool = False,
) -> bool:
    """
    Visually servo toward a tag until within target_dist_m.
    Yaw correction centres the tag horizontally; forward motion closes distance.
    Returns True on success, False on timeout.
    """
    print(f"[TagServo] Homing to tag IDs {target_tag_ids}, target dist={target_dist_m:.2f}m ...")
    t0 = time.time()
    stable = 0

    while time.time() - t0 < timeout_s:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        tags = tag_detector.find_tags(gray)
        relevant = [d for d in tags if int(d.tag_id) in target_tag_ids]

        if not relevant:
            # Spin slowly to search
            ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=TAG_SERVO_STEP_S)
            dead_reckoner.update_from_drive(0.0, 0.0, 8.0, TAG_SERVO_STEP_S)
            time.sleep(TAG_SERVO_STEP_S)
            continue

        # Pick closest tag
        best = min(relevant, key=tag_detector.tag_distance_m)
        dist = tag_detector.tag_distance_m(best)
        cx_px, _ = tag_detector.tag_center_px(best)
        frame_cx = frame.shape[1] / 2.0
        err_px = cx_px - frame_cx

        # Yaw correction
        vz = clamp(-TAG_SERVO_K_YAW * err_px, -TAG_SERVO_MAX_YAW_DPS, TAG_SERVO_MAX_YAW_DPS)
        # Forward correction (only if centred)
        if abs(err_px) < TAG_SERVO_CENTER_TOL_PX:
            err_dist = dist - target_dist_m
            vx = clamp(-TAG_SERVO_K_FWD * err_dist, -TAG_SERVO_MAX_V, TAG_SERVO_MAX_V)
            vz = 0.0
        else:
            vx = 0.0

        ep_chassis.drive_speed(x=vx, y=0.0, z=vz, timeout=TAG_SERVO_STEP_S)
        dead_reckoner.update_from_drive(vx, 0.0, vz, TAG_SERVO_STEP_S)
        time.sleep(TAG_SERVO_STEP_S)

        # Arrival check
        if abs(err_px) < TAG_SERVO_CENTER_TOL_PX and abs(dist - target_dist_m) < TAG_DIST_TOL_M:
            stable += 1
        else:
            stable = 0

        if stable >= 3:
            chassis_stop(ep_chassis)
            print(f"  [TagServo] Arrived. dist={dist:.3f}m err_px={err_px:.1f}")
            return True

        if show:
            dbg = frame.copy()
            pts = best.corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(dbg, [pts], True, (0, 0, 255), 2)
            cv2.putText(dbg, f"dist={dist:.2f}m err={err_px:.0f}px", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("tag_servo", dbg)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print(f"  [TagServo] Timeout after {timeout_s}s.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Recharging protocol
# ─────────────────────────────────────────────────────────────────────────────

def execute_recharge(
    ep_robot,
    ep_chassis,
    ep_camera,
    tag_detector: AprilTagDetector,
    tag_localizer: TagLocalizer,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    battery: BatteryManager,
    show: bool = False,
) -> None:
    """
    Full recharging protocol:
      1. Navigate to recharge station (near its mapped location).
      2. Face the tag head-on from ≥30 cm away.
      3. Servo forward to within 5 cm.
      4. Hold stationary for 5 s.
      5. Update battery to 100%.
    """
    if world_map.recharge is None:
        print("[Recharge] ERROR: recharge station not mapped. Cannot recharge!")
        return

    print(f"[Recharge] Navigating to recharge station at ({world_map.recharge.x:.2f}, {world_map.recharge.y:.2f}) ...")

    # Step 1: drive roughly toward recharge station using dead reckoning
    _navigate_to_world_point(
        ep_chassis, dead_reckoner,
        world_map.recharge.x, world_map.recharge.y,
        stop_dist_m=RECHARGE_APPROACH_DIST_M + 0.10,
    )

    # Step 2 & 3: visual servo to within 5 cm, head-on
    tag_ids = RECHARGE_TAG_IDS
    if world_map.recharge.tag_id is not None:
        tag_ids = {world_map.recharge.tag_id}

    success = servo_to_tag(
        ep_chassis=ep_chassis,
        ep_camera=ep_camera,
        tag_detector=tag_detector,
        target_tag_ids=tag_ids,
        target_dist_m=RECHARGE_STOP_DIST_M,
        dead_reckoner=dead_reckoner,
        timeout_s=25.0,
        show=show,
    )

    if not success:
        print("[Recharge] WARNING: could not precisely reach recharge tag. Holding anyway.")

    # Step 4: hold stationary for 5 s
    print(f"[Recharge] Holding stationary for {RECHARGE_HOLD_S}s ...")
    chassis_stop(ep_chassis, hold_s=RECHARGE_HOLD_S)

    # Step 5: update battery
    battery.recharge()


# ─────────────────────────────────────────────────────────────────────────────
# Simple point-to-point navigation (dead-reckoning + tag correction)
# ─────────────────────────────────────────────────────────────────────────────

def _navigate_to_world_point(
    ep_chassis,
    dead_reckoner: DeadReckoner,
    wx: float,
    wy: float,
    stop_dist_m: float = 0.20,
    speed: float = MOVE_SPEED_MPS,
) -> None:
    """
    Open-loop navigate to (wx, wy) in world coordinates.
    Turns to face the target, then drives the Euclidean distance.
    """
    pose = dead_reckoner.pose
    dx = wx - pose.x
    dy = wy - pose.y
    dist = math.hypot(dx, dy)

    if dist <= stop_dist_m:
        return

    target_yaw = math.atan2(dy, dx)
    turn_to_heading(ep_chassis, target_yaw, dead_reckoner)

    drive_dist = max(0.0, dist - stop_dist_m)
    open_loop_drive(ep_chassis, drive_dist, dead_reckoner, vx=speed)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO brick approach (visual servo, mirrors project 2 go_to_tower_recorded)
# ─────────────────────────────────────────────────────────────────────────────

def approach_brick(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    brick_class: int,
    action_stack: ActionStack,
    timeout_s: float = 30.0,
    show: bool = False,
) -> bool:
    """
    Visually servo toward the nearest brick of brick_class using lateral
    strafe (no turning), recording all drive commands onto action_stack.
    Returns True on success.
    """
    print(f"[BrickApproach] Approaching brick class {brick_class} ...")
    action_stack.clear()
    move_arm_to_default(ep_robot)

    stable = 0
    center_stable = 0
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        dets = get_detections(yolo_model, frame, conf_thresh=0.40, target_class=brick_class)
        if not dets:
            # Spin slowly to search
            vz = 8.0
            ep_chassis.drive_speed(x=0.0, y=0.0, z=vz, timeout=BRICK_SERVO_STEP_S)
            action_stack.push(DriveAction(0.0, 0.0, vz, BRICK_SERVO_STEP_S))
            dead_reckoner.update_from_drive(0.0, 0.0, vz, BRICK_SERVO_STEP_S)
            time.sleep(BRICK_SERVO_STEP_S)
            continue

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]
        frame_cx = frame_w / 2.0

        selected = select_detection(dets, selection_mode="conf", frame_center_x=frame_cx)
        y_top = selected.cy - selected.h / 2.0
        err_x = selected.cx - frame_cx
        target_top_y = BRICK_SERVO_TOP_Y_RATIO * frame_h
        err_fwd = target_top_y - y_top

        centered = abs(err_x) < BRICK_SERVO_CENTER_TOL_PX
        center_stable = center_stable + 1 if centered else 0

        if center_stable < 2:
            vx, vy = 0.0, clamp(BRICK_SERVO_K_LAT * err_x, -BRICK_SERVO_MAX_V, BRICK_SERVO_MAX_V)
        else:
            vx = clamp(BRICK_SERVO_K_FWD * err_fwd, -BRICK_SERVO_MAX_V, BRICK_SERVO_MAX_V)
            vy = 0.0

        ep_chassis.drive_speed(x=vx, y=vy, z=0.0, timeout=BRICK_SERVO_STEP_S)
        action_stack.push(DriveAction(vx, vy, 0.0, BRICK_SERVO_STEP_S))
        dead_reckoner.update_from_drive(vx, vy, 0.0, BRICK_SERVO_STEP_S)
        time.sleep(BRICK_SERVO_STEP_S)

        if abs(err_x) < BRICK_SERVO_CENTER_TOL_PX and abs(err_fwd) < BRICK_SERVO_TOP_TOL_PX:
            stable += 1
        else:
            stable = 0

        if stable >= BRICK_SERVO_STABLE_THRESH:
            chassis_stop(ep_chassis)
            action_stack.push(DriveAction(0.0, 0.0, 0.0, 0.1))
            print(f"  [BrickApproach] Arrived at brick.")
            return True

        if show:
            dbg = frame.copy()
            x1 = int(selected.cx - selected.w / 2)
            y1 = int(selected.cy - selected.h / 2)
            cv2.rectangle(dbg, (x1, y1), (x1 + int(selected.w), y1 + int(selected.h)), (0, 255, 0), 2)
            cv2.putText(dbg, f"err_x={err_x:+.0f} fwd={err_fwd:+.0f} stable={stable}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("brick_approach", dbg)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[BrickApproach] Timeout.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Delivery: approach goal tag, place brick, return
# ─────────────────────────────────────────────────────────────────────────────

def deliver_brick(
    ep_robot,
    ep_camera,
    ep_chassis,
    tag_detector: AprilTagDetector,
    dead_reckoner: DeadReckoner,
    goal_landmark: Landmark,
    approach_stack: ActionStack,
    show: bool = False,
) -> None:
    """
    Drive to the goal AprilTag, place the brick within 25 cm, back away.
    approach_stack is the route recorded during approach_brick — we reverse
    it to return to the loading dock area after delivery.
    """
    goal_tag_ids = (
        SMALL_GOAL_TAG_IDS if goal_landmark.kind == "small_goal" else LARGE_GOAL_TAG_IDS
    )
    if goal_landmark.tag_id is not None:
        goal_tag_ids = {goal_landmark.tag_id}

    print(f"[Deliver] Navigating to {goal_landmark.kind} at ({goal_landmark.x:.2f}, {goal_landmark.y:.2f}) ...")

    # Navigate near the goal via dead-reckoning
    _navigate_to_world_point(
        ep_chassis, dead_reckoner,
        goal_landmark.x, goal_landmark.y,
        stop_dist_m=0.5,
    )

    # Servo precisely to within 25 cm of the tag
    servo_to_tag(
        ep_chassis=ep_chassis,
        ep_camera=ep_camera,
        tag_detector=tag_detector,
        target_tag_ids=goal_tag_ids,
        target_dist_m=0.22,
        dead_reckoner=dead_reckoner,
        timeout_s=20.0,
        show=show,
    )

    # Place brick
    print("[Deliver] Placing brick ...")
    place_down_tower(ep_robot=ep_robot)

    # Back away from goal so we don't touch it on the way out
    open_loop_drive(ep_chassis, -0.30, dead_reckoner)
    print("[Deliver] Brick placed. Backing away.")


# ─────────────────────────────────────────────────────────────────────────────
# Main delivery loop
# ─────────────────────────────────────────────────────────────────────────────

def delivery_loop(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    tag_localizer: TagLocalizer,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    battery: BatteryManager,
    max_deliveries: int = 5,
    show: bool = False,
) -> None:
    """
    Battery-aware brick delivery loop.

    For each delivery:
      1. Check if we need to recharge.
      2. Cross the obstacle field (Side-1 → Side-2).
      3. Find and approach a brick on Side-1 (loading dock).
         NOTE: Bricks are on Side-1 (loading dock). Goals are on Side-2.
         The actual sequence is:
           - Approach brick on Side-1.
           - Pick it up.
           - Cross to Side-2.
           - Deliver to goal.
           - Cross back to Side-1.
      4. Deduct battery.
      5. Repeat.
    """
    action_stack = ActionStack()
    deliveries = 0

    while deliveries < max_deliveries and not battery.depleted:
        print(f"\n══ Delivery {deliveries + 1}/{max_deliveries} ║ Battery={battery.level:.0f}% ══")

        # ── Step 1: Decide whether to recharge first ───────────────────────
        # We need enough battery for the cheapest possible brick (small = 30%).
        if battery.needs_recharge_before(CLASS_SMALL_BRICK):
            print("[Loop] Battery low — recharging before next delivery.")
            execute_recharge(
                ep_robot, ep_chassis, ep_camera,
                tag_detector, tag_localizer, dead_reckoner,
                world_map, battery, show=show,
            )

        # ── Step 2: Navigate to loading dock and scan for bricks ──────────
        if world_map.dock is not None:
            _navigate_to_world_point(
                ep_chassis, dead_reckoner,
                world_map.dock.x, world_map.dock.y,
                stop_dist_m=0.5,
            )
        else:
            # Dock not mapped yet — stay on Side-1 and search
            print("[Loop] Dock location unknown; searching Side-1 ...")
            _search_for_dock(ep_chassis, ep_camera, dead_reckoner, yolo_model, world_map, show)

        # ── Step 3: Identify which brick class to pick ────────────────────
        brick_class = _choose_brick(ep_camera, yolo_model)
        if brick_class is None:
            print("[Loop] No bricks visible. Skipping.")
            break

        # Check battery for this specific brick before committing
        if battery.needs_recharge_before(brick_class):
            print(f"[Loop] Not enough battery for class {brick_class}. Recharging ...")
            execute_recharge(
                ep_robot, ep_chassis, ep_camera,
                tag_detector, tag_localizer, dead_reckoner,
                world_map, battery, show=show,
            )

        # ── Step 4: Approach and pick up the brick ────────────────────────
        move_arm_to_top(ep_robot)
        success = approach_brick(
            ep_robot, ep_camera, ep_chassis, yolo_model,
            dead_reckoner, brick_class, action_stack,
            timeout_s=30.0, show=show,
        )
        if not success:
            print("[Loop] Could not approach brick. Skipping delivery.")
            continue

        dock_return_route = action_stack.snapshot()   # reversed to get back to dock

        pick_up_tower(ep_robot=ep_robot)

        # ── Step 5: Deduct battery ────────────────────────────────────────
        battery.consume(brick_class)
        if battery.depleted:
            print("[Loop] BATTERY DEPLETED during pickup! Stopping deliveries.")
            # Still place the brick if possible (already holding it)
            break

        # ── Step 6: Cross obstacle field to Side-2 ────────────────────────
        cross_obstacle_field(
            ep_robot, ep_camera, ep_chassis, yolo_model,
            dead_reckoner, direction="forward", show=show,
        )

        # ── Step 7: Deliver to the correct goal ──────────────────────────
        goal = (
            world_map.small_goal if brick_class == CLASS_SMALL_BRICK else world_map.large_goal
        )
        if goal is None:
            print(f"[Loop] WARNING: goal for brick class {brick_class} not mapped! Placing at current position.")
            place_down_tower(ep_robot=ep_robot)
        else:
            deliver_brick(
                ep_robot, ep_camera, ep_chassis,
                tag_detector, dead_reckoner, goal,
                action_stack, show=show,
            )

        deliveries += 1
        print(f"[Loop] Delivery {deliveries} complete! Battery={battery.level:.0f}%")

        # ── Step 8: Cross back to Side-1 ──────────────────────────────────
        cross_obstacle_field(
            ep_robot, ep_camera, ep_chassis, yolo_model,
            dead_reckoner, direction="backward", show=show,
        )

    print(f"\n[Loop] Delivery loop complete. {deliveries} brick(s) delivered. Battery={battery.level:.0f}%")


def _choose_brick(ep_camera, yolo_model: YOLO) -> Optional[int]:
    """Look at one frame and pick the most-confident brick class visible."""
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
    except Empty:
        return None
    if frame is None:
        return None
    dets = get_detections(yolo_model, frame, conf_thresh=0.40)
    bricks = [d for d in dets if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
    if not bricks:
        return None
    best = max(bricks, key=lambda d: d.conf)
    return best.cls


def _search_for_dock(
    ep_chassis,
    ep_camera,
    dead_reckoner: DeadReckoner,
    yolo_model: YOLO,
    world_map: WorldMap,
    show: bool,
) -> None:
    """Spin slowly on Side-1 looking for purple tape or YOLO bricks to find the dock."""
    print("[SearchDock] Spinning to find loading dock ...")
    for _ in range(24):   # 24 × 15° = 360°
        open_loop_turn(ep_chassis, SWEEP_STEP_DEG, dead_reckoner)
        time.sleep(SWEEP_SETTLE_S)
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        purple = detect_purple_dock(frame)
        if purple:
            cx_px, _, _ = purple
            angle_cam = math.atan2(cx_px - K_CAM[0, 2], K_CAM[0, 0])
            bearing = wrap_to_pi(dead_reckoner.pose.yaw + angle_cam)
            dist_m = 0.6   # heuristic
            dx = dead_reckoner.pose.x + dist_m * math.cos(bearing)
            dy = dead_reckoner.pose.y + dist_m * math.sin(bearing)
            world_map.dock = Landmark("dock", dx, dy)
            print(f"  [SearchDock] Dock found at ({dx:.2f}, {dy:.2f})")
            return

        # Also check for bricks as proxy for dock
        dets = get_detections(yolo_model, frame, conf_thresh=0.40)
        bricks = [d for d in dets if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
        if bricks:
            print("  [SearchDock] Bricks visible — treating current position as near dock.")
            world_map.dock = Landmark("dock", dead_reckoner.pose.x, dead_reckoner.pose.y)
            return


# ─────────────────────────────────────────────────────────────────────────────
# Map visualisation (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_map(world_map: WorldMap, robot_pose: Optional[Pose2D] = None) -> None:
    """
    Generate a matplotlib bird's-eye-view map per the project report spec:
      - xy grid lines every 10 cm
      - Red circles for obstacles
      - Blue triangle for small goal
      - Green triangle for large goal
      - Yellow square for loading dock
      - Black square for recharge station
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Map] matplotlib not available. Skipping visualisation.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ARENA_W_M)
    ax.set_ylim(0, ARENA_H_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Project 3 – Arena Map (Bird's-Eye View)")

    # Grid lines every 10 cm
    for v in np.arange(0, ARENA_W_M + 0.01, 0.10):
        ax.axvline(v, color="lightgray", linewidth=0.3)
    for v in np.arange(0, ARENA_H_M + 0.01, 0.10):
        ax.axhline(v, color="lightgray", linewidth=0.3)

    # Perimeter
    ax.plot([0, ARENA_W_M, ARENA_W_M, 0, 0],
            [0, 0, ARENA_H_M, ARENA_H_M, 0], "k-", linewidth=2, label="Boundary")

    # Obstacles: red circles
    for obs in world_map.obstacles:
        circle = plt.Circle((obs.x, obs.y), 0.15, color="red", alpha=0.6)
        ax.add_patch(circle)
    if world_map.obstacles:
        ax.plot([], [], "ro", markersize=10, label="Obstacle")

    # Small goal: blue triangle
    if world_map.small_goal:
        ax.plot(world_map.small_goal.x, world_map.small_goal.y,
                "b^", markersize=14, label="Small goal")

    # Large goal: green triangle
    if world_map.large_goal:
        ax.plot(world_map.large_goal.x, world_map.large_goal.y,
                "g^", markersize=14, label="Large goal")

    # Loading dock: yellow square (25×25 cm)
    if world_map.dock:
        sq = patches.Rectangle(
            (world_map.dock.x - 0.125, world_map.dock.y - 0.125),
            0.25, 0.25,
            linewidth=1, edgecolor="goldenrod", facecolor="yellow", alpha=0.8,
            label="Loading dock",
        )
        ax.add_patch(sq)

    # Recharge station: black square
    if world_map.recharge:
        sq = patches.Rectangle(
            (world_map.recharge.x - 0.10, world_map.recharge.y - 0.10),
            0.20, 0.20,
            linewidth=1, edgecolor="black", facecolor="black", alpha=0.85,
            label="Recharge station",
        )
        ax.add_patch(sq)

    # Robot pose
    if robot_pose is not None:
        ax.plot(robot_pose.x, robot_pose.y, "ms", markersize=10, label="Robot")
        dx = 0.12 * math.cos(robot_pose.yaw)
        dy = 0.12 * math.sin(robot_pose.yaw)
        ax.annotate("", xy=(robot_pose.x + dx, robot_pose.y + dy),
                    xytext=(robot_pose.x, robot_pose.y),
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=2))

    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("arena_map.png", dpi=150)
    print("[Map] Saved to arena_map.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3: Energy-Aware Logistics")
    p.add_argument("--model-path",  default=MODEL_PATH)
    p.add_argument("--robot-ip",    default=ROBOT_IP)
    p.add_argument("--sn",          default=ROBOT_SN)
    p.add_argument("--conn-type",   default="sta", choices=["sta", "ap"])
    p.add_argument("--resolution",  default="360p", choices=["360p", "720p"])
    p.add_argument("--max-deliveries", type=int, default=5)
    p.add_argument("--skip-sweep",  action="store_true",
                   help="Skip the startup sweep (for debugging individual phases).")
    p.add_argument("--show",        action="store_true",
                   help="Show OpenCV debug windows.")
    p.add_argument("--map-only",    action="store_true",
                   help="Run the sweep then show the map and exit.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=== Project 3: Energy-Aware Logistics ===")
    print(f"Loading YOLO model from {args.model_path} ...")
    yolo_model = YOLO(args.model_path)

    print("Connecting to robot ...")
    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = args.robot_ip
    ep_robot  = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=args.sn)
    ep_camera  = ep_robot.camera
    ep_chassis = ep_robot.chassis

    res = rm_camera.STREAM_720P if args.resolution == "720p" else rm_camera.STREAM_360P
    ep_camera.start_video_stream(display=False, resolution=res)

    # ── Initialise subsystems ──────────────────────────────────────────────
    tag_detector  = AprilTagDetector()
    tag_localizer = TagLocalizer()
    dead_reckoner = DeadReckoner(Pose2D(x=0.0, y=0.0, yaw=0.0))
    world_map     = WorldMap()
    battery       = BatteryManager(start_pct=BATTERY_START_PCT)

    try:
        move_arm_to_default(ep_robot)
        ep_robot.gripper.open()

        # ── Phase 1: Startup sweep ─────────────────────────────────────────
        if not args.skip_sweep:
            startup_sweep(
                ep_robot, ep_camera, ep_chassis,
                yolo_model, tag_detector, tag_localizer,
                dead_reckoner, world_map,
                show=args.show,
            )

        if args.map_only:
            visualize_map(world_map, dead_reckoner.pose)
            return

        # ── Phase 2: Delivery loop ─────────────────────────────────────────
        delivery_loop(
            ep_robot, ep_camera, ep_chassis,
            yolo_model, tag_detector, tag_localizer,
            dead_reckoner, world_map, battery,
            max_deliveries=args.max_deliveries,
            show=args.show,
        )

        # ── Phase 3: Final map visualisation ──────────────────────────────
        visualize_map(world_map, dead_reckoner.pose)

    except (KeyboardInterrupt, TimeoutError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
    finally:
        try:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
        except Exception:
            pass
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        ep_robot.close()
        if args.show:
            cv2.destroyAllWindows()
        print("Run complete.")


if __name__ == "__main__":
    main()
