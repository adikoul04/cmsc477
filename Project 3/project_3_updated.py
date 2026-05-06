#!/usr/bin/env python3
"""
Project 3: Energy-Aware Logistics Challenge
CMSC477 - Robotics Perception and Planning

Architecture overview
─────────────────────
The robot operates as a side-aware state machine with three tiers:

  TIER 1  –  Side tracker
      The robot always knows which side of the arena it is on (SIDE1 or SIDE2).
      Transitions happen only through cross_obstacle_field().

  TIER 2  –  Phase state machine  (run_mission)
      SIDE1 actions : PICK_UP_BRICK | RECHARGE | CROSS_TO_SIDE2
      SIDE2 actions : DROP_OFF_BRICK | CROSS_TO_SIDE1
      At battery-low the robot always returns to SIDE1 and recharges before
      the next pick-up.

  TIER 3  –  Low-level closed-loop servos
      Every motion primitive is feedback-driven (visual servo or tag servo);
      open-loop commands are corrected on the next perception cycle.
      Kalman-filtered 2-D world estimates are maintained for all landmarks.

Coordinate system
─────────────────
  Origin : robot start (one of the two top corners of Side-1).
  x      : right (across arena width).
  y      : toward Side-2 (positive deeper into arena).
  yaw    : CCW positive (radians).  0 = facing Side-2 (+y).

Arena  : 3.0 × 3.0 m.
Side-1 : y ∈ [0.0, SIDE1_Y_LIMIT].
Side-2 : y ∈ [SIDE2_Y_START, 3.0].

ArUco tag IDs (set in config.py)
  Recharge station : RECHARGE_TAG_IDS
  Small brick goal : SMALL_GOAL_TAG_IDS
  Large brick goal : LARGE_GOAL_TAG_IDS

YOLO class indices
  0 = cone | 1 = box (obstacle/goal/recharge box) | 2 = small_brick | 3 = large_brick
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
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
    RECHARGE_APPROACH_DIST_M,
    RECHARGE_HOLD_S,
    RECHARGE_STOP_DIST_M,
    RECHARGE_TAG_IDS,
    ROBOT_IP,
    ROBOT_SN,
    SAFE_BOUNDARY_MARGIN_M,
    SIDE1_Y_LIMIT,
    SIDE2_Y_START,
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
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Side(Enum):
    """Which side of the arena the robot currently occupies."""
    SIDE1 = auto()   # loading dock / recharge station side
    SIDE2 = auto()   # delivery goal side


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
    """
    A detected landmark with a Kalman-filtered world position.
    The filter is a simple 2-D constant-position KF:
      state   = [x, y]
      process noise Q = diag(q_pos, q_pos)
      meas.   noise R = diag(r_pos, r_pos)
    """
    kind: str                    # "recharge" | "small_goal" | "large_goal" | "obstacle"
    tag_id: Optional[int] = None

    # Kalman state
    _x: float = field(default=0.0, repr=False)
    _y: float = field(default=0.0, repr=False)
    _P: np.ndarray = field(
        default_factory=lambda: np.eye(2) * 0.5, repr=False
    )   # error covariance (2×2)

    # Noise parameters
    _Q: float = 0.01    # process noise variance (per update)
    _R: float = 0.08    # measurement noise variance

    @classmethod
    def from_observation(cls, kind: str, obs_x: float, obs_y: float,
                         tag_id: Optional[int] = None,
                         obs_noise: float = 0.08) -> "Landmark":
        """Initialise landmark directly from a first observation."""
        lm = cls(kind=kind, tag_id=tag_id)
        lm._x = obs_x
        lm._y = obs_y
        lm._P = np.eye(2) * obs_noise
        lm._R = obs_noise
        return lm

    def update(self, obs_x: float, obs_y: float) -> None:
        """Kalman update step with a new position observation."""
        # Predict (static model — no motion)
        P_pred = self._P + np.eye(2) * self._Q

        # Innovation
        z = np.array([obs_x, obs_y])
        x_pred = np.array([self._x, self._y])
        y_inn = z - x_pred

        # Kalman gain
        S = P_pred + np.eye(2) * self._R
        K = P_pred @ np.linalg.inv(S)

        # State update
        x_upd = x_pred + K @ y_inn
        self._x = float(x_upd[0])
        self._y = float(x_upd[1])
        self._P = (np.eye(2) - K) @ P_pred

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def __repr__(self) -> str:
        return f"Landmark({self.kind}, x={self._x:.2f}, y={self._y:.2f}, tid={self.tag_id})"


@dataclass
class WorldMap:
    """
    Incrementally-built, Kalman-filtered map of the arena.
    Obstacle positions are fused via repeated observations; landmark
    (tag-based) positions are registered once and updated with each re-sighting.
    """
    recharge: Optional[Landmark]   = None
    small_goal: Optional[Landmark] = None
    large_goal: Optional[Landmark] = None
    obstacles: List[Landmark]      = field(default_factory=list)

    # The loading dock is defined as the cluster of bricks (towers) on Side-1.
    # We record it as a mean world position, refined as we re-visit.
    dock_x: Optional[float] = None
    dock_y: Optional[float] = None

    def register_or_update_tag_landmark(
        self,
        kind: str,
        obs_x: float,
        obs_y: float,
        tag_id: int,
    ) -> None:
        """Add or Kalman-update a tag-based landmark (recharge / goal)."""
        target_attr = kind   # "recharge", "small_goal", or "large_goal"
        lm: Optional[Landmark] = getattr(self, target_attr, None)
        if lm is None:
            lm = Landmark.from_observation(kind, obs_x, obs_y, tag_id=tag_id)
            setattr(self, target_attr, lm)
            print(f"  [Map] NEW {kind} at ({obs_x:.2f}, {obs_y:.2f}) tag={tag_id}")
        else:
            lm.update(obs_x, obs_y)
            print(f"  [Map] UPD {kind} → ({lm.x:.2f}, {lm.y:.2f}) tag={tag_id}")

    def register_or_update_obstacle(self, obs_x: float, obs_y: float,
                                    merge_radius: float = 0.40) -> None:
        """
        If a known obstacle is within merge_radius of the observation, fuse
        via Kalman update. Otherwise add a new obstacle landmark.
        """
        for obs in self.obstacles:
            if math.hypot(obs.x - obs_x, obs.y - obs_y) < merge_radius:
                obs.update(obs_x, obs_y)
                return
        # New obstacle
        self.obstacles.append(
            Landmark.from_observation("obstacle", obs_x, obs_y, obs_noise=0.10)
        )
        print(f"  [Map] NEW obstacle at ({obs_x:.2f}, {obs_y:.2f})")

    def set_dock(self, x: float, y: float) -> None:
        """Record / refine the loading-dock centroid (mean of observed positions)."""
        if self.dock_x is None:
            self.dock_x, self.dock_y = x, y
            print(f"  [Map] Loading dock set at ({x:.2f}, {y:.2f})")
        else:
            # EMA smoothing
            alpha = 0.3
            self.dock_x = (1 - alpha) * self.dock_x + alpha * x
            self.dock_y = (1 - alpha) * self.dock_y + alpha * y

    def is_fully_mapped(self) -> bool:
        return all([
            self.recharge is not None,
            self.small_goal is not None,
            self.large_goal is not None,
        ])

    def summary(self) -> str:
        lines = ["=== WorldMap ==="]
        for attr in ["recharge", "small_goal", "large_goal"]:
            lm = getattr(self, attr)
            lines.append(
                f"  {attr}: ({lm.x:.2f}, {lm.y:.2f})" if lm else f"  {attr}: NOT FOUND"
            )
        dock_str = (f"({self.dock_x:.2f}, {self.dock_y:.2f})"
                    if self.dock_x is not None else "NOT FOUND")
        lines.append(f"  dock: {dock_str}")
        lines.append(f"  obstacles: {len(self.obstacles)}")
        return "\n".join(lines)


@dataclass
class MoveAction:
    """Discrete move command recorded for path reversal."""
    dx: float
    dy: float
    dz: float


# ─────────────────────────────────────────────────────────────────────────────
# Battery manager
# ─────────────────────────────────────────────────────────────────────────────

class BatteryManager:
    """Tracks simulated battery level and enforces budget checks."""

    def __init__(self, start_pct: float = BATTERY_START_PCT):
        self.level = start_pct
        print(f"[Battery] Initialised at {self.level:.1f}%")

    def consume(self, brick_class: int) -> None:
        cost = (BATTERY_LARGE_BRICK_COST if brick_class == CLASS_LARGE_BRICK
                else BATTERY_SMALL_BRICK_COST)
        self.level = max(0.0, self.level - cost)
        print(f"[Battery] Consumed {cost}% → {self.level:.1f}% remaining")

    def recharge(self) -> None:
        self.level = BATTERY_RECHARGE_LEVEL
        print("[Battery] Recharged to 100%.")

    def can_afford(self, brick_class: int) -> bool:
        cost = (BATTERY_LARGE_BRICK_COST if brick_class == CLASS_LARGE_BRICK
                else BATTERY_SMALL_BRICK_COST)
        return (self.level - cost) >= BATTERY_RESERVE_PCT

    def needs_recharge(self, brick_class: int = CLASS_SMALL_BRICK) -> bool:
        return not self.can_afford(brick_class)

    @property
    def depleted(self) -> bool:
        return self.level <= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2-D → 3-D pose estimation from camera detections
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_world_position(
    cx_px: float,
    obj_height_px: float,
    assumed_height_m: float,
    robot_pose: Pose2D,
    K: np.ndarray = K_CAM,
) -> Tuple[float, float]:
    """
    Estimate the 2-D world (x, y) position of an object seen by the camera.

    Uses the thin-lens pinhole model:
      depth_m = (assumed_height_m * f_y) / obj_height_px

    Then projects the pixel centre through the camera intrinsics to get a
    bearing angle, and transforms from camera frame into world frame using
    the current robot pose.

    Parameters
    ----------
    cx_px          : pixel x-centre of the bounding box.
    obj_height_px  : bounding-box height in pixels.
    assumed_height_m : known / assumed real-world height of the object (m).
    robot_pose     : current dead-reckoned robot pose in world frame.
    K              : 3×3 camera intrinsic matrix.

    Returns
    -------
    (world_x, world_y) in metres.
    """
    f_y = K[1, 1]
    f_x = K[0, 0]
    cx0 = K[0, 2]   # principal point x

    # Depth from similar triangles
    depth_m = (assumed_height_m * f_y) / max(obj_height_px, 1.0)

    # Bearing angle from camera principal axis (+ = right)
    angle_cam = math.atan2(cx_px - cx0, f_x)

    # Robot-body bearing (camera assumed aligned with forward axis)
    world_bearing = wrap_to_pi(robot_pose.yaw + angle_cam)

    wx = robot_pose.x + depth_m * math.cos(world_bearing)
    wy = robot_pose.y + depth_m * math.sin(world_bearing)
    return wx, wy


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag detector
# ─────────────────────────────────────────────────────────────────────────────

class AprilTagDetector:
    """Thin wrapper around pupil_apriltags with pose estimation enabled."""

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
        t = np.array(detection.pose_t, dtype=float).reshape(3)
        return float(np.linalg.norm(t))

    def tag_center_px(self, detection) -> Tuple[float, float]:
        return float(detection.center[0]), float(detection.center[1])


# ─────────────────────────────────────────────────────────────────────────────
# Action stack — records drive commands for route reversal
# ─────────────────────────────────────────────────────────────────────────────

class ActionStack:
    """LIFO stack of MoveAction used to unwind a route."""

    def __init__(self):
        self.stack: Deque[MoveAction] = deque()

    def push(self, dx: float, dy: float, dz: float) -> None:
        """Push a move action onto the stack."""
        self.stack.append(MoveAction(dx, dy, dz))

    def clear(self) -> None:
        """Clear the stack."""
        self.stack.clear()

    def snapshot(self) -> List[MoveAction]:
        """Return a copy of the current stack as an ordered list."""
        return list(self.stack)

    def unwind(self, ep_chassis) -> None:
        """Drive the exact reverse of every recorded action."""
        print("[ActionStack] Reversing path...")
        while self.stack:
            a = self.stack.pop()
            ep_chassis.move(
                x=-a.dx,
                y=-a.dy,
                z=-a.dz,
                xy_speed=0.6,
                z_speed=45
            ).wait_for_completed()


# ─────────────────────────────────────────────────────────────────────────────
# Tag-based localiser: infer robot world pose from an AprilTag sighting
# ─────────────────────────────────────────────────────────────────────────────

class TagLocalizer:
    """
    Bootstraps tag world poses from dead-reckoned estimates, then uses them
    to produce higher-quality robot pose corrections.
    """

    # Assume camera is at robot centre and aligned with body forward axis.
    T_RC = np.eye(4, dtype=float)

    def __init__(self):
        # tag_id → (world_x, world_y, world_yaw)
        self.tag_world: Dict[int, Tuple[float, float, float]] = {}

    def register_tag_from_robot_pose(
        self, detection, robot_pose: Pose2D
    ) -> None:
        """
        Compute and store the tag's world pose from the current dead-reckoned
        robot pose plus the camera's tag measurement.  Only called once per
        tag (first sighting).
        """
        tag_id = int(detection.tag_id)
        if tag_id in self.tag_world:
            return

        t_ct = np.array(detection.pose_t, dtype=float).reshape(3)
        R_ct = np.array(detection.pose_R, dtype=float).reshape(3, 3)

        T_CT = T_from_Rt(R_ct, t_ct)
        T_WR = T_from_Rt(rotz(robot_pose.yaw),
                         np.array([robot_pose.x, robot_pose.y, 0.0]))
        T_WT = T_WR @ self.T_RC @ T_CT

        wx   = float(T_WT[0, 3])
        wy   = float(T_WT[1, 3])
        wyaw = yaw_from_R(T_WT[:3, :3])
        self.tag_world[tag_id] = (wx, wy, wyaw)
        print(f"  [Localizer] Registered tag {tag_id} at world "
              f"({wx:.2f}, {wy:.2f}, yaw={math.degrees(wyaw):.1f}°)")

    def estimate_pose(self, detection) -> Optional[Pose2D]:
        """
        Given a tag detection and its stored world pose, invert the
        camera-to-tag transform to recover the robot world pose.
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

        return Pose2D(
            x=float(T_WR[0, 3]),
            y=float(T_WR[1, 3]),
            yaw=yaw_from_R(T_WR[:3, :3]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dead-reckoning pose tracker (fallback when no tag is visible)
# ─────────────────────────────────────────────────────────────────────────────

class DeadReckoner:
    """
    Integrates chassis commands into a running world-frame pose estimate.
    Tag-based corrections are fused via a weighted average.
    """

    def __init__(self, initial_pose: Pose2D = Pose2D()):
        self.pose = initial_pose

    def update_from_drive(
        self, vx: float, vy: float, vz_dps: float, dt: float
    ) -> None:
        """
        Integrate one chassis command step.
        vx, vy are body-frame velocities (m/s); vz_dps is yaw rate (°/s).
        """
        dyaw = math.radians(vz_dps) * dt
        mid_yaw = self.pose.yaw + dyaw / 2.0
        dx_w = (vx * math.cos(mid_yaw) - vy * math.sin(mid_yaw)) * dt
        dy_w = (vx * math.sin(mid_yaw) + vy * math.cos(mid_yaw)) * dt
        self.pose.x   += dx_w
        self.pose.y   += dy_w
        self.pose.yaw  = wrap_to_pi(self.pose.yaw + dyaw)
        clamp_pose_to_safe_arena(self.pose)

    def fuse_tag_pose(self, tag_pose: Pose2D, weight: float = 0.7) -> None:
        """
        Weighted blend of current dead-reckoned pose with a tag-derived pose.
        Yaw is blended via unit-vector SLERP to avoid wrap-around issues.
        """
        w = weight
        self.pose.x   = (1 - w) * self.pose.x   + w * tag_pose.x
        self.pose.y   = (1 - w) * self.pose.y   + w * tag_pose.y
        u0  = np.array([math.cos(self.pose.yaw), math.sin(self.pose.yaw)])
        u1  = np.array([math.cos(tag_pose.yaw),  math.sin(tag_pose.yaw)])
        u   = (1 - w) * u0 + w * u1
        self.pose.yaw = math.atan2(u[1], u[0])
        clamp_pose_to_safe_arena(self.pose)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t.ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def shift_world_map_x(world_map: WorldMap, delta_x: float) -> None:
    """Shift every stored world X coordinate by a constant offset."""
    if abs(delta_x) < 1e-9:
        return

    if world_map.recharge is not None:
        world_map.recharge._x += delta_x
    if world_map.small_goal is not None:
        world_map.small_goal._x += delta_x
    if world_map.large_goal is not None:
        world_map.large_goal._x += delta_x
    for obs in world_map.obstacles:
        obs._x += delta_x
    if world_map.dock_x is not None:
        world_map.dock_x += delta_x


def reconcile_recharge_obstacles(world_map: WorldMap, radius_m: float = 0.40) -> None:
    """Drop obstacle entries that overlap the confirmed recharge landmark."""
    if world_map.recharge is None:
        return
    rx, ry = world_map.recharge.x, world_map.recharge.y
    world_map.obstacles = [
        obs for obs in world_map.obstacles
        if math.hypot(obs.x - rx, obs.y - ry) >= radius_m
    ]


def infer_start_corner(
    recharge_range_m: Optional[float],
    dock_range_m: Optional[float],
) -> Optional[str]:
    """Infer the top-left/top-right start corner from raw sweep distances."""
    if recharge_range_m is None or dock_range_m is None:
        return None
    return "top-left" if recharge_range_m < dock_range_m else "top-right"


def infer_start_corner_from_frame(
    dock_range_m: Optional[float],
    recharge_like_range_m: Optional[float],
) -> Optional[str]:
    """Infer the start corner from one sweep frame containing both landmarks."""
    if dock_range_m is None or recharge_like_range_m is None:
        return None
    return "top-right" if dock_range_m < recharge_like_range_m else "top-left"


def apply_start_corner_correction(
    assumed_corner: str,
    inferred_corner: Optional[str],
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
) -> str:
    """Correct the world frame if the actual start corner differs from the seed."""
    if inferred_corner is None or inferred_corner == assumed_corner:
        return assumed_corner

    assumed_x = SAFE_BOUNDARY_MARGIN_M if assumed_corner == "top-left" else ARENA_W_M - SAFE_BOUNDARY_MARGIN_M
    actual_x = SAFE_BOUNDARY_MARGIN_M if inferred_corner == "top-left" else ARENA_W_M - SAFE_BOUNDARY_MARGIN_M
    delta_x = actual_x - assumed_x

    shift_world_map_x(world_map, delta_x)
    dead_reckoner.pose.x = min(
        max(dead_reckoner.pose.x + delta_x, SAFE_BOUNDARY_MARGIN_M),
        ARENA_W_M - SAFE_BOUNDARY_MARGIN_M,
    )
    print(f"[StartCorner] Corrected world frame: {assumed_corner} → {inferred_corner} (dx={delta_x:.2f}m)")
    return inferred_corner


def clamp_pose_to_safe_arena(pose: Pose2D) -> None:
    """Keep the dead-reckoned pose inside the safe interior of the arena."""
    pose.x = min(max(pose.x, SAFE_BOUNDARY_MARGIN_M), ARENA_W_M - SAFE_BOUNDARY_MARGIN_M)
    pose.y = min(max(pose.y, SAFE_BOUNDARY_MARGIN_M), ARENA_H_M - SAFE_BOUNDARY_MARGIN_M)


def clamp_body_translation_to_safe_arena(
    pose: Pose2D,
    dx_body: float,
    dy_body: float,
) -> Tuple[float, float]:
    """Scale a body-frame translation so the predicted pose stays inside the safe interior."""
    if abs(dx_body) < 1e-9 and abs(dy_body) < 1e-9:
        return 0.0, 0.0

    mid_yaw = pose.yaw
    dx_w = dx_body * math.cos(mid_yaw) - dy_body * math.sin(mid_yaw)
    dy_w = dx_body * math.sin(mid_yaw) + dy_body * math.cos(mid_yaw)

    min_x = SAFE_BOUNDARY_MARGIN_M
    max_x = ARENA_W_M - SAFE_BOUNDARY_MARGIN_M
    min_y = SAFE_BOUNDARY_MARGIN_M
    max_y = ARENA_H_M - SAFE_BOUNDARY_MARGIN_M

    scale = 1.0
    if dx_w > 1e-9:
        scale = min(scale, max(0.0, (max_x - pose.x) / dx_w))
    elif dx_w < -1e-9:
        scale = min(scale, max(0.0, (min_x - pose.x) / dx_w))

    if dy_w > 1e-9:
        scale = min(scale, max(0.0, (max_y - pose.y) / dy_w))
    elif dy_w < -1e-9:
        scale = min(scale, max(0.0, (min_y - pose.y) / dy_w))

    scale = min(max(scale, 0.0), 1.0)
    return dx_body * scale, dy_body * scale


# ─────────────────────────────────────────────────────────────────────────────
# Low-level chassis helpers
# ─────────────────────────────────────────────────────────────────────────────

def chassis_stop(ep_chassis, hold_s: float = 0.1) -> None:
    ep_chassis.move(x=0.0, y=0.0, z=0.0, xy_speed=0.7, z_speed=45).wait_for_completed()
    time.sleep(hold_s)


def open_loop_turn(
    ep_chassis, degrees: float, dead_reckoner: DeadReckoner, stack: Optional[ActionStack] = None
) -> None:
    """Turn in place by a given angle (+° = CCW). Updates dead reckoner and action stack."""
    if abs(degrees) < 0.5:
        return
    ep_chassis.move(x=0, y=0, z=degrees, z_speed=45).wait_for_completed()
    if stack:
        stack.push(0, 0, degrees)
    dead_reckoner.update_from_drive(0.0, 0.0, TURN_SPEED_DPS * (1 if degrees > 0 else -1), abs(degrees) / TURN_SPEED_DPS)


def open_loop_drive(
    ep_chassis,
    dist_m: float,
    dead_reckoner: DeadReckoner,
    vx: float = MOVE_SPEED_MPS,
    vy: float = 0.0,
    stack: Optional[ActionStack] = None,
) -> None:
    """
    Drive forward (or laterally) a fixed distance in metres.
    Negative dist_m reverses the robot. Updates dead reckoner and action stack.
    """
    if abs(dist_m) < 0.01:
        return
    dist_m, vy = clamp_body_translation_to_safe_arena(dead_reckoner.pose, dist_m, vy)
    if abs(dist_m) < 0.01 and abs(vy) < 0.01:
        return
    ep_chassis.move(x=dist_m, y=vy if abs(vy) > 0.005 else 0, z=0, xy_speed=0.7).wait_for_completed()
    if stack:
        stack.push(dist_m, vy if abs(vy) > 0.005 else 0, 0)
    speed = math.hypot(abs(vx), abs(vy)) if abs(vy) > 0.005 else abs(vx)
    dt = abs(dist_m) / (speed if speed > 1e-6 else 0.7)
    sign = 1.0 if dist_m > 0 else -1.0
    act_vx = sign * abs(vx)
    act_vy = vy
    dead_reckoner.update_from_drive(act_vx, act_vy, 0.0, dt)


def turn_to_heading(
    ep_chassis, target_yaw_rad: float, dead_reckoner: DeadReckoner
) -> None:
    """Turn to face an absolute world yaw angle."""
    delta = wrap_to_pi(target_yaw_rad - dead_reckoner.pose.yaw)
    open_loop_turn(ep_chassis, math.degrees(delta), dead_reckoner)


def navigate_to_world_point(
    ep_chassis,
    dead_reckoner: DeadReckoner,
    wx: float,
    wy: float,
    stop_dist_m: float = 0.20,
    speed: float = MOVE_SPEED_MPS,
) -> None:
    """
    Dead-reckoning point-to-point navigation.
    Turns to face the target then drives the Euclidean distance minus
    stop_dist_m.  Fine positioning is left to the visual servos.
    """
    wx = min(max(wx, SAFE_BOUNDARY_MARGIN_M), ARENA_W_M - SAFE_BOUNDARY_MARGIN_M)
    wy = min(max(wy, SAFE_BOUNDARY_MARGIN_M), ARENA_H_M - SAFE_BOUNDARY_MARGIN_M)
    pose = dead_reckoner.pose
    dx = wx - pose.x
    dy = wy - pose.y
    dist = math.hypot(dx, dy)
    if dist <= stop_dist_m:
        return
    turn_to_heading(ep_chassis, math.atan2(dy, dx), dead_reckoner)
    drive_dist = max(0.0, dist - stop_dist_m)
    open_loop_drive(ep_chassis, drive_dist, dead_reckoner, vx=speed)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP SWEEP — 360° spin to map tags and obstacle field
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
) -> Optional[str]:
    """
    Spin the robot 360° in SWEEP_STEP_DEG steps.  At each step:
      • Run AprilTag detection → register recharge and goal landmarks.
      • Run YOLO → detect obstacle boxes (those without ArUco tags are
        obstacles, those with tags belong to goals/recharge, tracked via tags).
      • Fuse any tag-based pose corrections into the dead reckoner.

    After the sweep the robot is back at its starting orientation.
    Tag-bearing boxes (recharge / goals) are identified purely by their
    ArUco ID; plain boxes without tags are obstacles.
    """
    print("[Sweep] Starting 360° mapping sweep ...")
    total_turned = 0.0
    best_recharge_range_m: Optional[float] = None
    best_dock_range_m: Optional[float] = None
    corner_votes: List[str] = []

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

        # ── AprilTag detection ─────────────────────────────────────────────
        tags = tag_detector.find_tags(gray)
        # Collect the pixel columns occupied by tags this frame so we can
        # suppress obstacle registration for boxes that are carrying a tag.
        tag_cols = set()
        for det in tags:
            tid = int(det.tag_id)
            if tid not in ALL_LANDMARK_TAG_IDS:
                continue

            tag_cols.add(int(det.center[0]))   # rough column

            # Bootstrap tag world pose from dead-reckoned robot pose
            tag_localizer.register_tag_from_robot_pose(det, dead_reckoner.pose)

            tag_range_m = tag_detector.tag_distance_m(det)
            if tid in RECHARGE_TAG_IDS:
                if best_recharge_range_m is None or tag_range_m < best_recharge_range_m:
                    best_recharge_range_m = tag_range_m

            # If already registered, refine robot pose and fuse
            refined = tag_localizer.estimate_pose(det)
            if refined is not None:
                dead_reckoner.fuse_tag_pose(refined, weight=0.6)

            # Map the landmark
            wx, wy, _ = tag_localizer.tag_world[tid]
            if tid in RECHARGE_TAG_IDS:
                kind = "recharge"
            elif tid in SMALL_GOAL_TAG_IDS:
                kind = "small_goal"
            elif tid in LARGE_GOAL_TAG_IDS:
                kind = "large_goal"
            else:
                continue
            world_map.register_or_update_tag_landmark(kind, wx, wy, tid)

        # Track the loading dock as the brick cluster nearest the robot
        dets = get_detections(yolo_model, frame, conf_thresh=0.40)
        bricks = [d for d in dets if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
        dock_range_m: Optional[float] = None
        dock_cx: Optional[float] = None
        if len(bricks) >= 2:
            dock_cx = float(np.mean([b.cx for b in bricks]))
            mean_h = float(np.mean([b.h for b in bricks]))
            dock_range_m = (0.10 * K_CAM[1, 1]) / max(mean_h, 1.0)
            if best_dock_range_m is None or dock_range_m < best_dock_range_m:
                best_dock_range_m = dock_range_m
                wx, wy = pixel_to_world_position(
                    cx_px=dock_cx,
                    obj_height_px=mean_h,
                    assumed_height_m=0.10,
                    robot_pose=dead_reckoner.pose,
                )
                world_map.set_dock(wx, wy)

        # Detect a recharge-like block in the same frame even if the tag face
        # is not visible. This lets us infer the start corner from relative depth.
        recharge_like_range_m: Optional[float] = None
        recharge_tag_dets = [det for det in tags if int(det.tag_id) in RECHARGE_TAG_IDS]
        if recharge_tag_dets:
            best_recharge_det = min(recharge_tag_dets, key=tag_detector.tag_distance_m)
            recharge_like_range_m = tag_detector.tag_distance_m(best_recharge_det)
        else:
            obstacle_candidates = [d for d in dets if d.cls == CLASS_BOX]
            if dock_cx is not None and obstacle_candidates:
                nearby = [
                    d for d in obstacle_candidates
                    if abs(float(d.cx) - dock_cx) < 120.0
                ]
                if nearby:
                    recharge_candidate = min(nearby, key=lambda d: abs(float(d.cx) - dock_cx))
                    recharge_like_range_m = (0.30 * K_CAM[1, 1]) / max(recharge_candidate.h, 1.0)

        if dock_range_m is not None and recharge_like_range_m is not None:
            frame_corner = infer_start_corner_from_frame(dock_range_m, recharge_like_range_m)
            if frame_corner is not None:
                corner_votes.append(frame_corner)

        # ── YOLO: detect obstacle boxes (no ArUco tag on them) ────────────
        dets = [d for d in dets if d.cls == CLASS_BOX]
        for d in dets:
            if d.cls != CLASS_BOX:
                continue
            # Suppress boxes that visually overlap a detected ArUco tag column
            if any(abs(int(d.cx) - tc) < 40 for tc in tag_cols):
                continue   # this box is carrying a tag → not a plain obstacle
            # 2-D → 3-D projection using assumed fabric-box height
            wx, wy = pixel_to_world_position(
                cx_px=d.cx,
                obj_height_px=d.h,
                assumed_height_m=0.30,   # typical fabric-box height
                robot_pose=dead_reckoner.pose,
            )
            world_map.register_or_update_obstacle(wx, wy)

        reconcile_recharge_obstacles(world_map)

        if show:
            dbg = frame.copy()
            for det in tags:
                pts = det.corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(dbg, [pts], True, (0, 0, 255), 2)
                c = det.center.astype(int)
                cv2.putText(dbg, str(det.tag_id), tuple(c),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("sweep", dbg)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[Sweep] Complete.")
    print(world_map.summary())

    inferred_corner: Optional[str] = None
    if corner_votes:
        inferred_corner = max(set(corner_votes), key=corner_votes.count)
    else:
        inferred_corner = infer_start_corner(best_recharge_range_m, best_dock_range_m)
    if inferred_corner is not None:
        print(f"[StartCorner] Inferred {inferred_corner} from initial sweep")
    return inferred_corner


# ─────────────────────────────────────────────────────────────────────────────
# LOADING DOCK SEARCH — find bricks cluster on Side-1
# ─────────────────────────────────────────────────────────────────────────────

def find_loading_dock(
    ep_chassis,
    ep_camera,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    show: bool = False,
) -> bool:
    """
    The loading dock is wherever multiple towers (bricks) are found on Side-1.
    We spin slowly on Side-1, collecting YOLO detections.  As soon as a frame
    contains ≥ 2 bricks we declare that bearing as the dock direction and
    project a world position using monocular depth.

    Returns True if the dock was found, False on a full 360° without success.
    """
    print("[Dock] Searching for loading dock (brick cluster on Side-1) ...")

    total_turned = 0.0
    best_frame: Optional[np.ndarray] = None
    best_count = 0
    best_pose: Optional[Pose2D] = None

    while total_turned < 360.0:
        open_loop_turn(ep_chassis, SWEEP_STEP_DEG, dead_reckoner)
        total_turned += SWEEP_STEP_DEG
        time.sleep(SWEEP_SETTLE_S)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        dets = get_detections(yolo_model, frame, conf_thresh=0.40)
        bricks = [d for d in dets
                  if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]

        if len(bricks) > best_count:
            best_count = len(bricks)
            best_frame = frame
            # Snapshot the robot pose at this orientation
            best_pose = Pose2D(
                x=dead_reckoner.pose.x,
                y=dead_reckoner.pose.y,
                yaw=dead_reckoner.pose.yaw,
            )

        if best_count >= 2:
            # Good enough — project the centroid of visible bricks into world frame
            assert best_pose is not None
            assert best_frame is not None
            dets2 = get_detections(yolo_model, best_frame, conf_thresh=0.40)
            bricks2 = [d for d in dets2
                       if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
            mean_cx = float(np.mean([b.cx for b in bricks2]))
            mean_h  = float(np.mean([b.h  for b in bricks2]))
            wx, wy = pixel_to_world_position(
                cx_px=mean_cx,
                obj_height_px=mean_h,
                assumed_height_m=0.10,   # small / large brick height
                robot_pose=best_pose,
            )
            world_map.set_dock(wx, wy)
            print(f"[Dock] Loading dock found ({best_count} bricks) at ({wx:.2f}, {wy:.2f})")
            if show and best_frame is not None:
                cv2.imshow("dock_found", best_frame)
                cv2.waitKey(500)
            return True

    # Fallback: use best single-brick sighting, or stay at current position
    if best_count >= 1 and best_pose is not None:
        assert best_frame is not None
        dets2 = get_detections(yolo_model, best_frame, conf_thresh=0.40)
        bricks2 = [d for d in dets2
                   if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
        wx, wy = pixel_to_world_position(
            cx_px=bricks2[0].cx,
            obj_height_px=bricks2[0].h,
            assumed_height_m=0.10,
            robot_pose=best_pose,
        )
        world_map.set_dock(wx, wy)
        print(f"[Dock] Fallback: dock from single brick at ({wx:.2f}, {wy:.2f})")
        return True

    print("[Dock] WARNING: no bricks found. Dock location unknown.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE-FIELD CROSSING — reactive navigation between sides
# ─────────────────────────────────────────────────────────────────────────────

def _find_obstacle_field_heading(
    ep_chassis,
    ep_camera,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    min_obstacles: int = 1,
    tag_detector: Optional[AprilTagDetector] = None,
    show: bool = False,
) -> bool:
    """
    Rotate until the robot faces at least `min_obstacles` plain obstacle boxes
    (i.e. boxes with no ArUco tag visible alongside them).  This aligns the
    robot with the obstacle field before attempting a crossing.

    Returns True once a suitable heading is found, False after a full 360°.
    """
    print("[ObsField] Searching for obstacle-field heading ...")
    total_turned = 0.0

    while total_turned < 360.0:
        open_loop_turn(ep_chassis, SWEEP_STEP_DEG, dead_reckoner)
        total_turned += SWEEP_STEP_DEG
        time.sleep(SWEEP_SETTLE_S)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        # Identify any ArUco tags in this frame (columns to suppress)
        tag_cols: set = set()
        if tag_detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            tags = tag_detector.find_tags(gray)
            for det in tags:
                if int(det.tag_id) in ALL_LANDMARK_TAG_IDS:
                    tag_cols.add(int(det.center[0]))

        # Count plain obstacle boxes (no tag overlap)
        dets = get_detections(yolo_model, frame, conf_thresh=0.40)
        plain_boxes = [
            d for d in dets
            if d.cls == CLASS_BOX
            and not any(abs(int(d.cx) - tc) < 40 for tc in tag_cols)
        ]

        if len(plain_boxes) >= min_obstacles:
            print(f"  [ObsField] Found {len(plain_boxes)} obstacle(s) ahead — aligned.")
            if show:
                cv2.imshow("obs_align", frame)
                cv2.waitKey(300)
            return True

    print("[ObsField] WARNING: could not find obstacle field heading.")
    return False


def _nearest_obstacle_dist(yolo_model: YOLO, frame: np.ndarray,
                            tag_cols: set = None) -> float:
    """
    Monocular depth estimate to the closest plain obstacle box in the frame.
    Returns inf if none found.
    """
    dets = get_detections(yolo_model, frame, conf_thresh=0.40)
    boxes = [d for d in dets if d.cls == CLASS_BOX]
    if tag_cols:
        boxes = [b for b in boxes
                 if not any(abs(int(b.cx) - tc) < 40 for tc in tag_cols)]
    if not boxes:
        return float("inf")
    tallest = max(boxes, key=lambda d: d.h)   # tallest → closest
    dist_m = (0.30 * K_CAM[1, 1]) / max(tallest.h, 1.0)
    return dist_m


def cross_obstacle_field(
    ep_robot,
    ep_camera,
    ep_chassis,
    yolo_model: YOLO,
    tag_detector: AprilTagDetector,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    current_side: Side,
    timeout_s: float = 60.0,
    show: bool = False,
) -> Side:
    """
    Cross the obstacle field from the current side to the other.

    Strategy
    ────────
    1. Orient toward obstacle field (≥ 1 plain obstacle box visible).
    2. While not yet across:
       a. Grab a frame; check for obstacles ahead.
       b. If clear → drive forward one step; update obstacle map via Kalman.
       c. If blocked → slide laterally to find a gap.
    3. Declare side transition once the dead-reckoner y-coordinate crosses
       the expected threshold.

    Returns the new Side.
    """
    direction = "forward" if current_side == Side.SIDE1 else "backward"
    fwd_sign  = 1.0 if direction == "forward" else -1.0
    target_y  = ARENA_H_M * 0.65 if direction == "forward" else SIDE1_Y_LIMIT

    print(f"[CrossField] Crossing obstacle field ({direction}: Side-1→Side-2)...")

    # Step 1: align with obstacle field
    _find_obstacle_field_heading(
        ep_chassis, ep_camera, yolo_model, dead_reckoner,
        min_obstacles=1, tag_detector=tag_detector, show=show,
    )

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        # Identify tag columns to suppress from obstacle distance estimate
        tag_cols: set = set()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        for det in tag_detector.find_tags(gray):
            if int(det.tag_id) in ALL_LANDMARK_TAG_IDS:
                tag_cols.add(int(det.center[0]))

        dist = _nearest_obstacle_dist(yolo_model, frame, tag_cols)

        if dist >= OBS_CLEAR_DIST_M:
            # ── Free corridor ahead: drive one step ───────────────────────
            dist_step, _ = clamp_body_translation_to_safe_arena(dead_reckoner.pose, fwd_sign * 0.25, 0.0)
            if abs(dist_step) < 0.01:
                chassis_stop(ep_chassis)
                print("[CrossField] WARNING: boundary buffer reached. Stopping before edge.")
                return current_side
            ep_chassis.move(x=dist_step, y=0.0, z=0.0, xy_speed=OBS_FWD_SPEED_MPS).wait_for_completed()
            dead_reckoner.update_from_drive(fwd_sign * OBS_FWD_SPEED_MPS, 0.0, 0.0, 0.3)

            # Update obstacle map for any visible obstacles this frame
            dets = get_detections(yolo_model, frame, conf_thresh=0.40)
            for d in dets:
                if d.cls == CLASS_BOX and not any(
                    abs(int(d.cx) - tc) < 40 for tc in tag_cols
                ):
                    ox, oy = pixel_to_world_position(
                        d.cx, d.h, 0.30, dead_reckoner.pose
                    )
                    world_map.register_or_update_obstacle(ox, oy)

            # Check arrival criterion
            crossed = (
                (direction == "forward"  and dead_reckoner.pose.y > target_y) or
                (direction == "backward" and dead_reckoner.pose.y < target_y)
            )
            if crossed:
                chassis_stop(ep_chassis)
                new_side = Side.SIDE2 if direction == "forward" else Side.SIDE1
                print(f"[CrossField] Reached {new_side.name}.")
                return new_side
        else:
            # ── Obstacle ahead: slide to find a gap ───────────────────────
            chassis_stop(ep_chassis)
            cleared = _slide_to_clear_corridor(
                ep_chassis, ep_camera, yolo_model, dead_reckoner, fwd_sign, tag_cols, show, stack=None
            )
            if not cleared:
                print("[CrossField] WARNING: could not find gap. Retrying ...")
                time.sleep(0.5)

        if show:
            cv2.imshow("cross_field", frame)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[CrossField] WARNING: timeout reached. Assuming crossed.")
    return Side.SIDE2 if direction == "forward" else Side.SIDE1


def _slide_to_clear_corridor(
    ep_chassis,
    ep_camera,
    yolo_model: YOLO,
    dead_reckoner: DeadReckoner,
    fwd_sign: float,
    tag_cols: set,
    show: bool,
    stack: Optional[ActionStack] = None,
) -> bool:
    """
    Slide left then right (two steps right if left fails) to find a clear
    forward corridor.  Returns True once a gap is found.
    """
    for lateral_sign in [1.0, -1.0, -1.0]:
        dist, _ = clamp_body_translation_to_safe_arena(dead_reckoner.pose, 0.0, lateral_sign * 0.3)
        if abs(dist) < 0.01:
            continue
        ep_chassis.move(x=0.0, y=dist, z=0.0, xy_speed=0.6).wait_for_completed()
        if stack:
            stack.push(0.0, dist, 0.0)
        dead_reckoner.update_from_drive(0.0, lateral_sign * OBS_SLIDE_SPEED_MPS, 0.0, 0.5)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue
        if _nearest_obstacle_dist(yolo_model, frame, tag_cols) >= OBS_CLEAR_DIST_M:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# AprilTag visual servo — approach any of a set of tag IDs head-on
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
    Closed-loop visual servo toward an AprilTag.

    Control law:
      • If tag not centred → yaw correction only (no forward motion).
      • If centred → forward/backward correction to reach target_dist_m.

    The robot continually corrects both yaw and distance on every camera
    frame, so open-loop error from chassis slip is continuously compensated.

    Returns True on success, False on timeout.
    """
    print(f"[TagServo] Servoing to tag IDs {target_tag_ids}, "
          f"target dist={target_dist_m:.2f}m ...")
    t0     = time.time()
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
            # Tag not visible — spin slowly to search
            ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=TAG_SERVO_STEP_S)
            dead_reckoner.update_from_drive(0.0, 0.0, 8.0, TAG_SERVO_STEP_S)
            time.sleep(TAG_SERVO_STEP_S)
            stable = 0
            continue

        # Pick closest tag to approach
        best    = min(relevant, key=tag_detector.tag_distance_m)
        dist    = tag_detector.tag_distance_m(best)
        cx_px, _ = tag_detector.tag_center_px(best)
        err_px  = cx_px - frame.shape[1] / 2.0

        if abs(err_px) > TAG_SERVO_CENTER_TOL_PX:
            # Phase 1: yaw-only correction to centre the tag
            vz = clamp(-TAG_SERVO_K_YAW * err_px,
                       -TAG_SERVO_MAX_YAW_DPS, TAG_SERVO_MAX_YAW_DPS)
            vx = 0.0
        else:
            # Phase 2: forward correction only once centred
            err_dist = dist - target_dist_m
            vx = clamp(-TAG_SERVO_K_FWD * err_dist, -TAG_SERVO_MAX_V, TAG_SERVO_MAX_V)
            vz = 0.0

        step_dx, step_dy = clamp_body_translation_to_safe_arena(
            dead_reckoner.pose,
            vx * TAG_SERVO_STEP_S,
            0.0,
        )
        vx = step_dx / TAG_SERVO_STEP_S

        ep_chassis.drive_speed(x=vx, y=0.0, z=vz, timeout=TAG_SERVO_STEP_S)
        dead_reckoner.update_from_drive(vx, 0.0, vz, TAG_SERVO_STEP_S)
        time.sleep(TAG_SERVO_STEP_S)

        # Stable-arrival check: must be close AND centred for ≥ 3 frames
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
# RECHARGING protocol
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
    Full recharging protocol (must be called from Side-1):
      1. Navigate near the recharge station (dead-reckoning).
      2. Visually servo to within 5 cm head-on (≥ 30 cm approach start).
      3. Hold stationary for RECHARGE_HOLD_S seconds.
      4. Restore battery to 100%.
    """
    if world_map.recharge is None:
        print("[Recharge] ERROR: recharge station not mapped!")
        return

    print(f"[Recharge] Navigating to recharge station at "
          f"({world_map.recharge.x:.2f}, {world_map.recharge.y:.2f}) ...")

    # Step 1: approach to within RECHARGE_APPROACH_DIST_M via dead-reckoning
    navigate_to_world_point(
        ep_chassis, dead_reckoner,
        world_map.recharge.x, world_map.recharge.y,
        stop_dist_m=RECHARGE_APPROACH_DIST_M + 0.10,
    )

    # Step 2: visual servo to 5 cm, head-on
    tag_ids = (
        {world_map.recharge.tag_id}
        if world_map.recharge.tag_id is not None
        else RECHARGE_TAG_IDS
    )
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

    # Step 3: hold stationary
    print(f"[Recharge] Holding for {RECHARGE_HOLD_S}s ...")
    chassis_stop(ep_chassis, hold_s=RECHARGE_HOLD_S)

    # Step 4: restore battery
    battery.recharge()


# ─────────────────────────────────────────────────────────────────────────────
# BRICK APPROACH — closed-loop YOLO visual servo on Side-1
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
    Visually servo toward the nearest brick of the given class using
    lateral (strafe) corrections to centre the brick, then forward
    corrections to close distance.

    Every chassis command is pushed onto action_stack so the route can be
    reversed to return to the dock area after delivery.

    Returns True on success (brick centred and close), False on timeout.
    """
    print(f"[BrickApproach] Approaching brick class {brick_class} ...")
    action_stack.clear()
    move_arm_to_default(ep_robot)

    stable        = 0
    center_stable = 0
    t0            = time.time()

    while time.time() - t0 < timeout_s:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        dets = get_detections(yolo_model, frame, conf_thresh=0.40,
                              target_class=brick_class)
        if not dets:
            # Brick not visible — spin slowly to search
            ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=BRICK_SERVO_STEP_S)
            action_stack.push(0.0, 0.0, 0.0)  # Store spinning as no net displacement
            dead_reckoner.update_from_drive(0.0, 0.0, 8.0, BRICK_SERVO_STEP_S)
            time.sleep(BRICK_SERVO_STEP_S)
            stable = 0
            continue

        frame_w  = frame.shape[1]
        frame_h  = frame.shape[0]
        frame_cx = frame_w / 2.0

        # Choose the most-confident detection
        selected   = select_detection(dets, selection_mode="conf",
                                      frame_center_x=frame_cx)
        y_top      = selected.cy - selected.h / 2.0
        err_x      = selected.cx - frame_cx
        target_top = BRICK_SERVO_TOP_Y_RATIO * frame_h
        err_fwd    = target_top - y_top     # positive → brick too far

        # Phase 1: strafe to centre brick horizontally
        centered      = abs(err_x) < BRICK_SERVO_CENTER_TOL_PX
        center_stable = center_stable + 1 if centered else 0

        if center_stable < 2:
            vy = clamp(BRICK_SERVO_K_LAT * err_x, -BRICK_SERVO_MAX_V, BRICK_SERVO_MAX_V)
            vx = 0.0
        else:
            # Phase 2: drive forward once centred
            vx = clamp(BRICK_SERVO_K_FWD * err_fwd, -BRICK_SERVO_MAX_V, BRICK_SERVO_MAX_V)
            vy = 0.0

        step_dx, step_dy = clamp_body_translation_to_safe_arena(
            dead_reckoner.pose,
            vx * BRICK_SERVO_STEP_S,
            vy * BRICK_SERVO_STEP_S,
        )
        vx = step_dx / BRICK_SERVO_STEP_S
        vy = step_dy / BRICK_SERVO_STEP_S

        ep_chassis.drive_speed(x=vx, y=vy, z=0.0, timeout=BRICK_SERVO_STEP_S)
        # Store small incremental movements on action stack
        action_stack.push(0.02 * vx, 0.02 * vy, 0)
        dead_reckoner.update_from_drive(vx, vy, 0.0, BRICK_SERVO_STEP_S)
        time.sleep(BRICK_SERVO_STEP_S)

        # Arrival: brick is centred AND at the correct forward distance
        if centered and abs(err_fwd) < BRICK_SERVO_TOP_TOL_PX:
            stable += 1
        else:
            stable = 0

        if stable >= BRICK_SERVO_STABLE_THRESH:
            chassis_stop(ep_chassis)
            print("  [BrickApproach] Arrived at brick.")
            return True

        if show:
            dbg = frame.copy()
            x1 = int(selected.cx - selected.w / 2)
            y1 = int(selected.cy - selected.h / 2)
            cv2.rectangle(dbg, (x1, y1),
                          (x1 + int(selected.w), y1 + int(selected.h)), (0, 255, 0), 2)
            cv2.putText(dbg, f"err_x={err_x:+.0f} fwd={err_fwd:+.0f} stb={stable}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("brick_approach", dbg)
            cv2.waitKey(1)

    chassis_stop(ep_chassis)
    print("[BrickApproach] Timeout.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERY — approach goal, place brick, back away (Side-2)
# ─────────────────────────────────────────────────────────────────────────────

def deliver_brick(
    ep_robot,
    ep_camera,
    ep_chassis,
    tag_detector: AprilTagDetector,
    dead_reckoner: DeadReckoner,
    goal_landmark: Landmark,
    show: bool = False,
) -> None:
    """
    Navigate to the correct goal AprilTag on Side-2 and place the brick.

    Sequence:
      1. Dead-reckon toward mapped goal position (coarse).
      2. Visual servo to 22 cm head-on from the tag (fine).
      3. Place brick (arm action).
      4. Back 30 cm to avoid touching the goal.
    """
    goal_tag_ids = (
        SMALL_GOAL_TAG_IDS if goal_landmark.kind == "small_goal"
        else LARGE_GOAL_TAG_IDS
    )
    if goal_landmark.tag_id is not None:
        goal_tag_ids = {goal_landmark.tag_id}

    print(f"[Deliver] Heading to {goal_landmark.kind} at "
          f"({goal_landmark.x:.2f}, {goal_landmark.y:.2f}) ...")

    # Coarse navigation
    navigate_to_world_point(
        ep_chassis, dead_reckoner,
        goal_landmark.x, goal_landmark.y,
        stop_dist_m=0.5,
    )

    # Fine visual servo
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

    # Back away to avoid touching the goal
    open_loop_drive(ep_chassis, -0.30, dead_reckoner)
    print("[Deliver] Brick placed. Backed away.")

    # Turn away from goal to face obstacle field (for return crossing)
    print("[Deliver] Turning 180° to face return path...")
    open_loop_turn(ep_chassis, 180.0, dead_reckoner)


# ─────────────────────────────────────────────────────────────────────────────
# CHOOSE BRICK — identify best brick class from current camera view
# ─────────────────────────────────────────────────────────────────────────────

def choose_brick(ep_camera, yolo_model: YOLO) -> Optional[int]:
    """
    Grab one camera frame and return the class (small or large) of the most-
    confident brick detection, or None if no bricks are visible.
    """
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
    except Empty:
        return None
    if frame is None:
        return None
    dets   = get_detections(yolo_model, frame, conf_thresh=0.40)
    bricks = [d for d in dets if d.cls in (CLASS_SMALL_BRICK, CLASS_LARGE_BRICK)]
    if not bricks:
        return None
    return max(bricks, key=lambda d: d.conf).cls


# ─────────────────────────────────────────────────────────────────────────────
# GOAL SEARCH — discover goal landmarks on Side 2 if missed during sweep
# ─────────────────────────────────────────────────────────────────────────────

def search_for_unmapped_goals(
    ep_chassis,
    ep_camera,
    tag_detector: AprilTagDetector,
    tag_localizer: TagLocalizer,
    dead_reckoner: DeadReckoner,
    world_map: WorldMap,
    show: bool = False,
) -> None:
    """
    If goals weren't detected during startup sweep (e.g., blocked by obstacles),
    search for them by rotating on Side 2. Updates world_map with discovered goals.
    """
    if world_map.small_goal is not None and world_map.large_goal is not None:
        return  # Both goals already found
    
    print("[GoalSearch] Searching for unmapped goals on Side 2...")
    total_turned = 0.0
    
    while total_turned < 360.0:
        open_loop_turn(ep_chassis, SWEEP_STEP_DEG, dead_reckoner)
        total_turned += SWEEP_STEP_DEG
        time.sleep(SWEEP_SETTLE_S)
        
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        tags = tag_detector.find_tags(gray)
        
        for det in tags:
            tid = int(det.tag_id)
            if tid not in SMALL_GOAL_TAG_IDS and tid not in LARGE_GOAL_TAG_IDS:
                continue
            
            # Register tag location from dead reckoning
            tag_localizer.register_tag_from_robot_pose(det, dead_reckoner.pose)
            wx, wy, _ = tag_localizer.tag_world[tid]
            
            if tid in SMALL_GOAL_TAG_IDS:
                if world_map.small_goal is None:
                    world_map.register_or_update_tag_landmark("small_goal", wx, wy, tid)
                    print(f"  [GoalSearch] Discovered small_goal at ({wx:.2f}, {wy:.2f})")
            elif tid in LARGE_GOAL_TAG_IDS:
                if world_map.large_goal is None:
                    world_map.register_or_update_tag_landmark("large_goal", wx, wy, tid)
                    print(f"  [GoalSearch] Discovered large_goal at ({wx:.2f}, {wy:.2f})")
    
    print("[GoalSearch] Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MISSION LOOP — top-level side-aware state machine
# ─────────────────────────────────────────────────────────────────────────────

def run_mission(
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
) -> int:
    """
    Side-aware delivery loop.

    The robot tracks which side it is on at all times.  Actions are gated
    by the current side:

      SIDE1 : PICK_UP_BRICK | RECHARGE | CROSS_TO_SIDE2
      SIDE2 : DROP_OFF_BRICK | CROSS_TO_SIDE1

    Battery is checked before committing to a brick interaction.  If low,
    the robot first returns to Side-1 and recharges.

    Returns the number of successful deliveries.
    """
    current_side  = Side.SIDE1
    action_stack  = ActionStack()
    deliveries    = 0
    held_brick_class: Optional[int] = None

    while deliveries < max_deliveries and not battery.depleted:
        print(f"\n══ Delivery {deliveries + 1}/{max_deliveries} "
              f"| Battery={battery.level:.0f}% | Side={current_side.name} ══")

        # ═══════════════════════════════════════════════════════════════════
        # SIDE 1 actions
        # ═══════════════════════════════════════════════════════════════════
        if current_side == Side.SIDE1:

            # ── Battery check: recharge if needed before next pick-up ─────
            if battery.needs_recharge(CLASS_SMALL_BRICK):
                print("[Mission] Battery low → recharging on Side-1.")
                execute_recharge(
                    ep_robot, ep_chassis, ep_camera,
                    tag_detector, tag_localizer,
                    dead_reckoner, world_map, battery, show=show,
                )

            # ── Navigate toward loading dock (best estimate) ──────────────
            if world_map.dock_x is not None:
                navigate_to_world_point(
                    ep_chassis, dead_reckoner,
                    world_map.dock_x, world_map.dock_y,
                    stop_dist_m=0.5,
                )
            else:
                # Dock not yet located — do a search spin
                print("[Mission] Dock location unknown — searching ...")
                find_loading_dock(
                    ep_chassis, ep_camera, yolo_model,
                    dead_reckoner, world_map, show=show,
                )

            # ── Identify which brick to pick ──────────────────────────────
            brick_class = choose_brick(ep_camera, yolo_model)
            if brick_class is None:
                print("[Mission] No bricks visible at dock. Skipping.")
                break

            # ── Per-brick battery check ────────────────────────────────────
            if battery.needs_recharge(brick_class):
                print(f"[Mission] Not enough battery for class {brick_class} → recharging.")
                execute_recharge(
                    ep_robot, ep_chassis, ep_camera,
                    tag_detector, tag_localizer,
                    dead_reckoner, world_map, battery, show=show,
                )

            # ── Approach and pick up the brick ────────────────────────────
            move_arm_to_top(ep_robot)
            success = approach_brick(
                ep_robot, ep_camera, ep_chassis, yolo_model,
                dead_reckoner, brick_class, action_stack,
                timeout_s=30.0, show=show,
            )
            if not success:
                print("[Mission] Could not approach brick. Skipping delivery.")
                continue

            pick_up_tower(ep_robot=ep_robot)
            battery.consume(brick_class)
            held_brick_class = brick_class

            if battery.depleted:
                print("[Mission] BATTERY DEPLETED during pick-up! Aborting.")
                break

            # Refine dock position from the current robot location
            world_map.set_dock(dead_reckoner.pose.x, dead_reckoner.pose.y)

            # ── Turn away from dock to face obstacle field ────────────────
            print("[Mission] Turning 180° to face obstacle field...")
            open_loop_turn(ep_chassis, 180.0, dead_reckoner)

            # ── Cross to Side-2 ───────────────────────────────────────────
            current_side = cross_obstacle_field(
                ep_robot, ep_camera, ep_chassis, yolo_model,
                tag_detector, dead_reckoner, world_map,
                current_side=current_side, show=show,
            )

        # ═══════════════════════════════════════════════════════════════════
        # SIDE 2 actions
        # ═══════════════════════════════════════════════════════════════════
        elif current_side == Side.SIDE2:

            if held_brick_class is None:
                # Should not happen — return to Side-1 to pick something up
                print("[Mission] On Side-2 with no brick held. Crossing back.")
                current_side = cross_obstacle_field(
                    ep_robot, ep_camera, ep_chassis, yolo_model,
                    tag_detector, dead_reckoner, world_map,
                    current_side=current_side, show=show,
                )
                continue

            # ── Search for unmapped goals (if blocked during sweep) ────────
            search_for_unmapped_goals(
                ep_chassis, ep_camera, tag_detector, tag_localizer,
                dead_reckoner, world_map, show=show,
            )

            # ── Deliver to the correct goal ───────────────────────────────
            goal = (
                world_map.small_goal
                if held_brick_class == CLASS_SMALL_BRICK
                else world_map.large_goal
            )
            if goal is None:
                print(f"[Mission] CRITICAL ERROR: goal for brick class "
                      f"{held_brick_class} not found even after search! This should never happen.")
                print("[Mission] Aborting mission.")
                break
            
            deliver_brick(
                ep_robot, ep_camera, ep_chassis,
                tag_detector, dead_reckoner, goal, show=show,
            )

            deliveries += 1
            held_brick_class = None
            print(f"[Mission] Delivery {deliveries} complete! "
                  f"Battery={battery.level:.0f}%")

            # ── Cross back to Side-1 ──────────────────────────────────────
            current_side = cross_obstacle_field(
                ep_robot, ep_camera, ep_chassis, yolo_model,
                tag_detector, dead_reckoner, world_map,
                current_side=current_side, show=show,
            )

    print(f"\n[Mission] Complete. {deliveries} brick(s) delivered. "
          f"Battery={battery.level:.0f}%")
    return deliveries


# ─────────────────────────────────────────────────────────────────────────────
# MAP VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_map(world_map: WorldMap, robot_pose: Optional[Pose2D] = None) -> None:
    """
    Generate a matplotlib bird's-eye-view map per the project report spec:
      - xy grid lines every 10 cm.
      - Red circles for obstacles.
      - Blue  triangle for small goal.
      - Green triangle for large goal.
      - Yellow square for loading dock.
      - Black square for recharge station.
    Saves to arena_map.png.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Map] matplotlib not available. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ARENA_W_M)
    ax.set_ylim(0, ARENA_H_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Project 3 – Arena Map (Bird's-Eye View)")

    # Grid every 10 cm
    for v in np.arange(0, ARENA_W_M + 0.01, 0.10):
        ax.axvline(v, color="lightgray", linewidth=0.3)
    for v in np.arange(0, ARENA_H_M + 0.01, 0.10):
        ax.axhline(v, color="lightgray", linewidth=0.3)

    # Arena perimeter
    ax.plot([0, ARENA_W_M, ARENA_W_M, 0, 0],
            [0, 0, ARENA_H_M, ARENA_H_M, 0], "k-", linewidth=2, label="Boundary")

    # Obstacles: red circles
    for obs in world_map.obstacles:
        ax.add_patch(plt.Circle((obs.x, obs.y), 0.15, color="red", alpha=0.6))
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
    if world_map.dock_x is not None:
        ax.add_patch(patches.Rectangle(
            (world_map.dock_x - 0.125, world_map.dock_y - 0.125),
            0.25, 0.25,
            linewidth=1, edgecolor="goldenrod", facecolor="yellow", alpha=0.8,
            label="Loading dock",
        ))

    # Recharge station: black square (20×20 cm)
    if world_map.recharge:
        ax.add_patch(patches.Rectangle(
            (world_map.recharge.x - 0.10, world_map.recharge.y - 0.10),
            0.20, 0.20,
            linewidth=1, edgecolor="black", facecolor="black", alpha=0.85,
            label="Recharge station",
        ))

    # Robot pose arrow
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
    p.add_argument("--model-path",      default=MODEL_PATH)
    p.add_argument("--robot-ip",        default=ROBOT_IP)
    p.add_argument("--sn",              default=ROBOT_SN)
    p.add_argument("--conn-type",       default="sta", choices=["sta", "ap"])
    p.add_argument("--resolution",      default="360p", choices=["360p", "720p"])
    p.add_argument("--max-deliveries",  type=int, default=5)
    p.add_argument("--skip-sweep",      action="store_true",
                   help="Skip the startup sweep (debugging).")
    p.add_argument("--show",            action="store_true",
                   help="Show OpenCV debug windows.")
    p.add_argument("--map-only",        action="store_true",
                   help="Run the sweep, show map, then exit.")
    p.add_argument("--start-corner",    choices=["top-left", "top-right"],
                   default="top-left",
                   help="Manual override for the starting corner.")
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
    ep_robot   = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=args.sn)
    ep_camera  = ep_robot.camera
    ep_chassis = ep_robot.chassis

    res = (rm_camera.STREAM_720P if args.resolution == "720p"
           else rm_camera.STREAM_360P)
    ep_camera.start_video_stream(display=False, resolution=res)

    # ── Initialise subsystems ──────────────────────────────────────────────
    tag_detector  = AprilTagDetector()
    tag_localizer = TagLocalizer()
    # Seed dead reckoning from the requested corner; a startup sweep can
    # correct the frame if automatic inference is enabled.
    seed_corner = args.start_corner
    start_x = SAFE_BOUNDARY_MARGIN_M if seed_corner == "top-left" else ARENA_W_M - SAFE_BOUNDARY_MARGIN_M
    start_y = SAFE_BOUNDARY_MARGIN_M
    dead_reckoner = DeadReckoner(Pose2D(x=start_x, y=start_y, yaw=0.0))
    world_map     = WorldMap()
    battery       = BatteryManager(start_pct=BATTERY_START_PCT)

    try:
        move_arm_to_default(ep_robot)
        ep_robot.gripper.open()

        # ── Phase 1: Startup 360° sweep ───────────────────────────────────
        if not args.skip_sweep:
            inferred_corner = startup_sweep(
                ep_robot, ep_camera, ep_chassis,
                yolo_model, tag_detector, tag_localizer,
                dead_reckoner, world_map, show=args.show,
            )
            if args.start_corner == "top-left" and inferred_corner == "top-right":
                apply_start_corner_correction(
                    "top-left",
                    inferred_corner,
                    dead_reckoner,
                    world_map,
                )
            elif args.start_corner == "top-right" and inferred_corner == "top-left":
                apply_start_corner_correction(
                    "top-right",
                    inferred_corner,
                    dead_reckoner,
                    world_map,
                )

        # ── Phase 2: Locate loading dock from brick cluster ───────────────
        if world_map.dock_x is None:
            find_loading_dock(
                ep_chassis, ep_camera, yolo_model,
                dead_reckoner, world_map, show=args.show,
            )

        if args.map_only:
            visualize_map(world_map, dead_reckoner.pose)
            return

        # ── Phase 3: Main delivery mission ────────────────────────────────
        run_mission(
            ep_robot, ep_camera, ep_chassis,
            yolo_model, tag_detector, tag_localizer,
            dead_reckoner, world_map, battery,
            max_deliveries=args.max_deliveries,
            show=args.show,
        )

        # ── Phase 4: Final map visualisation ─────────────────────────────
        visualize_map(world_map, dead_reckoner.pose)

    except (KeyboardInterrupt, TimeoutError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
    finally:
        try:
            ep_chassis.move(x=0.0, y=0.0, z=0.0, xy_speed=0.7, z_speed=45).wait_for_completed()
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