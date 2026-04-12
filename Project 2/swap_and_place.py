#!/usr/bin/env python3
"""Swap-and-place workflow using heading-aware geometric pose tracking.

Core behavior:
1) Detect two towers.
2) Go to tower 1 by visual alignment (rotate to center, then drive forward to top-y target), pick it up.
3) Turn left 90 degrees, move to a temporary drop point, place tower 1, and return home pose.
4) Go to tower 2 in the same way, pick it up, and place at tower 1 original pose.
5) Return home, reacquire tower 1 with exclusion logic so tower 2 is ignored, pick tower 1,
   and place it at tower 2 original pose.
"""

import argparse
import math
import time
from dataclasses import dataclass
from queue import Empty
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

from robomaster import camera

from tower_utils import (
    DEFAULT_ALIGN_TOP_TOL_PX,
    DEFAULT_MODEL_PATH,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
    DEFAULT_STOP_METRIC,
    DEFAULT_TARGET_TOP_Y_RATIO,
    connect_robot,
    get_detections,
    go_to_tower,
    pick_up_tower,
    place_down_tower,
)

FX_PX = 314.0
CX_PX = 320.0
DEFAULT_TURN_SPEED_DPS = 45.0
DEFAULT_TOWER2_EXCLUSION_CX_TOL_PX = 90.0


@dataclass
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    cls: int


@dataclass
class Pose2D:
    x_m: float
    y_m: float
    yaw_deg: float


class PoseTracker:
    """Track commanded chassis motion relative to home pose."""

    def __init__(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0
        self.heading_deg = 0.0

    def integrate_body_motion(self, forward_m: float, lateral_m: float) -> None:
        heading_rad = math.radians(self.heading_deg)
        world_dx = forward_m * math.cos(heading_rad) - lateral_m * math.sin(heading_rad)
        world_dy = forward_m * math.sin(heading_rad) + lateral_m * math.cos(heading_rad)
        self.x_m += world_dx
        self.y_m += world_dy

    def integrate_turn(self, delta_deg: float) -> None:
        self.heading_deg = normalize_heading_deg(self.heading_deg + delta_deg)

    def current_pose(self) -> Pose2D:
        return Pose2D(x_m=self.x_m, y_m=self.y_m, yaw_deg=self.heading_deg)

    def reset(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0
        self.heading_deg = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap two LEGO towers with YOLO + RoboMaster")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to fine-tuned YOLO weights.")
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"], help="Robot connection mode.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Robot IP address for STA mode.")
    parser.add_argument("--sn", default=DEFAULT_ROBOT_SN, help="Robot serial number.")
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"], help="Camera stream resolution.")

    parser.add_argument("--detect-conf", type=float, default=0.45, help="YOLO confidence threshold.")
    parser.add_argument("--target-class", type=int, default=None, help="Optional class id for towers.")

    parser.add_argument(
        "--stop-metric",
        default=DEFAULT_STOP_METRIC,
        choices=["top_y"],
        help="Forward stop metric for visual servo: bbox top y-position.",
    )
    parser.add_argument(
        "--target-top-y-ratio",
        type=float,
        default=DEFAULT_TARGET_TOP_Y_RATIO,
        help="Target top-of-bbox y as a fraction of frame height; tune manually on robot.",
    )
    parser.add_argument("--align-center-tol-px", type=float, default=24.0, help="Horizontal center tolerance in px.")
    parser.add_argument("--align-top-tol-px", type=float, default=DEFAULT_ALIGN_TOP_TOL_PX, help="Top-y tolerance in px.")

    parser.add_argument("--k-forward", type=float, default=0.0028, help="P gain from top-y error to forward speed.")
    parser.add_argument("--k-yaw", type=float, default=0.12, help="P gain from x error to yaw rate (deg/s per px).")
    parser.add_argument("--max-v", type=float, default=0.16, help="Max forward speed for visual approach (m/s).")
    parser.add_argument("--max-yaw-dps", type=float, default=45.0, help="Max yaw rate for visual approach (deg/s).")
    parser.add_argument("--servo-step-s", type=float, default=0.12, help="Duration of each drive_speed command.")

    parser.add_argument("--turn-speed-dps", type=float, default=DEFAULT_TURN_SPEED_DPS, help="Turn speed for geometry moves.")
    parser.add_argument("--xy-speed", type=float, default=0.22, help="chassis.move speed for coarse moves.")
    parser.add_argument("--temp-forward-m", type=float, default=0.20, help="Forward move after 90-left for temporary drop.")

    parser.add_argument("--scan-side-m", type=float, default=0.28, help="Half-width for lateral search sweeps (m).")
    parser.add_argument("--scan-forward-m", type=float, default=0.10, help="Forward step per sweep row (m).")
    parser.add_argument("--scan-rows", type=int, default=5, help="Number of sweep rows for reacquisition.")
    parser.add_argument(
        "--tower2-exclusion-cx-tol-px",
        type=float,
        default=DEFAULT_TOWER2_EXCLUSION_CX_TOL_PX,
        help="Pixel tolerance around expected tower2 image x used to reject wrong tower during reacquire.",
    )

    parser.add_argument("--show", action="store_true", help="Show live debug view.")
    return parser.parse_args()


def resolve_resolution(name: str):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_heading_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def move_relative(ep_chassis, tracker: PoseTracker, forward_m: float, lateral_m: float, xy_speed: float) -> None:
    if abs(forward_m) < 1e-4 and abs(lateral_m) < 1e-4:
        return
    ep_chassis.move(x=forward_m, y=lateral_m, z=0, xy_speed=xy_speed).wait_for_completed()
    tracker.integrate_body_motion(forward_m, lateral_m)


def turn_relative(ep_chassis, tracker: PoseTracker, delta_deg: float, turn_speed_dps: float) -> None:
    if abs(delta_deg) < 1e-3:
        return
    speed = abs(turn_speed_dps)
    if speed <= 1e-6:
        raise ValueError("turn_speed_dps must be positive.")

    turn_time_s = abs(delta_deg) / speed
    turn_rate = speed if delta_deg > 0 else -speed
    ep_chassis.drive_speed(x=0.0, y=0.0, z=turn_rate, timeout=turn_time_s)
    tracker.integrate_turn(delta_deg)


def move_to_pose(
    ep_chassis,
    tracker: PoseTracker,
    target_pose: Pose2D,
    xy_speed: float,
    turn_speed_dps: float,
    match_final_yaw: bool = True,
) -> None:
    delta_x = target_pose.x_m - tracker.x_m
    delta_y = target_pose.y_m - tracker.y_m
    distance_m = math.hypot(delta_x, delta_y)

    if distance_m > 1e-4:
        desired_heading_deg = math.degrees(math.atan2(delta_y, delta_x))
        turn_relative(ep_chassis, tracker, normalize_heading_deg(desired_heading_deg - tracker.heading_deg), turn_speed_dps)
        move_relative(ep_chassis, tracker, distance_m, 0.0, xy_speed)

    if match_final_yaw:
        turn_relative(ep_chassis, tracker, normalize_heading_deg(target_pose.yaw_deg - tracker.heading_deg), turn_speed_dps)


def return_home(ep_chassis, tracker: PoseTracker, xy_speed: float, turn_speed_dps: float) -> None:
    move_to_pose(
        ep_chassis=ep_chassis,
        tracker=tracker,
        target_pose=Pose2D(x_m=0.0, y_m=0.0, yaw_deg=0.0),
        xy_speed=xy_speed,
        turn_speed_dps=turn_speed_dps,
        match_final_yaw=True,
    )
    tracker.reset()


def detect_stable_two_towers(
    ep_camera,
    model: YOLO,
    conf_thresh: float,
    target_class: Optional[int],
    required_stable_frames: int = 8,
    timeout_s: float = 25.0,
    show: bool = False,
) -> Tuple[Detection, Detection]:
    t0 = time.time()
    stable = 0
    latest_pair: Optional[Tuple[Detection, Detection]] = None

    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out waiting for two stable tower detections.")

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        dets = get_detections(model, frame, conf_thresh, target_class)
        dets = sorted(dets, key=lambda d: d.conf, reverse=True)

        if len(dets) >= 2:
            left_right = sorted(dets[:2], key=lambda d: d.cx)
            latest_pair = (
                Detection(**left_right[0].__dict__),
                Detection(**left_right[1].__dict__),
            )
            stable += 1
        else:
            stable = 0

        if show:
            dbg = frame.copy()
            for d in dets:
                x1 = int(d.cx - d.w / 2)
                y1 = int(d.cy - d.h / 2)
                x2 = int(d.cx + d.w / 2)
                y2 = int(d.cy + d.h / 2)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                dbg,
                f"Need 2 towers: stable={stable}/{required_stable_frames}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow("swap_and_place", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if stable >= required_stable_frames and latest_pair is not None:
            return latest_pair


def expected_forbidden_cx(tracker: PoseTracker, forbidden_pose: Pose2D) -> Optional[float]:
    dx = forbidden_pose.x_m - tracker.x_m
    dy = forbidden_pose.y_m - tracker.y_m
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    world_bearing_deg = math.degrees(math.atan2(dy, dx))
    rel_bearing_deg = normalize_heading_deg(world_bearing_deg - tracker.heading_deg)

    if abs(rel_bearing_deg) >= 80.0:
        return None

    return CX_PX + FX_PX * math.tan(math.radians(rel_bearing_deg))


def choose_detection_excluding_forbidden(
    detections: List,
    tracker: PoseTracker,
    forbidden_pose: Optional[Pose2D],
    exclusion_tol_px: float,
) -> Optional[object]:
    if not detections:
        return None

    expected_cx = None
    if forbidden_pose is not None:
        expected_cx = expected_forbidden_cx(tracker, forbidden_pose)

    valid = []
    for det in detections:
        if expected_cx is not None and abs(det.cx - expected_cx) <= exclusion_tol_px:
            continue
        valid.append(det)

    if not valid:
        return None

    return max(valid, key=lambda det: det.conf)


def reacquire_tower_with_sweep(
    ep_chassis,
    ep_camera,
    model: YOLO,
    tracker: PoseTracker,
    conf_thresh: float,
    target_class: Optional[int],
    forbidden_pose: Optional[Pose2D],
    exclusion_tol_px: float,
    scan_side_m: float,
    scan_forward_m: float,
    scan_rows: int,
    xy_speed: float,
    show: bool,
) -> str:
    """Reacquire a tower while excluding tower2 by expected image bearing.

    Returns:
        selection hint for go_to_tower: leftmost or rightmost.
    """
    direction = 1.0

    for row in range(scan_rows):
        for _ in range(2):
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.4)
            except Empty:
                frame = None

            if frame is not None:
                dets = get_detections(model, frame, conf_thresh, target_class)
                candidate = choose_detection_excluding_forbidden(
                    detections=dets,
                    tracker=tracker,
                    forbidden_pose=forbidden_pose,
                    exclusion_tol_px=exclusion_tol_px,
                )
                if candidate is not None:
                    frame_center_x = frame.shape[1] / 2.0
                    return "leftmost" if candidate.cx < frame_center_x else "rightmost"

                if show:
                    dbg = frame.copy()
                    cv2.putText(
                        dbg,
                        f"reacquire row {row + 1}/{scan_rows}",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cx_forbidden = expected_forbidden_cx(tracker, forbidden_pose) if forbidden_pose is not None else None
                    if cx_forbidden is not None:
                        cv2.line(
                            dbg,
                            (int(cx_forbidden), 0),
                            (int(cx_forbidden), dbg.shape[0] - 1),
                            (255, 0, 0),
                            1,
                        )
                    cv2.imshow("swap_and_place", dbg)
                    cv2.waitKey(1)

            if _ == 0:
                side_target = direction * scan_side_m
                move_relative(ep_chassis, tracker, 0.0, side_target - tracker.y_m, xy_speed)

        if row < scan_rows - 1:
            move_relative(ep_chassis, tracker, scan_forward_m, 0.0, xy_speed)

        direction *= -1.0

    raise RuntimeError("Could not reacquire moved tower 1 during search sweeps.")


def main() -> None:
    args = parse_args()
    model = YOLO(args.model_path)

    ep_robot = connect_robot(conn_type=args.conn_type, robot_ip=args.robot_ip, sn=args.sn)
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    tracker = PoseTracker()

    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    try:
        print("[1/8] Detecting two initial towers...")
        detect_stable_two_towers(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            show=args.show,
        )

        print("[2/8] Going to tower 1 and picking it up...")
        go_to_tower(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            target_class=args.target_class,
            conf_thresh=args.detect_conf,
            stop_metric=args.stop_metric,
            target_top_y_ratio=args.target_top_y_ratio,
            center_tol_px=args.align_center_tol_px,
            top_y_tol_px=args.align_top_tol_px,
            k_forward=args.k_forward,
            k_lateral=0.0,
            k_yaw=args.k_yaw,
            max_v=args.max_v,
            max_yaw_dps=args.max_yaw_dps,
            step_s=args.servo_step_s,
            pose_tracker=tracker,
            selection_mode="leftmost",
            show=args.show,
        )
        tower1_slot = tracker.current_pose()
        pick_up_tower(ep_robot=ep_robot)

        print("[3/8] Turn left 90, place tower 1 temporarily, then go home...")
        turn_relative(ep_chassis, tracker, 90.0, args.turn_speed_dps)
        move_relative(ep_chassis, tracker, args.temp_forward_m, 0.0, args.xy_speed)
        place_down_tower(ep_robot=ep_robot)
        return_home(ep_chassis, tracker, args.xy_speed, args.turn_speed_dps)

        print("[4/8] Going to tower 2 and picking it up...")
        go_to_tower(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            target_class=args.target_class,
            conf_thresh=args.detect_conf,
            stop_metric=args.stop_metric,
            target_top_y_ratio=args.target_top_y_ratio,
            center_tol_px=args.align_center_tol_px,
            top_y_tol_px=args.align_top_tol_px,
            k_forward=args.k_forward,
            k_lateral=0.0,
            k_yaw=args.k_yaw,
            max_v=args.max_v,
            max_yaw_dps=args.max_yaw_dps,
            step_s=args.servo_step_s,
            pose_tracker=tracker,
            selection_mode="rightmost",
            show=args.show,
        )
        tower2_slot = tracker.current_pose()
        pick_up_tower(ep_robot=ep_robot)

        print("[5/8] Placing tower 2 at tower 1 original slot...")
        move_to_pose(
            ep_chassis=ep_chassis,
            tracker=tracker,
            target_pose=tower1_slot,
            xy_speed=args.xy_speed,
            turn_speed_dps=args.turn_speed_dps,
            match_final_yaw=True,
        )
        place_down_tower(ep_robot=ep_robot)
        tower2_current_world = Pose2D(x_m=tower1_slot.x_m, y_m=tower1_slot.y_m, yaw_deg=tower1_slot.yaw_deg)

        print("[6/8] Returning home and reacquiring moved tower 1...")
        return_home(ep_chassis, tracker, args.xy_speed, args.turn_speed_dps)
        reacquire_hint = reacquire_tower_with_sweep(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            model=model,
            tracker=tracker,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            forbidden_pose=tower2_current_world,
            exclusion_tol_px=args.tower2_exclusion_cx_tol_px,
            scan_side_m=args.scan_side_m,
            scan_forward_m=args.scan_forward_m,
            scan_rows=args.scan_rows,
            xy_speed=args.xy_speed,
            show=args.show,
        )

        print("[7/8] Going to reacquired tower 1 and picking it up...")
        go_to_tower(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            target_class=args.target_class,
            conf_thresh=args.detect_conf,
            stop_metric=args.stop_metric,
            target_top_y_ratio=args.target_top_y_ratio,
            center_tol_px=args.align_center_tol_px,
            top_y_tol_px=args.align_top_tol_px,
            k_forward=args.k_forward,
            k_lateral=0.0,
            k_yaw=args.k_yaw,
            max_v=args.max_v,
            max_yaw_dps=args.max_yaw_dps,
            step_s=args.servo_step_s,
            pose_tracker=tracker,
            selection_mode=reacquire_hint,
            show=args.show,
        )
        pick_up_tower(ep_robot=ep_robot)

        print("[8/8] Placing tower 1 at tower 2 original slot...")
        move_to_pose(
            ep_chassis=ep_chassis,
            tracker=tracker,
            target_pose=tower2_slot,
            xy_speed=args.xy_speed,
            turn_speed_dps=args.turn_speed_dps,
            match_final_yaw=True,
        )
        place_down_tower(ep_robot=ep_robot)

        print("Swap-and-place sequence complete.")
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


if __name__ == "__main__":
    main()
