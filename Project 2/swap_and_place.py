#!/usr/bin/env python3
"""
Project 2: Swap-and-place state machine using YOLO detections and RoboMaster control.

High-level sequence:
1) Find two towers and record their initial slot geometry.
2) Pick tower 1 and place it in a temporary location.
3) Pick tower 2 and place it at tower 1's original slot.
4) Re-find tower 1 even if a human moved it, then place it at tower 2's original slot.

This script intentionally reuses existing project helpers in tower_utils.py for arm/gripper
pickup/place operations and uses chassis motion commands already used in project1_nav.py.
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

from robomaster import camera

from tower_utils import DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN, connect_robot, pick_up_tower, place_down_tower


# Camera intrinsics used in your nav/perception codebase.
FX_PX = 314.0
CX_PX = 320.0
DEFAULT_TOWER2_EXCLUSION_RADIUS_M = 0.30


@dataclass
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    cls: int


@dataclass
class RelativeTarget:
    forward_m: float
    lateral_m: float


@dataclass
class SearchCandidate:
    detection: Detection
    world_pose: RelativeTarget
    exclusion_distance_m: float


class PoseTracker:
    """Track commanded chassis x/y offsets relative to the home pose."""

    def __init__(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0

    def integrate(self, dx: float, dy: float) -> None:
        self.x_m += dx
        self.y_m += dy

    def reset(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap two LEGO towers with YOLO + RoboMaster")
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().parents[1] / "runs" / "detect" / "train5" / "weights" / "best.pt"),
        help="Path to fine-tuned YOLO weights.",
    )
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"], help="Robot connection mode.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Robot IP address for STA mode.")
    parser.add_argument("--sn", default=DEFAULT_ROBOT_SN, help="Robot serial number.")
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"], help="Camera stream resolution.")

    parser.add_argument("--tower-height-m", type=float, default=0.10, help="Approx tower height in meters.")
    parser.add_argument("--detect-conf", type=float, default=0.45, help="YOLO confidence threshold.")
    parser.add_argument("--target-class", type=int, default=None, help="Optional class id for towers.")

    parser.add_argument(
        "--pickup-standoff-m",
        type=float,
        default=0.32,
        help="Desired chassis standoff from target before arm pickup.",
    )
    parser.add_argument(
        "--align-desired-h-px",
        type=float,
        default=170.0,
        help="Desired bbox height in pixels when at pickup standoff.",
    )
    parser.add_argument("--align-center-tol-px", type=float, default=24.0, help="Horizontal center tolerance in px.")
    parser.add_argument("--align-height-tol-px", type=float, default=16.0, help="BBox height tolerance in px.")

    parser.add_argument("--k-forward", type=float, default=0.0028, help="P gain from height error to forward speed.")
    parser.add_argument("--k-lateral", type=float, default=0.0038, help="P gain from x error to lateral speed.")
    parser.add_argument(
        "--lateral-sign",
        type=float,
        default=-1.0,
        help="Set to +1 or -1 depending on your chassis lateral sign convention.",
    )
    parser.add_argument("--max-v", type=float, default=0.16, help="Max visual servo translation speed (m/s).")
    parser.add_argument("--servo-step-s", type=float, default=0.12, help="Duration of each drive_speed command.")

    parser.add_argument("--temp-back-m", type=float, default=0.20, help="Temporary placement move backward from pickup.")
    parser.add_argument("--temp-side-m", type=float, default=0.22, help="Temporary placement move lateral from pickup.")

    parser.add_argument("--scan-side-m", type=float, default=0.28, help="Half-width for lateral search sweeps (m).")
    parser.add_argument("--scan-forward-m", type=float, default=0.10, help="Forward step per sweep row (m).")
    parser.add_argument("--scan-rows", type=int, default=5, help="Number of sweep rows for reacquisition.")
    parser.add_argument("--xy-speed", type=float, default=0.22, help="chassis.move speed for coarse moves.")
    parser.add_argument(
        "--tower2-exclusion-radius-m",
        type=float,
        default=DEFAULT_TOWER2_EXCLUSION_RADIUS_M,
        help="Radius around tower 2's placed position to ignore when reacquiring tower 1.",
    )

    parser.add_argument("--show", action="store_true", help="Show live debug view.")
    return parser.parse_args()


def resolve_resolution(name: str):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def get_detections(model: YOLO, frame, conf_thresh: float, target_class: Optional[int]) -> List[Detection]:
    result = model.predict(source=frame, show=False, conf=conf_thresh, verbose=False)[0]
    out: List[Detection] = []

    if result.boxes is None:
        return out

    for b in result.boxes:
        xyxy = b.xyxy.cpu().numpy().flatten()
        conf = float(b.conf.item())
        cls = int(b.cls.item())
        if target_class is not None and cls != target_class:
            continue

        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        out.append(Detection(cx=0.5 * (x1 + x2), cy=0.5 * (y1 + y2), w=w, h=h, conf=conf, cls=cls))

    return out


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
            latest_pair = (left_right[0], left_right[1])
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


def estimate_relative_target(det: Detection, tower_height_m: float) -> RelativeTarget:
    range_forward = (tower_height_m * FX_PX) / max(det.h, 1.0)
    lateral = ((det.cx - CX_PX) * range_forward) / FX_PX
    return RelativeTarget(forward_m=range_forward, lateral_m=lateral)


def estimate_world_target(det: Detection, tracker: PoseTracker, tower_height_m: float) -> RelativeTarget:
    rel = estimate_relative_target(det, tower_height_m)
    return RelativeTarget(
        forward_m=tracker.x_m + rel.forward_m,
        lateral_m=tracker.y_m + rel.lateral_m,
    )


def distance_m(a: RelativeTarget, b: RelativeTarget) -> float:
    return math.hypot(a.forward_m - b.forward_m, a.lateral_m - b.lateral_m)


def select_detection_outside_exclusion(
    detections: List[Detection],
    tracker: PoseTracker,
    tower_height_m: float,
    forbidden_center_world: Optional[RelativeTarget],
    forbidden_radius_m: float,
    expected_lateral_m: Optional[float] = None,
) -> Optional[SearchCandidate]:
    valid: List[SearchCandidate] = []

    for det in detections:
        world_pose = estimate_world_target(det, tracker, tower_height_m)
        if forbidden_center_world is not None:
            exclusion_distance_m = distance_m(world_pose, forbidden_center_world)
            if exclusion_distance_m <= forbidden_radius_m:
                continue
        else:
            exclusion_distance_m = float("inf")

        valid.append(
            SearchCandidate(
                detection=det,
                world_pose=world_pose,
                exclusion_distance_m=exclusion_distance_m,
            )
        )

    if not valid:
        return None

    if expected_lateral_m is None:
        return max(valid, key=lambda cand: (cand.detection.conf, cand.exclusion_distance_m))

    scored = []
    for cand in valid:
        rel = estimate_relative_target(cand.detection, tower_height_m)
        score = abs(rel.lateral_m - expected_lateral_m)
        scored.append((score, -cand.detection.conf, -cand.exclusion_distance_m, cand))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][3]


def choose_detection(
    detections: List[Detection],
    expected_lateral_m: Optional[float],
    tower_height_m: float,
    tracker: Optional[PoseTracker] = None,
    forbidden_center_world: Optional[RelativeTarget] = None,
    forbidden_radius_m: float = 0.0,
) -> Optional[Detection]:
    if not detections:
        return None

    if tracker is not None and forbidden_center_world is not None:
        candidate = select_detection_outside_exclusion(
            detections=detections,
            tracker=tracker,
            tower_height_m=tower_height_m,
            forbidden_center_world=forbidden_center_world,
            forbidden_radius_m=forbidden_radius_m,
            expected_lateral_m=expected_lateral_m,
        )
        return None if candidate is None else candidate.detection

    if expected_lateral_m is None:
        return max(detections, key=lambda d: d.conf)

    scored = []
    for d in detections:
        rel = estimate_relative_target(d, tower_height_m)
        score = abs(rel.lateral_m - expected_lateral_m)
        scored.append((score, -d.conf, d))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def move_relative(ep_chassis, tracker: PoseTracker, dx: float, dy: float, xy_speed: float) -> None:
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return
    ep_chassis.move(x=dx, y=dy, z=0, xy_speed=xy_speed).wait_for_completed()
    tracker.integrate(dx, dy)


def return_home(ep_chassis, tracker: PoseTracker, xy_speed: float) -> None:
    move_relative(ep_chassis, tracker, -tracker.x_m, -tracker.y_m, xy_speed)
    tracker.reset()


def align_and_approach_target(
    ep_chassis,
    ep_camera,
    model: YOLO,
    tracker: PoseTracker,
    conf_thresh: float,
    target_class: Optional[int],
    expected_lateral_m: Optional[float],
    tower_height_m: float,
    forbidden_center_world: Optional[RelativeTarget],
    forbidden_radius_m: float,
    desired_h_px: float,
    center_tol_px: float,
    height_tol_px: float,
    k_forward: float,
    k_lateral: float,
    lateral_sign: float,
    max_v: float,
    step_s: float,
    show: bool,
    timeout_s: float = 20.0,
) -> Detection:
    t0 = time.time()
    stable = 0
    selected: Optional[Detection] = None

    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out while aligning to a tower.")

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        if frame is None:
            continue

        detections = get_detections(model, frame, conf_thresh, target_class)
        selected = choose_detection(
            detections,
            expected_lateral_m,
            tower_height_m,
            tracker=tracker,
            forbidden_center_world=forbidden_center_world,
            forbidden_radius_m=forbidden_radius_m,
        )
        if selected is None:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=step_s)
            continue

        err_x_px = selected.cx - CX_PX
        err_h_px = desired_h_px - selected.h

        v_forward = clamp(k_forward * err_h_px, -max_v, max_v)
        v_lateral = clamp(lateral_sign * k_lateral * err_x_px, -max_v, max_v)

        ep_chassis.drive_speed(x=v_forward, y=v_lateral, z=0.0, timeout=step_s)

        if abs(err_x_px) <= center_tol_px and abs(err_h_px) <= height_tol_px:
            stable += 1
        else:
            stable = 0

        if show:
            dbg = frame.copy()
            x1 = int(selected.cx - selected.w / 2)
            y1 = int(selected.cy - selected.h / 2)
            x2 = int(selected.cx + selected.w / 2)
            y2 = int(selected.cy + selected.h / 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(dbg, (int(CX_PX), 0), (int(CX_PX), dbg.shape[0] - 1), (0, 255, 255), 1)
            cv2.putText(
                dbg,
                f"err_x={err_x_px:+.1f}px err_h={err_h_px:+.1f}px stable={stable}/4",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow("swap_and_place", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if stable >= 4:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
            return selected


def reacquire_any_tower_with_sweep(
    ep_chassis,
    ep_camera,
    model: YOLO,
    tracker: PoseTracker,
    conf_thresh: float,
    target_class: Optional[int],
    tower_height_m: float,
    forbidden_center_world: Optional[RelativeTarget],
    forbidden_radius_m: float,
    scan_side_m: float,
    scan_forward_m: float,
    scan_rows: int,
    xy_speed: float,
    show: bool,
) -> bool:
    """Sweep translationally to find a moved tower without assuming a fixed temporary spot."""
    direction = 1.0

    for row in range(scan_rows):
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.4)
        except Empty:
            frame = None

        if frame is not None:
            dets = get_detections(model, frame, conf_thresh, target_class)
            candidate = select_detection_outside_exclusion(
                detections=dets,
                tracker=tracker,
                tower_height_m=tower_height_m,
                forbidden_center_world=forbidden_center_world,
                forbidden_radius_m=forbidden_radius_m,
                expected_lateral_m=None,
            )
            if candidate is not None:
                return True
            if show:
                dbg = frame.copy()
                cv2.putText(
                    dbg,
                    f"reacquire sweep row {row + 1}/{scan_rows}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("swap_and_place", dbg)
                cv2.waitKey(1)

        side_target = direction * scan_side_m
        side_move = side_target - tracker.y_m
        move_relative(ep_chassis, tracker, 0.0, side_move, xy_speed)

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.4)
        except Empty:
            frame = None
        if frame is not None:
            dets = get_detections(model, frame, conf_thresh, target_class)
            candidate = select_detection_outside_exclusion(
                detections=dets,
                tracker=tracker,
                tower_height_m=tower_height_m,
                forbidden_center_world=forbidden_center_world,
                forbidden_radius_m=forbidden_radius_m,
                expected_lateral_m=None,
            )
            if candidate is not None:
                return True

        if row < scan_rows - 1:
            move_relative(ep_chassis, tracker, scan_forward_m, 0.0, xy_speed)

        direction *= -1.0

    return False


def main() -> None:
    args = parse_args()
    model = YOLO(args.model_path)

    ep_robot = connect_robot(
        conn_type=args.conn_type,
        robot_ip=args.robot_ip,
        sn=args.sn,
    )

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    tracker = PoseTracker()

    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    try:
        print("[1/8] Detecting two initial towers...")
        tower1_det, tower2_det = detect_stable_two_towers(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            show=args.show,
        )

        tower1_rel = estimate_relative_target(tower1_det, args.tower_height_m)
        tower2_rel = estimate_relative_target(tower2_det, args.tower_height_m)

        tower1_slot = RelativeTarget(
            forward_m=max(0.05, tower1_rel.forward_m - args.pickup_standoff_m),
            lateral_m=tower1_rel.lateral_m,
        )
        tower2_slot = RelativeTarget(
            forward_m=max(0.05, tower2_rel.forward_m - args.pickup_standoff_m),
            lateral_m=tower2_rel.lateral_m,
        )

        print("[2/8] Going to tower 1 and picking it up...")
        move_relative(ep_chassis, tracker, tower1_slot.forward_m, tower1_slot.lateral_m, args.xy_speed)
        align_and_approach_target(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            model=model,
            tracker=tracker,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            expected_lateral_m=tower1_slot.lateral_m,
            tower_height_m=args.tower_height_m,
            forbidden_center_world=None,
            forbidden_radius_m=0.0,
            desired_h_px=args.align_desired_h_px,
            center_tol_px=args.align_center_tol_px,
            height_tol_px=args.align_height_tol_px,
            k_forward=args.k_forward,
            k_lateral=args.k_lateral,
            lateral_sign=args.lateral_sign,
            max_v=args.max_v,
            step_s=args.servo_step_s,
            show=args.show,
        )
        pick_up_tower(ep_robot=ep_robot)

        print("[3/8] Placing tower 1 in temporary location...")
        move_relative(ep_chassis, tracker, -args.temp_back_m, args.temp_side_m, args.xy_speed)
        place_down_tower(ep_robot=ep_robot)

        print("[4/8] Returning home and fetching tower 2...")
        return_home(ep_chassis, tracker, args.xy_speed)
        move_relative(ep_chassis, tracker, tower2_slot.forward_m, tower2_slot.lateral_m, args.xy_speed)
        align_and_approach_target(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            model=model,
            tracker=tracker,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            expected_lateral_m=tower2_slot.lateral_m,
            tower_height_m=args.tower_height_m,
            forbidden_center_world=None,
            forbidden_radius_m=0.0,
            desired_h_px=args.align_desired_h_px,
            center_tol_px=args.align_center_tol_px,
            height_tol_px=args.align_height_tol_px,
            k_forward=args.k_forward,
            k_lateral=args.k_lateral,
            lateral_sign=args.lateral_sign,
            max_v=args.max_v,
            step_s=args.servo_step_s,
            show=args.show,
        )
        pick_up_tower(ep_robot=ep_robot)

        print("[5/8] Placing tower 2 at tower 1 original slot...")
        return_home(ep_chassis, tracker, args.xy_speed)
        move_relative(ep_chassis, tracker, tower1_slot.forward_m, tower1_slot.lateral_m, args.xy_speed)
        place_down_tower(ep_robot=ep_robot)

        tower2_current_world = RelativeTarget(
            forward_m=tower1_slot.forward_m,
            lateral_m=tower1_slot.lateral_m,
        )

        print("[6/8] Reacquiring moved tower 1 (not assuming it stayed in temp spot)...")
        return_home(ep_chassis, tracker, args.xy_speed)
        found = reacquire_any_tower_with_sweep(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            model=model,
            tracker=tracker,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            tower_height_m=args.tower_height_m,
            forbidden_center_world=tower2_current_world,
            forbidden_radius_m=args.tower2_exclusion_radius_m,
            scan_side_m=args.scan_side_m,
            scan_forward_m=args.scan_forward_m,
            scan_rows=args.scan_rows,
            xy_speed=args.xy_speed,
            show=args.show,
        )
        if not found:
            raise RuntimeError("Could not reacquire moved tower 1 during search sweeps.")

        print("[7/8] Aligning to reacquired tower 1 and picking it...")
        align_and_approach_target(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            model=model,
            tracker=tracker,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            expected_lateral_m=None,
            tower_height_m=args.tower_height_m,
            forbidden_center_world=tower2_current_world,
            forbidden_radius_m=args.tower2_exclusion_radius_m,
            desired_h_px=args.align_desired_h_px,
            center_tol_px=args.align_center_tol_px,
            height_tol_px=args.align_height_tol_px,
            k_forward=args.k_forward,
            k_lateral=args.k_lateral,
            lateral_sign=args.lateral_sign,
            max_v=args.max_v,
            step_s=args.servo_step_s,
            show=args.show,
        )
        pick_up_tower(ep_robot=ep_robot)

        print("[8/8] Placing tower 1 at tower 2 original slot...")
        return_home(ep_chassis, tracker, args.xy_speed)
        move_relative(ep_chassis, tracker, tower2_slot.forward_m, tower2_slot.lateral_m, args.xy_speed)
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
