#!/usr/bin/env python3
"""Swap-and-place workflow using an action-stack for drift-resistant homing.

Instead of maintaining a live geometric pose estimate and solving for a
return path, every chassis command issued while traveling to a tower is
pushed onto a stack.  Returning home simply pops and *reverses* each
command: forward becomes backward, lateral becomes opposite lateral, and
yaw turns in the opposite direction.  Because the robot can strafe, no
U-turn is required – the reversed commands drive it straight back.

High-level sequence
───────────────────
 1.  Scan from home: detect two towers, record left (T1) and right (T2).
 2.  Go to T1 (leftmost), recording route_to_t1.
 3.  Pick up T1.
 4.  Drive to stash spot (turn 90°, drive forward), recording stash_route.
 5.  Place T1 at stash spot.
 6.  Reverse stash_route → back at T1's original (now empty) slot.
 7.  Reverse route_to_t1 → back at home.
 8.  Go to T2 (rightmost), recording route_to_t2.
 9.  Pick up T2.
10.  Reverse route_to_t2 → back at home.
11.  Replay route_to_t1 → arrive at T1's original (empty) slot.
12.  Place T2 down (T2 is now at T1's original slot).
13.  Reverse route_to_t1 → back at home.
14.  Rescan: find T1 at stash spot (exclude T2's column).
15.  Go to stashed T1, recording route_to_stash.
16.  Pick up T1.
17.  Reverse route_to_stash → back at home.
18.  Replay route_to_t2 → arrive at T2's original slot.
19.  Place T1 down (T1 is now at T2's original slot).
20.  Reverse route_to_t2 → back at home.
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Deque, List, Optional, Tuple

import cv2
from ultralytics import YOLO

import robomaster
from robomaster import robot, camera

from tower_utils import (
    DEFAULT_ALIGN_TOP_TOL_PX,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
    DEFAULT_TARGET_TOP_Y_RATIO,
    Detection,
    get_detections,
    pick_up_tower,
    place_down_tower,
    clamp,
    select_detection,
    move_arm_to_top,
    move_arm_to_default
)

# ── Model / robot constants ────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt"
ROBOT_IP   = "192.168.50.117"
ROBOT_SN   = "3JKCH8800100RC"

# ── Stash parameters ───────────────────────────────────────────────────────────
STASH_YAW_DEG      = 90.0   # degrees to turn before driving to stash spot
STASH_YAW_DPS      = 45.0   # yaw rate used for the stash turn (deg/s)
STASH_FORWARD_M    = 0.35   # metres to drive forward to the stash spot
STASH_FORWARD_MPS  = 0.15   # forward speed used while stashing (m/s)


# ──────────────────────────────────────────────────────────────────────────────
# Action stack
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DriveAction:
    """One timestep of chassis motion."""
    vx: float   # forward speed (m/s)
    vy: float   # lateral speed (m/s)
    vz: float   # yaw rate   (deg/s)
    dt: float   # duration   (s)


class ActionStack:
    """Records drive actions so they can be replayed or reversed."""

    def __init__(self) -> None:
        self._stack: Deque[DriveAction] = deque()

    def push(self, action: DriveAction) -> None:
        self._stack.append(action)

    def clear(self) -> None:
        self._stack.clear()

    def snapshot(self) -> List[DriveAction]:
        """Return an ordered copy (start → destination) for later forward replay."""
        return list(self._stack)

    def unwind(self, ep_chassis, ep_robot=None, pause_s: float = 0.05) -> None:
        """Reverse every recorded action to drive back to where recording started."""
        if ep_robot is not None:
            move_arm_to_default(ep_robot)
        while self._stack:
            action = self._stack.pop()
            ep_chassis.drive_speed(
                x=-action.vx,
                y=-action.vy,
                z=-action.vz,
                timeout=action.dt,
            )
            time.sleep(action.dt + pause_s)
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)

    def replay(self, ep_chassis, pause_s: float = 0.05) -> None:
        """Replay actions in forward order (re-travel a previously recorded route)."""
        for action in self._stack:
            ep_chassis.drive_speed(
                x=action.vx,
                y=action.vy,
                z=action.vz,
                timeout=action.dt,
            )
            time.sleep(action.dt + pause_s)
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)


def replay_route(ep_chassis, route: List[DriveAction], ep_robot=None, pause_s: float = 0.05) -> None:
    """Replay a snapshot (list) of actions in forward order."""
    if ep_robot is not None:
        move_arm_to_default(ep_robot)
    for action in route:
        ep_chassis.drive_speed(
            x=action.vx,
            y=action.vy,
            z=action.vz,
            timeout=action.dt,
        )
        time.sleep(action.dt + pause_s)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)


def reverse_route(ep_chassis, route: List[DriveAction], ep_robot=None, pause_s: float = 0.05) -> None:
    """Replay a snapshot in reverse order with negated velocities."""
    if ep_robot is not None:
        move_arm_to_default(ep_robot)
    for action in reversed(route):
        ep_chassis.drive_speed(
            x=-action.vx,
            y=-action.vy,
            z=-action.vz,
            timeout=action.dt,
        )
        time.sleep(action.dt + pause_s)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Stash helper
# ──────────────────────────────────────────────────────────────────────────────

def drive_to_stash(ep_chassis, ep_robot=None) -> List[DriveAction]:
    """Turn 90° LEFT then drive forward to the stash spot.

    The robot stops AT the stash spot with the tower still in the gripper.
    The caller is responsible for calling place_down_tower() before reversing.

    Returns stash_route (the two actions taken) so the caller can reverse them
    to get back to T1's original slot.
    """
    stash_route: List[DriveAction] = []
    pause_s = 0.05

    if ep_robot is not None:
        move_arm_to_default(ep_robot)

    # 1. Yaw 90° LEFT
    yaw_dt = STASH_YAW_DEG / STASH_YAW_DPS
    left_yaw_dps = -abs(STASH_YAW_DPS)
    yaw_action = DriveAction(vx=0.0, vy=0.0, vz=left_yaw_dps, dt=yaw_dt)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=left_yaw_dps, timeout=yaw_dt)
    time.sleep(yaw_dt + pause_s)
    stash_route.append(yaw_action)

    # 2. Drive forward to stash spot
    fwd_dt = STASH_FORWARD_M / STASH_FORWARD_MPS
    fwd_action = DriveAction(vx=STASH_FORWARD_MPS, vy=0.0, vz=0.0, dt=fwd_dt)
    ep_chassis.drive_speed(x=STASH_FORWARD_MPS, y=0.0, z=0.0, timeout=fwd_dt)
    time.sleep(fwd_dt + pause_s)
    stash_route.append(fwd_action)

    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    time.sleep(0.15)

    # Robot is now at the stash spot. Caller places the tower, then calls
    # reverse_route(ep_chassis, stash_route) to return to T1's original slot.
    return stash_route


# ──────────────────────────────────────────────────────────────────────────────
# Visual-servo go_to_tower with stack recording
# ──────────────────────────────────────────────────────────────────────────────

def go_to_tower_recorded(
    ep_robot,
    model: YOLO,
    ep_camera,
    ep_chassis,
    action_stack: ActionStack,
    target_class: Optional[int] = None,
    conf_thresh: float = 0.45,
    target_top_y_ratio: float = DEFAULT_TARGET_TOP_Y_RATIO,
    center_tol_px: float = 24.0,
    top_y_tol_px: float = DEFAULT_ALIGN_TOP_TOL_PX,
    k_forward: float = 0.0028,
    k_yaw: float = 0.12,
    max_v: float = 0.16,
    max_yaw_dps: float = 45.0,
    step_s: float = 0.12,
    timeout_s: float = 30.0,
    selection_mode: str = "conf",
    show: bool = False,
) -> Detection:
    """Drive toward a tower using visual servoing, recording every drive command.

    Every ``drive_speed`` command is pushed onto *action_stack* so the caller
    can later unwind (return to start) or snapshot+replay (re-visit the spot).

    Returns the final Detection used to declare arrival.
    """
    action_stack.clear()
    move_arm_to_default(ep_robot)

    stable = 0
    center_stable = 0
    t0 = time.time()
    selected: Optional[Detection] = None

    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out while approaching tower.")

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
        except Empty:
            continue
        except Exception as e:
            print(f"camera read error: {e}")
            continue
        if frame is None:
            continue

        detections = get_detections(model, frame, conf_thresh, target_class)
        if not detections:
            # Hold still but record the idle action so unwind stays accurate.
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=step_s)
            action_stack.push(DriveAction(vx=0.0, vy=0.0, vz=0.0, dt=step_s))
            continue

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]
        frame_center_x = frame_w / 2.0

        selected = select_detection(
            detections=detections,
            selection_mode=selection_mode,
            frame_center_x=frame_center_x,
        )

        y_top_px = selected.cy - selected.h / 2.0
        err_x_px = selected.cx - frame_center_x
        target_top_y_px = target_top_y_ratio * frame_h
        err_forward_px = target_top_y_px - y_top_px

        centered = abs(err_x_px) <= center_tol_px
        if centered:
            center_stable += 1
        else:
            center_stable = 0

        allow_forward = center_stable >= 2

        if not allow_forward:
            vx = 0.0
            # Reverse yaw sign so left-side detections produce the correct turn direction
            vz = clamp(k_yaw * err_x_px, -max_yaw_dps, max_yaw_dps)
        else:
            vx = clamp(k_forward * err_forward_px, -max_v, max_v)
            vz = 0.0

        ep_chassis.drive_speed(x=vx, y=0.0, z=vz, timeout=step_s)
        action_stack.push(DriveAction(vx=vx, vy=0.0, vz=vz, dt=step_s))

        if abs(err_x_px) <= center_tol_px and abs(err_forward_px) <= top_y_tol_px:
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
            cv2.line(dbg, (int(frame_center_x), 0), (int(frame_center_x), frame_h - 1), (0, 255, 255), 1)
            # Add class/conf label if available
            try:
                name = model.names[selected.cls]
            except Exception:
                name = str(selected.cls)
            label = f"{name} {selected.conf:.2f}"
            cv2.putText(dbg, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(
                dbg,
                f"err_x={err_x_px:+.1f} vz={vz:+.1f} fwd_err={err_forward_px:+.1f} stable={stable}/4",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.imshow("go_to_tower_recorded", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if stable >= 4:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
            return selected

    raise RuntimeError("go_to_tower_recorded exited loop unexpectedly.")


# ──────────────────────────────────────────────────────────────────────────────
# Initial two-tower scan
# ──────────────────────────────────────────────────────────────────────────────

def detect_stable_two_towers(
    ep_camera,
    model: YOLO,
    conf_thresh: float,
    target_class: Optional[int],
    required_stable_frames: int = 8,
    timeout_s: float = 25.0,
    show: bool = False,
) -> Tuple[Detection, Detection]:
    """Block until two towers are confidently detected in several consecutive frames.

    Returns (left_tower, right_tower) sorted by image x-coordinate.
    """
    t0 = time.time()
    stable = 0
    latest_pair: Optional[Tuple[Detection, Detection]] = None

    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out waiting to see two towers.")

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
        except Empty:
            stable = 0
            continue
        except Exception as e:
            print(f"camera read error: {e}")
            stable = 0
            continue
        if frame is None:
            stable = 0
            continue

        dets = get_detections(model, frame, conf_thresh, target_class)
        if len(dets) < 2:
            stable = 0
            if show and frame is not None:
                cv2.putText(frame, f"need 2 towers, got {len(dets)}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("scan", frame)
                cv2.waitKey(1)
            continue

        dets_sorted_conf = sorted(dets, key=lambda d: d.conf, reverse=True)[:2]
        left, right = sorted(dets_sorted_conf, key=lambda d: d.cx)
        latest_pair = (left, right)
        stable += 1

        if show:
            dbg = frame.copy()
            for det in [left, right]:
                x1 = int(det.cx - det.w / 2); y1 = int(det.cy - det.h / 2)
                x2 = int(det.cx + det.w / 2); y2 = int(det.cy + det.h / 2)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(dbg, f"stable={stable}/{required_stable_frames}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("scan", dbg)
            cv2.waitKey(1)

        if stable >= required_stable_frames:
            print(f"  Two towers confirmed: left cx={latest_pair[0].cx:.0f}px, right cx={latest_pair[1].cx:.0f}px")
            return latest_pair


def detect_single_tower_excluding(
    ep_camera,
    model: YOLO,
    conf_thresh: float,
    target_class: Optional[int],
    forbidden_cx: float,
    exclusion_tol_px: float = 90.0,
    required_stable_frames: int = 5,
    timeout_s: float = 25.0,
    show: bool = False,
) -> str:
    """Scan for a single tower that is NOT near *forbidden_cx*.

    Returns a selection_mode hint ("leftmost" or "rightmost") for use with
    go_to_tower_recorded.
    """
    t0 = time.time()
    stable = 0

    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out searching for stashed Tower 1.")

        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
        except Empty:
            stable = 0
            continue
        except Exception as e:
            print(f"camera read error: {e}")
            stable = 0
            continue
        if frame is None:
            stable = 0
            continue

        dets = get_detections(model, frame, conf_thresh, target_class)
        candidates = [d for d in dets if abs(d.cx - forbidden_cx) > exclusion_tol_px]

        if not candidates:
            stable = 0
            if show:
                cv2.putText(frame, "searching for T1 ...", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("scan", frame)
                cv2.waitKey(1)
            continue

        best = max(candidates, key=lambda d: d.conf)
        stable += 1

        if show:
            dbg = frame.copy()
            x1 = int(best.cx - best.w / 2); y1 = int(best.cy - best.h / 2)
            x2 = int(best.cx + best.w / 2); y2 = int(best.cy + best.h / 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(dbg, f"T1 stable={stable}/{required_stable_frames}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("scan", dbg)
            cv2.waitKey(1)

        if stable >= required_stable_frames:
            frame_center_x = frame.shape[1] / 2.0
            hint = "leftmost" if best.cx < frame_center_x else "rightmost"
            print(f"  Tower 1 reacquired at cx={best.cx:.0f}px → selection_mode='{hint}'")
            return hint


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap two towers with action-stack homing")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN)
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"])
    parser.add_argument("--detect-conf", type=float, default=0.45)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--target-top-y-ratio", type=float, default=DEFAULT_TARGET_TOP_Y_RATIO)
    parser.add_argument("--align-center-tol-px", type=float, default=24.0)
    parser.add_argument("--align-top-tol-px", type=float, default=DEFAULT_ALIGN_TOP_TOL_PX)
    parser.add_argument("--k-forward", type=float, default=0.0028)
    parser.add_argument("--k-yaw", type=float, default=0.12)
    parser.add_argument("--max-v", type=float, default=0.16)
    parser.add_argument("--max-yaw-dps", type=float, default=45.0)
    parser.add_argument("--servo-step-s", type=float, default=0.12)
    parser.add_argument("--exclusion-tol-px", type=float, default=90.0,
                        help="Pixel tolerance for excluding the already-placed tower during T1 rescan.")
    parser.add_argument("--stash-yaw-deg", type=float, default=STASH_YAW_DEG,
                        help="Degrees to turn before driving to the stash spot.")
    parser.add_argument("--stash-forward-m", type=float, default=STASH_FORWARD_M,
                        help="Metres to drive forward to the stash spot.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_resolution(name: str):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("Loading model ...")
    model = YOLO(args.model_path)

    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = args.robot_ip

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=args.sn)
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    # Disable the SDK's built-in display so we can show annotated frames
    # (bounding boxes) using OpenCV windows from the processing code.
    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    global STASH_YAW_DEG, STASH_FORWARD_M
    STASH_YAW_DEG   = args.stash_yaw_deg
    STASH_FORWARD_M = args.stash_forward_m

    servo_kwargs = dict(
        target_class=args.target_class,
        conf_thresh=args.detect_conf,
        target_top_y_ratio=args.target_top_y_ratio,
        center_tol_px=args.align_center_tol_px,
        top_y_tol_px=args.align_top_tol_px,
        k_forward=args.k_forward,
        k_yaw=args.k_yaw,
        max_v=args.max_v,
        max_yaw_dps=args.max_yaw_dps,
        step_s=args.servo_step_s,
        show=args.show,
    )

    stack = ActionStack()

    try:
        move_arm_to_default(ep_robot)
        ep_robot.gripper.open()
        # ── Step 1: Scan from home ─────────────────────────────────────────
        print("[1] Scanning for two towers from home ...")
        t1_det, t2_det = detect_stable_two_towers(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            show=args.show,
        )
        # t1_det.cx is T1's pixel column at home — used later to exclude it
        # when searching for the stashed T1 after T2 has been placed there.
        t1_home_cx = t1_det.cx

        # ── Step 2: Go to T1, pick it up ──────────────────────────────────
        print("[2] Going to T1 (leftmost) ...")
        go_to_tower_recorded(
            ep_robot=ep_robot, model=model,
            ep_camera=ep_camera, ep_chassis=ep_chassis,
            action_stack=stack, selection_mode="leftmost",
            **servo_kwargs,
        )
        route_to_t1: List[DriveAction] = stack.snapshot()
        print("    Picking up T1 ...")

        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 3-4: Drive to stash spot, place T1 ───────────────────────
        print("[3] Driving to stash spot ...")
        stash_route: List[DriveAction] = drive_to_stash(ep_chassis, ep_robot=ep_robot)
        # Robot is now AT the stash spot holding T1.
        print("[4] Placing T1 at stash spot ...")
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        place_down_tower(ep_robot=ep_robot)

        # ── Step 5: Reverse stash route → back at T1's original slot ──────
        print("[5] Reversing stash route → back at T1's original (now empty) slot ...")
        reverse_route(ep_chassis, stash_route, ep_robot=ep_robot)

        # ── Step 6: Reverse route_to_t1 → back at home ────────────────────
        print("[6] Reversing route_to_t1 → back at home ...")
        reverse_route(ep_chassis, route_to_t1, ep_robot=ep_robot)

        # ── Step 7-8: Go to T2, pick it up ────────────────────────────────
        print("[7] Going to T2 (rightmost) ...")
        go_to_tower_recorded(
            ep_robot=ep_robot, model=model,
            ep_camera=ep_camera, ep_chassis=ep_chassis,
            action_stack=stack, selection_mode="rightmost",
            **servo_kwargs,
        )
        route_to_t2: List[DriveAction] = stack.snapshot()
        print("    Picking up T2 ...")
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 9: Reverse route_to_t2 → back at home ────────────────────
        print("[9] Reversing route_to_t2 → back at home ...")
        stack.unwind(ep_chassis, ep_robot=ep_robot)

        # ── Step 10-11: Replay route_to_t1, place T2 at T1's original slot
        print("[10] Replaying route_to_t1 → arriving at T1's original slot ...")
        replay_route(ep_chassis, route_to_t1, ep_robot=ep_robot)
        print("[11] Placing T2 at T1's original slot ...")
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        place_down_tower(ep_robot=ep_robot)

        # ── Step 12: Reverse route_to_t1 → back at home ───────────────────
        print("[12] Reversing route_to_t1 → back at home ...")
        reverse_route(ep_chassis, route_to_t1, ep_robot=ep_robot)

        # ── Step 13: Rescan — find stashed T1, excluding T2's column ──────
        # T2 now sits where T1 was, so suppress detections near t1_home_cx.
        print("[13] Rescanning for stashed T1 (excluding T2 at T1's original column) ...")
        t1_hint = detect_single_tower_excluding(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            forbidden_cx=t1_home_cx,
            exclusion_tol_px=args.exclusion_tol_px,
            show=args.show,
        )

        # ── Step 14-15: Go to stashed T1, pick it up ──────────────────────
        print(f"[14] Going to stashed T1 (hint='{t1_hint}') ...")
        go_to_tower_recorded(
            ep_robot=ep_robot, model=model,
            ep_camera=ep_camera, ep_chassis=ep_chassis,
            action_stack=stack, selection_mode=t1_hint,
            **servo_kwargs,
        )
        print("     Picking up T1 ...")
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 16: Reverse route_to_stash → back at home ────────────────
        print("[16] Reversing route_to_stash → back at home ...")
        stack.unwind(ep_chassis, ep_robot=ep_robot)

        # ── Step 17-18: Replay route_to_t2, place T1 at T2's original slot
        print("[17] Replaying route_to_t2 → arriving at T2's original slot ...")
        replay_route(ep_chassis, route_to_t2, ep_robot=ep_robot)
        print("[18] Placing T1 at T2's original slot ...")
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.2)
        time.sleep(0.3)
        place_down_tower(ep_robot=ep_robot)

        # ── Step 19: Reverse route_to_t2 → back at home ───────────────────
        print("[19] Reversing route_to_t2 → back at home ...")
        reverse_route(ep_chassis, route_to_t2, ep_robot=ep_robot)

        print("\n✓ Swap complete. T1 is at T2's original slot; T2 is at T1's original slot.")

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


if __name__ == "__main__":
    main()