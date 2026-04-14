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
1.  Scan: detect two towers from home; record which is left / right.
2.  Go to Tower 1 (leftmost), recording every drive command on stack_t1.
3.  Pick up Tower 1; snapshot stack_t1 as route_to_t1.
4.  STASH Tower 1: turn 90°, drive forward, place Tower 1 down, then
    reverse those actions to return to Tower 1's original slot (now empty).
5.  Unwind stack_t1 → robot returns to home.
6.  Go to Tower 2 (rightmost), recording every drive command on stack_t2.
7.  Pick up Tower 2; snapshot stack_t2 as route_to_t2.
8.  Replay route_to_t1 → robot arrives at Tower 1's original (now empty) slot.
9.  Place Tower 2 down (Tower 2 is now at Tower 1's original slot).
10. Unwind stack_t1 → robot returns to home.
11. Scan for Tower 1 (ignoring the tower just placed at T1-slot).
12. Go to Tower 1's stash location, recording every drive command.
13. Pick up Tower 1.
14. Unwind stash stack → robot returns to home.
15. Replay route_to_t2 → robot arrives at Tower 2's original slot.
16. Place Tower 1 down (Tower 1 is now at Tower 2's original slot).
17. Unwind stack_t2 → robot returns to home.
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
)

# ── Model / robot constants (calibrated from bounding_box_capture.py) ─────────
MODEL_PATH = r"C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt"
ROBOT_IP   = "192.168.50.117"
ROBOT_SN   = "3JKCH8800100RC"

# ──────────────────────────────────────────────
# Stash parameters
# ──────────────────────────────────────────────
STASH_YAW_DEG      = 90.0   # degrees to turn before driving to stash spot
STASH_YAW_DPS      = 45.0   # yaw rate used for the stash turn (deg/s)
STASH_FORWARD_M    = 0.35   # metres to drive forward to the stash spot
STASH_FORWARD_MPS  = 0.15   # forward speed used while stashing (m/s)


# ──────────────────────────────────────────────
# Action stack
# ──────────────────────────────────────────────

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
        """Return an ordered copy (home → tower) for later forward replay."""
        return list(self._stack)

    def unwind(self, ep_chassis, ep_robot=None, pause_s: float = 0.05) -> None:
        """Reverse every recorded action to drive back to where recording started."""
        if ep_robot is not None:
            move_arm_to_top(ep_robot)
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
        move_arm_to_top(ep_robot)
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
        move_arm_to_top(ep_robot)
    for action in reversed(route):
        ep_chassis.drive_speed(
            x=-action.vx,
            y=-action.vy,
            z=-action.vz,
            timeout=action.dt,
        )
        time.sleep(action.dt + pause_s)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)


# ──────────────────────────────────────────────
# Stash / un-stash helpers
# ──────────────────────────────────────────────

def stash_tower(ep_chassis, ep_robot=None) -> List[DriveAction]:
    """Turn 90°, drive forward, place the held tower, then undo both moves.

    The robot ends up back at the exact spot it started (Tower 1's original
    slot), which is now empty.  Returns the stash route so the caller can
    re-use it later to visit the stash location again if needed.

    Route built:
        1. Yaw LEFT by STASH_YAW_DEG in place
        2. Drive forward STASH_FORWARD_M
        [place_down_tower is called by the caller between steps 2 and 3]
        3. Drive backward STASH_FORWARD_M   (reverse of step 2)
        4. Yaw RIGHT by STASH_YAW_DEG      (reverse of step 1)
    """
    stash_route: List[DriveAction] = []
    pause_s = 0.05

    if ep_robot is not None:
        move_arm_to_top(ep_robot)

    # ── 1. Yaw 90° LEFT ───────────────────────────────────────────────────
    yaw_dt = STASH_YAW_DEG / STASH_YAW_DPS          # seconds needed for the turn
    left_yaw_dps = -abs(STASH_YAW_DPS)
    yaw_action = DriveAction(vx=0.0, vy=0.0, vz=left_yaw_dps, dt=yaw_dt)
    ep_chassis.drive_speed(x=0.0, y=0.0, z=left_yaw_dps, timeout=yaw_dt)
    time.sleep(yaw_dt + pause_s)
    stash_route.append(yaw_action)

    # ── 2. Drive forward to stash spot ────────────────────────────────────
    fwd_dt = STASH_FORWARD_M / STASH_FORWARD_MPS     # seconds needed to cover distance
    fwd_action = DriveAction(vx=STASH_FORWARD_MPS, vy=0.0, vz=0.0, dt=fwd_dt)
    ep_chassis.drive_speed(x=STASH_FORWARD_MPS, y=0.0, z=0.0, timeout=fwd_dt)
    time.sleep(fwd_dt + pause_s)
    stash_route.append(fwd_action)

    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
    time.sleep(0.15)

    # Caller places the tower here (see main sequence).
    return stash_route


def return_from_stash(ep_chassis, stash_route: List[DriveAction], ep_robot=None) -> None:
    """Reverse the stash route to return to Tower 1's original (now empty) slot."""
    reverse_route(ep_chassis, stash_route, ep_robot=ep_robot)


# ──────────────────────────────────────────────
# Visual-servo go_to_tower with stack recording
# ──────────────────────────────────────────────

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
    """Drive toward a tower using visual servoing.

    Every ``drive_speed`` command is pushed onto *action_stack* so the
    caller can later unwind (return home) or replay (re-visit) the route.

    Returns the final Detection used to stop.
    """
    action_stack.clear()
    move_arm_to_top(ep_robot)

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
            vz = clamp(-k_yaw * err_x_px, -max_yaw_dps, max_yaw_dps)
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


# ──────────────────────────────────────────────
# Initial two-tower scan
# ──────────────────────────────────────────────

def detect_stable_two_towers(
    ep_camera,
    model: YOLO,
    conf_thresh: float,
    target_class: Optional[int],
    required_stable_frames: int = 8,
    timeout_s: float = 25.0,
    show: bool = False,
) -> Tuple[Detection, Detection]:
    """Block until two towers are detected in several consecutive frames.

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
            raise TimeoutError("Timed out searching for stashed tower 1.")

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


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap two LEGO towers with action-stack homing")
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load model (path calibrated from bounding_box_capture.py) ─────────
    print("Loading model ...")
    model = YOLO(args.model_path)

    # ── Connect robot (conn_type / SN calibrated from bounding_box_capture.py)
    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = args.robot_ip

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=args.sn)
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    # Update module-level stash constants from CLI args so stash_tower() picks
    # them up even though it reads the globals directly.
    global STASH_YAW_DEG, STASH_FORWARD_M
    STASH_YAW_DEG   = args.stash_yaw_deg
    STASH_FORWARD_M = args.stash_forward_m

    # Shared servo kwargs forwarded to every go_to_tower_recorded call.
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

    stack = ActionStack()   # live recording stack (reused per leg)

    try:
        # ── Step 1: Initial scan ───────────────────────────────────────────
        print("[1/9] Scanning for two towers from home position ...")
        t1_det, t2_det = detect_stable_two_towers(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            show=args.show,
        )
        t1_home_cx = t1_det.cx   # leftmost

        # ── Step 2: Go to Tower 1, pick it up ─────────────────────────────
        print("[2/9] Going to Tower 1 (leftmost) ...")
        go_to_tower_recorded(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            action_stack=stack,
            selection_mode="leftmost",
            **servo_kwargs,
        )
        route_to_t1: List[DriveAction] = stack.snapshot()   # save home→T1 route
        print("       Picking up Tower 1 ...")
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 3: STASH Tower 1 off to the side ─────────────────────────
        # Turn 90°, drive forward, place Tower 1 at the stash spot, then
        # reverse back to Tower 1's original (now empty) slot.
        print("[3/9] Stashing Tower 1 (turn 90°, drive forward, place, return) ...")
        stash_route: List[DriveAction] = stash_tower(ep_chassis, ep_robot=ep_robot)
        print("       Placing Tower 1 at stash spot ...")
        place_down_tower(ep_robot=ep_robot)
        print("       Returning to Tower 1's original slot ...")
        return_from_stash(ep_chassis, stash_route, ep_robot=ep_robot)
        # Robot is now at Tower 1's original, empty slot.

        # ── Step 4: Return home by unwinding stack_t1 ─────────────────────
        print("[4/9] Returning home (unwind route_to_t1) ...")
        stack.unwind(ep_chassis, ep_robot=ep_robot)
        # stack is now empty; ready for next recording.

        # ── Step 5: Go to Tower 2, pick it up ─────────────────────────────
        print("[5/9] Going to Tower 2 (rightmost) ...")
        go_to_tower_recorded(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            action_stack=stack,
            selection_mode="rightmost",
            **servo_kwargs,
        )
        route_to_t2: List[DriveAction] = stack.snapshot()   # save home→T2 route
        print("       Picking up Tower 2 ...")
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 6: Drive to T1's original (now empty) slot; place Tower 2 ─
        print("[6/9] Unwinding to home; replaying route to T1 slot; placing Tower 2 ...")
        stack.unwind(ep_chassis, ep_robot=ep_robot)         # back to home from T2
        replay_route(ep_chassis, route_to_t1, ep_robot=ep_robot)
        print("       Placing Tower 2 at Tower 1's original slot ...")
        place_down_tower(ep_robot=ep_robot)

        # ── Step 7: Return home; rescan for stashed Tower 1 ───────────────
        print("[7/9] Returning home (reverse route_to_t1); rescanning for Tower 1 ...")
        reverse_route(ep_chassis, route_to_t1, ep_robot=ep_robot)

        # Tower 2 now occupies what was Tower 1's column at home view, so
        # exclude that column when searching for the stashed Tower 1.
        hint = detect_single_tower_excluding(
            ep_camera=ep_camera,
            model=model,
            conf_thresh=args.detect_conf,
            target_class=args.target_class,
            forbidden_cx=t1_home_cx,
            exclusion_tol_px=args.exclusion_tol_px,
            show=args.show,
        )

        # ── Step 8: Go to stashed Tower 1, pick it up ─────────────────────
        print(f"[8/9] Going to stashed Tower 1 (hint='{hint}') ...")
        go_to_tower_recorded(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            action_stack=stack,
            selection_mode=hint,
            **servo_kwargs,
        )
        print("       Picking up Tower 1 ...")
        pick_up_tower(ep_robot=ep_robot)

        # ── Step 9: Drive to T2's original slot; place Tower 1 ────────────
        print("[9/9] Unwinding to home; replaying route to T2 slot; placing Tower 1 ...")
        stack.unwind(ep_chassis, ep_robot=ep_robot)             # back to home from stash
        replay_route(ep_chassis, route_to_t2, ep_robot=ep_robot)
        print("       Placing Tower 1 at Tower 2's original slot ...")
        place_down_tower(ep_robot=ep_robot)

        # ── Final: return home ─────────────────────────────────────────────
        print("Returning home ...")
        reverse_route(ep_chassis, route_to_t2, ep_robot=ep_robot)

        print("\n✓ Swap-and-place sequence complete.")

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