#!/usr/bin/env python3
# Run from repo root:
# python "Project 2/test_go_to_tower.py" --show

"""Single-tower go-and-pick test using the same camera/model pipeline as swap_and_place_2.

Use this script to tune visual-servo parameters before running the full swap flow.
"""

import argparse

import cv2
from ultralytics import YOLO

import robomaster
from robomaster import camera, robot

from swap_and_place_2 import (
    ActionStack,
    MODEL_PATH,
    ROBOT_IP,
    ROBOT_SN,
    go_to_tower_recorded,
)
from tower_utils import (
    DEFAULT_ALIGN_TOP_TOL_PX,
    DEFAULT_TARGET_TOP_Y_RATIO,
    pick_up_tower, 
    move_arm_to_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go to one detected tower and pick it up (tuning helper)."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN)
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"])

    parser.add_argument("--detect-conf", type=float, default=0.45)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        default="conf",
        choices=["conf", "leftmost", "rightmost", "center"],
        help="How to choose a tower when multiple are visible.",
    )

    parser.add_argument("--target-top-y-ratio", type=float, default=DEFAULT_TARGET_TOP_Y_RATIO)
    parser.add_argument("--align-center-tol-px", type=float, default=24.0)
    parser.add_argument("--align-top-tol-px", type=float, default=DEFAULT_ALIGN_TOP_TOL_PX)
    parser.add_argument("--k-forward", type=float, default=0.0028)
    parser.add_argument("--k-yaw", type=float, default=0.12)
    parser.add_argument("--max-v", type=float, default=0.16)
    parser.add_argument("--max-yaw-dps", type=float, default=45.0)
    parser.add_argument("--servo-step-s", type=float, default=0.12)
    parser.add_argument("--timeout-s", type=float, default=30.0)

    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_resolution(name: str):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


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
    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    stack = ActionStack()

    try:
        move_arm_to_default(ep_robot)
        ep_robot.gripper.open()

        print("[1] Approaching selected tower ...")
        final_det = go_to_tower_recorded(
            ep_robot=ep_robot,
            model=model,
            ep_camera=ep_camera,
            ep_chassis=ep_chassis,
            action_stack=stack,
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
            timeout_s=args.timeout_s,
            selection_mode=args.selection_mode,
            show=args.show,
        )
        print(
            f"    Reached tower: conf={final_det.conf:.2f}, cx={final_det.cx:.1f}, cy={final_det.cy:.1f}"
        )

        print("[2] Picking up tower ...")
        pick_up_tower(ep_robot=ep_robot)
        print("\n✓ Pickup test complete.")

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
