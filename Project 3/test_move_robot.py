#!/usr/bin/env python3
"""
Simple motion test for Project 3's move_robot helper.

Sequence:
1. Turn 90 degrees left
2. Turn 90 degrees right
3. Move forward 1 meter
4. Turn 180 degrees
5. Move forward 1 meter
6. Turn 180 degrees
"""

from __future__ import annotations

import argparse
import math
import time

import robomaster
from config import DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN, ROBOT_IP, ROBOT_SN
from robomaster import robot

import project_3_updated as p3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Project 3 move_robot helper")
    parser.add_argument("--robot-ip", default=ROBOT_IP or DEFAULT_ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN or DEFAULT_ROBOT_SN)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--pause-s", type=float, default=1.5, help="Pause between motion commands.")
    return parser.parse_args()


def print_pose(label: str, pose: p3.Pose2D) -> None:
    print(f"[Test] {label}: x={pose.x:.2f} y={pose.y:.2f} yaw={math.degrees(pose.yaw):.1f} deg")


def main() -> None:
    args = parse_args()

    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = str(args.robot_ip)

    pose = p3.Pose2D(x=p3.START_X_M, y=p3.START_Y_M, yaw=p3.START_YAW_RAD)
    print("=== move_robot Motion Test ===")
    print_pose("start pose", pose)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=str(args.sn))
    ep_chassis = ep_robot.chassis

    try:
        p3.move_robot(ep_chassis, pose, z_deg=90.0)
        print_pose("after 90 left", pose)
        time.sleep(args.pause_s)

        p3.move_robot(ep_chassis, pose, z_deg=-90.0)
        print_pose("after 90 right", pose)
        time.sleep(args.pause_s)

        p3.move_robot(ep_chassis, pose, x_m=1.0)
        print_pose("after forward 1m", pose)
        time.sleep(args.pause_s)

        p3.move_robot(ep_chassis, pose, z_deg=180.0)
        print_pose("after 180 turn", pose)
        time.sleep(args.pause_s)

        p3.move_robot(ep_chassis, pose, x_m=1.0)
        print_pose("after second forward 1m", pose)
        time.sleep(args.pause_s)

        p3.move_robot(ep_chassis, pose, z_deg=180.0)
        print_pose("after final 180 turn", pose)

    finally:
        try:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
