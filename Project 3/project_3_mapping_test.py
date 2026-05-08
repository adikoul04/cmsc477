#!/usr/bin/env python3
"""
Mapping-only test runner for the Project 3 updated workflow.

This script follows the same deterministic mapping sequence used by
`project_3_updated.py` but stops after mapping. It does not perform any pickup,
delivery, or recharge actions beyond the initial mapping pass.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import robomaster
from config import DEFAULT_ARM_X, DEFAULT_ARM_Y, DEFAULT_MODEL_PATH, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN, MODEL_PATH, ROBOT_IP, ROBOT_SN
from robomaster import camera as rm_camera
from robomaster import robot

import project_3_updated as p3


def visualize_mapping_test(
    world_map: p3.WorldMap,
    start_pose: p3.Pose2D,
    final_pose: Optional[p3.Pose2D] = None,
) -> None:
    """Render the mapping results, including the required start position."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[Map] matplotlib not available. Skipping visualisation.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0.0, p3.WORKSPACE_W_M)
    ax.set_ylim(0.0, p3.WORKSPACE_H_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Project 3 Mapping Test")

    for value in np.arange(0.0, p3.WORKSPACE_W_M + 0.01, 0.10):
        ax.axvline(value, color="lightgray", linewidth=0.3)
    for value in np.arange(0.0, p3.WORKSPACE_H_M + 0.01, 0.10):
        ax.axhline(value, color="lightgray", linewidth=0.3)

    ax.plot(
        [0, p3.WORKSPACE_W_M, p3.WORKSPACE_W_M, 0, 0],
        [0, 0, p3.WORKSPACE_H_M, p3.WORKSPACE_H_M, 0],
        "k-",
        linewidth=2,
    )

    ax.plot(start_pose.x, start_pose.y, "ks", markersize=9)
    ax.annotate("start", (start_pose.x + 0.04, start_pose.y + 0.04), color="black")

    for obstacle in world_map.obstacles:
        ax.add_patch(plt.Circle((obstacle.x, obstacle.y), 0.12, color="red", alpha=0.6))
        label = f"obs {obstacle.tag_id}" if obstacle.tag_id is not None else "obs"
        ax.annotate(label, (obstacle.x + 0.04, obstacle.y + 0.04), color="red")

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
        ax.annotate("recharge", (world_map.recharge.x + 0.04, world_map.recharge.y + 0.04), color="black")

    if world_map.small_goal:
        ax.plot(world_map.small_goal.x, world_map.small_goal.y, "b^", markersize=14)
        ax.annotate("small goal", (world_map.small_goal.x + 0.04, world_map.small_goal.y + 0.04), color="blue")

    if world_map.large_goal:
        ax.plot(world_map.large_goal.x, world_map.large_goal.y, "g^", markersize=14)
        ax.annotate("large goal", (world_map.large_goal.x + 0.04, world_map.large_goal.y + 0.04), color="green")

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
        ax.annotate("dock", (world_map.dock.x + 0.04, world_map.dock.y + 0.04), color="darkgoldenrod")

    if world_map.intermediate:
        ax.plot(world_map.intermediate.x, world_map.intermediate.y, "co", markersize=10)
        ax.annotate("intermediate", (world_map.intermediate.x + 0.04, world_map.intermediate.y + 0.04), color="teal")

    if final_pose is not None:
        ax.plot(final_pose.x, final_pose.y, "ms", markersize=10)
        dx = 0.15 * math.cos(final_pose.yaw)
        dy = 0.15 * math.sin(final_pose.yaw)
        ax.annotate(
            "",
            xy=(final_pose.x + dx, final_pose.y + dy),
            xytext=(final_pose.x, final_pose.y),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=2),
        )
        ax.annotate("final pose", (final_pose.x + 0.04, final_pose.y - 0.08), color="magenta")

    plt.tight_layout()
    plt.savefig("arena_map_mapping_test.png", dpi=150)
    print("[Map] Saved to arena_map_mapping_test.png")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3 mapping-only test")
    parser.add_argument("--model-path", default=str(MODEL_PATH or DEFAULT_MODEL_PATH))
    parser.add_argument("--robot-ip", default=ROBOT_IP or DEFAULT_ROBOT_IP)
    parser.add_argument("--sn", default=ROBOT_SN or DEFAULT_ROBOT_SN)
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"])
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"])
    parser.add_argument("--show-map", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Project 3 Mapping Test ===")
    print(f"[Setup] workspace = {p3.WORKSPACE_W_M:.3f}m x {p3.WORKSPACE_H_M:.3f}m")
    print(
        "[Setup] start pose = "
        f"({p3.START_X_M:.3f}, {p3.START_Y_M:.3f}, {math.degrees(p3.START_YAW_RAD):.1f} deg)"
    )

    yolo_model = YOLO(str(args.model_path))
    tag_detector = p3.AprilTagDetector()
    pose = p3.Pose2D(x=p3.START_X_M, y=p3.START_Y_M, yaw=p3.START_YAW_RAD)
    start_pose = p3.copy_pose(pose)
    world_map = p3.WorldMap()

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

        target_goal = p3.execute_mapping_sequence(
            ep_camera,
            ep_chassis,
            yolo_model,
            tag_detector,
            pose,
            world_map,
        )
        print(f"[Mission] Mapping finished. Selected target goal: {target_goal.kind}")
        print(world_map.summary())
        print(
            "[Pose] final mapping pose = "
            f"({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):.1f} deg)"
        )

        if args.show_map:
            visualize_mapping_test(world_map, start_pose, pose)

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
