from dataclasses import dataclass
from queue import Empty
import math
import time

import robomaster
from robomaster import robot
import cv2
from ultralytics import YOLO
from robomaster import camera

from config import (
    DEFAULT_APPROACH_Y,
    DEFAULT_ARM_X,
    DEFAULT_DETECT_CONF,
    DEFAULT_GRIPPER_POWER,
    DEFAULT_GRIPPER_WAIT_SECONDS,
    DEFAULT_LOWER_Y,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAISED_Y,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
)


@dataclass
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    cls: int


def _start_arm_position_logging(ep_arm, enabled=False, freq=10):
    """Subscribe to arm position updates and keep the latest value available."""
    arm_state = {"enabled": enabled, "pos": None}
    if not enabled:
        return arm_state

    def on_arm_position(pos_info):
        pos_x, pos_y = pos_info
        arm_state["pos"] = (pos_x, pos_y)
        print(f"[Arm] live position x={pos_x}, y={pos_y}")

    ep_arm.sub_position(freq=freq, callback=on_arm_position)
    return arm_state


def _stop_arm_position_logging(ep_arm, arm_state):
    if arm_state["enabled"]:
        ep_arm.unsub_position()


def _print_latest_arm_position(arm_state, label):
    if not arm_state["enabled"]:
        return
    if arm_state["pos"] is None:
        print(f"[Arm] {label}: position not received yet")
        return
    pos_x, pos_y = arm_state["pos"]
    print(f"[Arm] {label}: latest x={pos_x}, y={pos_y}")


def print_current_arm_position(ep_robot, wait_seconds=0.5, freq=10):
    """Subscribe briefly to the arm position feed and print the current position."""
    ep_arm = ep_robot.robotic_arm
    arm_state = _start_arm_position_logging(ep_arm, enabled=True, freq=freq)
    try:
        deadline = time.time() + wait_seconds
        while arm_state["pos"] is None and time.time() < deadline:
            time.sleep(0.05)

        if arm_state["pos"] is None:
            print("[Arm] current position not received")
            return None

        pos_x, pos_y = arm_state["pos"]
        print(f"[Arm] current position x={pos_x}, y={pos_y}")
        return arm_state["pos"]
    finally:
        _stop_arm_position_logging(ep_arm, arm_state)


def resolve_resolution(name):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


def load_model(model_path=DEFAULT_MODEL_PATH):
    return YOLO(str(model_path))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def start_camera_stream(ep_robot, resolution="360p"):
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(resolution))
    return ep_camera


def move_arm_to_top(
    ep_robot,
    arm_x=DEFAULT_ARM_X,
    raised_y=DEFAULT_RAISED_Y,
    arm_state=None,
):
    """Move the arm to the raised reference posture used for travel/calibration."""
    ep_robot.robotic_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
    if arm_state is not None:
        _print_latest_arm_position(arm_state, "after move_arm_to_top")

def move_arm_to_default(
    ep_robot,
    default_x=0,
    default_y=0,
    arm_state=None,
):
    """Move the arm to default position."""
    ep_robot.robotic_arm.moveto(x=default_x, y=default_y).wait_for_completed()
    if arm_state is not None:
        _print_latest_arm_position(arm_state, "after move_arm_to_default")


def get_detections(model, frame, conf_thresh=DEFAULT_DETECT_CONF, target_class=None):
    result = model.predict(source=frame, show=False, conf=conf_thresh, verbose=False)[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy.cpu().numpy().flatten()
        conf = float(box.conf.item())
        cls = int(box.cls.item())
        if target_class is not None and cls != target_class:
            continue

        x1, y1, x2, y2 = [float(value) for value in xyxy]
        detections.append(
            Detection(
                cx=0.5 * (x1 + x2),
                cy=0.5 * (y1 + y2),
                w=max(1.0, x2 - x1),
                h=max(1.0, y2 - y1),
                conf=conf,
                cls=cls,
            )
        )

    return detections


def select_detection(detections, selection_mode="conf", frame_center_x=None):
    if not detections:
        return None

    if selection_mode == "leftmost":
        return min(detections, key=lambda detection: (detection.cx, -detection.conf))

    if selection_mode == "rightmost":
        return max(detections, key=lambda detection: (detection.cx, detection.conf))

    if selection_mode == "center" and frame_center_x is not None:
        return min(detections, key=lambda detection: (abs(detection.cx - frame_center_x), -detection.conf))

    return max(detections, key=lambda detection: detection.conf)

def connect_robot(
    conn_type="sta",
    robot_ip=DEFAULT_ROBOT_IP,
    sn=DEFAULT_ROBOT_SN,
):
    """Create and initialize a RoboMaster connection."""
    if conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = robot_ip

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=conn_type, sn=sn)
    return ep_robot


def pick_up_tower(
    ep_robot=None,
    conn_type="sta",
    robot_ip=DEFAULT_ROBOT_IP,
    sn=DEFAULT_ROBOT_SN,
    arm_x=DEFAULT_ARM_X,
    approach_y=DEFAULT_APPROACH_Y,
    lower_y=DEFAULT_LOWER_Y,
    raised_y=DEFAULT_RAISED_Y,
    gripper_power=DEFAULT_GRIPPER_POWER,
    grip_wait_seconds=DEFAULT_GRIPPER_WAIT_SECONDS,
    log_arm_position=False,
):
    """Pick up a tower by lowering close-in first, then extending and lifting."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    arm_state = {"enabled": False, "pos": None}
    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper
        arm_state = _start_arm_position_logging(ep_arm, enabled=log_arm_position)

        # Start from home, lower while retracted, then extend at that lowered height.
        move_arm_to_default(ep_robot, arm_state=arm_state)
        ep_arm.moveto(x=0, y=lower_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after lower while retracted")
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after extend to pickup")

        ep_gripper.close(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        # Lift first, then retract to a stable raised posture near the robot.
        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after lift")
        ep_arm.moveto(x=0, y=raised_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after retract raised")
        return ep_robot
    finally:
        _stop_arm_position_logging(ep_robot.robotic_arm, arm_state)
        if owns_robot:
            ep_robot.close()


def place_down_tower(
    ep_robot=None,
    conn_type="sta",
    robot_ip=DEFAULT_ROBOT_IP,
    sn=DEFAULT_ROBOT_SN,
    arm_x=DEFAULT_ARM_X,
    approach_y=DEFAULT_APPROACH_Y,
    lower_y=DEFAULT_LOWER_Y,
    raised_y=DEFAULT_RAISED_Y,
    gripper_power=DEFAULT_GRIPPER_POWER,
    grip_wait_seconds=DEFAULT_GRIPPER_WAIT_SECONDS,
    log_arm_position=False,
):
    """Place a tower by extending high, lowering, releasing, then returning home."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    arm_state = {"enabled": False, "pos": None}
    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper
        arm_state = _start_arm_position_logging(ep_arm, enabled=log_arm_position)

        # Keep the arm high while moving forward, then lower to place.
        # ep_arm.moveto(x=0, y=raised_y).wait_for_completed() 
        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after extend raised")
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()
        _print_latest_arm_position(arm_state, "after lower to place")

        ep_gripper.open(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        move_arm_to_default(ep_robot, arm_state=arm_state)
        return ep_robot
    finally:
        _stop_arm_position_logging(ep_robot.robotic_arm, arm_state)
        if owns_robot:
            ep_robot.close()
