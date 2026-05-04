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
):
    """Move the arm to the raised reference posture used for travel/calibration."""
    ep_robot.robotic_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()

def move_arm_to_default(
    ep_robot,
    default_x=0,
    default_y=0,
):
    """Move the arm to default position."""
    ep_robot.robotic_arm.moveto(x=default_x, y=default_y).wait_for_completed()


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
):
    """Pick up a tower by lowering close-in first, then extending and lifting."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper

        # Start from home, lower while retracted, then extend at that lowered height.
        move_arm_to_default(ep_robot)
        ep_arm.moveto(x=0, y=lower_y).wait_for_completed()
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()

        ep_gripper.close(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        # Lift first, then retract to a stable raised posture near the robot.
        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        ep_arm.moveto(x=0, y=raised_y).wait_for_completed()
        return ep_robot
    finally:
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
):
    """Place a tower by extending high, lowering, releasing, then returning home."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper

        # Keep the arm high while moving forward, then lower to place.
        # ep_arm.moveto(x=0, y=raised_y).wait_for_completed() 
        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()

        ep_gripper.open(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        move_arm_to_default(ep_robot)
        return ep_robot
    finally:
        if owns_robot:
            ep_robot.close()
