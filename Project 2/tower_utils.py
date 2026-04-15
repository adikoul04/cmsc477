from dataclasses import dataclass
from pathlib import Path
from queue import Empty
import math
import time

import robomaster
from robomaster import robot
import cv2
from ultralytics import YOLO
from robomaster import camera

from capture_robot_images import DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN


_PROJECT_DIR = Path(__file__).resolve().parent
_MODEL_CANDIDATES = (
    _PROJECT_DIR / "cmsc477_yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
    Path(__file__).resolve().parents[1] / "runs" / "detect" / "train5" / "weights" / "best.pt",
)
DEFAULT_MODEL_PATH = r"C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt"
DEFAULT_DETECT_CONF = 0.45
DEFAULT_STOP_METRIC = "top_y"
DEFAULT_DESIRED_H_PX = 170.0
DEFAULT_TARGET_TOP_Y_RATIO = 0.72
DEFAULT_ALIGN_CENTER_TOL_PX = 24.0
DEFAULT_ALIGN_HEIGHT_TOL_PX = 16.0
DEFAULT_ALIGN_TOP_TOL_PX = 18.0
DEFAULT_K_FORWARD = 0.0028
DEFAULT_K_LATERAL = 0.0038
DEFAULT_K_YAW = 0.12
DEFAULT_LATERAL_SIGN = -1.0
DEFAULT_MAX_V = 0.16
DEFAULT_MAX_YAW_DPS = 45.0
DEFAULT_SERVO_STEP_S = 0.12

DEFAULT_ARM_X = 180
DEFAULT_APPROACH_Y = 30
DEFAULT_LOWER_Y = -50
DEFAULT_RAISED_Y = 100
DEFAULT_GRIPPER_POWER = 50
DEFAULT_GRIPPER_WAIT_SECONDS = 1.0


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


def go_to_tower(
    ep_robot,
    model,
    ep_camera=None,
    ep_chassis=None,
    target_class=None,
    conf_thresh=DEFAULT_DETECT_CONF,
    stop_metric=DEFAULT_STOP_METRIC,
    desired_h_px=DEFAULT_DESIRED_H_PX,
    target_top_y_ratio=DEFAULT_TARGET_TOP_Y_RATIO,
    center_tol_px=DEFAULT_ALIGN_CENTER_TOL_PX,
    height_tol_px=DEFAULT_ALIGN_HEIGHT_TOL_PX,
    top_y_tol_px=DEFAULT_ALIGN_TOP_TOL_PX,
    k_forward=DEFAULT_K_FORWARD,
    k_lateral=DEFAULT_K_LATERAL,
    k_yaw=DEFAULT_K_YAW,
    lateral_sign=DEFAULT_LATERAL_SIGN,
    max_v=DEFAULT_MAX_V,
    max_yaw_dps=DEFAULT_MAX_YAW_DPS,
    step_s=DEFAULT_SERVO_STEP_S,
    timeout_s=20.0,
    pose_tracker=None,
    selection_mode="conf",
    show=False,
):
    owns_camera = ep_camera is None
    valid_stop_metrics = {"top_y"}
    if stop_metric not in valid_stop_metrics:
        raise ValueError(f"Unsupported stop_metric '{stop_metric}'. Use 'top_y'.")

    if ep_camera is None:
        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    if ep_chassis is None:
        ep_chassis = ep_robot.chassis

    stable = 0
    center_stable = 0
    selected = None
    t0 = time.time()

    try:
        move_arm_to_top(ep_robot)
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError("Timed out while approaching a tower.")

            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                continue

            if frame is None:
                continue

            detections = get_detections(
                model=model,
                frame=frame,
                conf_thresh=conf_thresh,
                target_class=target_class,
            )
            if not detections:
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=step_s)
                continue

            frame_center_x = frame.shape[1] / 2.0
            selected = select_detection(
                detections=detections,
                selection_mode=selection_mode,
                frame_center_x=frame_center_x,
            )
            y_top_px = selected.cy - selected.h / 2.0
            err_x_px = selected.cx - frame_center_x

            target_top_y_px = target_top_y_ratio * frame.shape[0]
            err_forward_px = target_top_y_px - y_top_px
            forward_tol_px = top_y_tol_px
            err_label = "err_top"

            centered = abs(err_x_px) <= center_tol_px
            if centered:
                center_stable += 1
            else:
                center_stable = 0

            allow_forward = center_stable >= 2

            if not allow_forward:
                v_forward = 0.0
                v_yaw = clamp(-k_yaw * err_x_px, -max_yaw_dps, max_yaw_dps)
            else:
                v_forward = clamp(k_forward * err_forward_px, -max_v, max_v)
                v_yaw = 0.0

            ep_chassis.drive_speed(x=v_forward, y=0.0, z=v_yaw, timeout=step_s)
            if pose_tracker is not None:
                pose_tracker.integrate_turn(v_yaw * step_s)
                pose_tracker.integrate_body_motion(v_forward * step_s, 0.0)

            if abs(err_x_px) <= center_tol_px and abs(err_forward_px) <= forward_tol_px:
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
                cv2.line(dbg, (int(frame_center_x), 0), (int(frame_center_x), dbg.shape[0] - 1), (0, 255, 255), 1)
                cv2.putText(
                    dbg,
                    f"err_x={err_x_px:+.1f}px yaw={v_yaw:+.1f}dps {err_label}={err_forward_px:+.1f}px center={center_stable} stable={stable}/4",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("go_to_tower", dbg)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            if stable >= 4:
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
                return selected
    finally:
        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.1)
        if owns_camera:
            ep_camera.stop_video_stream()


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
    """Lower the arm, grip the tower, and raise it back up."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper

        move_arm_to_top(ep_robot, arm_x=arm_x, raised_y=raised_y)
        ep_arm.moveto(x=arm_x, y=approach_y).wait_for_completed()
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()

        ep_gripper.close(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
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
    """Lower the arm, release the tower, and raise the arm back up."""
    owns_robot = ep_robot is None
    if owns_robot:
        ep_robot = connect_robot(conn_type=conn_type, robot_ip=robot_ip, sn=sn)

    try:
        ep_arm = ep_robot.robotic_arm
        ep_gripper = ep_robot.gripper

        move_arm_to_top(ep_robot, arm_x=arm_x, raised_y=raised_y)
        ep_arm.moveto(x=arm_x, y=approach_y).wait_for_completed()
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()

        ep_gripper.open(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        move_arm_to_top(ep_robot, arm_x=arm_x, raised_y=raised_y)
        return ep_robot
    finally:
        if owns_robot:
            ep_robot.close()
