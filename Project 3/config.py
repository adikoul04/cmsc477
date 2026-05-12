from pathlib import Path

import numpy as np


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
# MODEL_CANDIDATES = (
#     PROJECT_DIR / "cmsc477_yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
#     Path(__file__).resolve().parents[1] / "runs" / "detect" / "train5" / "weights" / "best.pt",
# )
# DEFAULT_MODEL_PATH = next(
#     (candidate for candidate in MODEL_CANDIDATES if candidate.exists()),
#     MODEL_CANDIDATES[-1],
# )
# DEFAULT_MODEL_PATH = PROJECT_DIR / "final_yolo.pt"
# DEFAULT_MODEL_PATH = PROJECT_DIR / "last.pt"
DEFAULT_MODEL_PATH = PROJECT_DIR / "new_best.pt"



# Robot connection
# DEFAULT_ROBOT_IP = "192.168.50.121"
# DEFAULT_ROBOT_SN = "3JKCH8800100UB"

# Robot 7
# DEFAULT_ROBOT_IP = "192.168.50.117"
# DEFAULT_ROBOT_SN = "3JKCH8800100RC"

# Robot 5
# DEFAULT_ROBOT_IP = "192.168.50.115"
# DEFAULT_ROBOT_SN = "3JKCH8800101CE"

# Robot 7
DEFAULT_ROBOT_IP = "192.168.50.119"
DEFAULT_ROBOT_SN = "3JKCH88001012K"
ROBOT_IP = DEFAULT_ROBOT_IP
ROBOT_SN = DEFAULT_ROBOT_SN
MODEL_PATH = DEFAULT_MODEL_PATH


# Arena geometry
ARENA_W_M = 3.048
ARENA_H_M = 3.048
SAFE_BOUNDARY_MARGIN_M = 0.20
SIDE1_Y_LIMIT = 1.0
SIDE2_Y_START = 2.0


# Camera and AprilTag localization
K_CAM = np.array(
    [[314.0, 0.0, 320.0],
     [0.0, 314.0, 180.0],
     [0.0, 0.0, 1.0]],
    dtype=float,
)

# Camera extrinsics (camera optical frame -> robot frame).
# AprilTag/OpenCV camera frame convention:
# - +x points right in the image
# - +y points down in the image
# - +z points forward out of the camera
#
# Robot frame convention assumed by the navigation code:
# - +x forward
# - +y left
# - +z up
#
# Measure these on the real robot:
# - CAM_OFFSET_X_M: forward distance from robot origin to camera center.
# - CAM_OFFSET_Y_M: left/right distance from robot origin to camera center.
# - CAM_OFFSET_Z_M: vertical distance from robot origin to camera center.
# - CAM_ROLL_DEG: roll of the mounted camera relative to its nominal forward-facing mount.
# - CAM_PITCH_DEG: downward tilt of the mounted camera. Negative means pitched downward.
# - CAM_YAW_DEG: yaw offset of the mounted camera relative to robot forward.
#
# A practical way to measure them:
# 1. Pick a robot origin, usually the chassis center on the ground plane.
# 2. Measure the camera lens center position relative to that origin in meters.
# 3. Measure pitch with a phone inclinometer or by fitting against known points.
# 4. Measure yaw/roll if the camera is not mounted square to the chassis.
CAM_OFFSET_X_M = 0.0
CAM_OFFSET_Y_M = 0.0
CAM_OFFSET_Z_M = 0.32
CAM_ROLL_DEG = 0.0
CAM_PITCH_DEG = -19.6
CAM_YAW_DEG = 0.0


def _rotx(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=float,
    )


def _roty(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=float,
    )


def _rotz(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


# Base alignment from the OpenCV optical frame to the robot body frame for a
# forward-facing camera with zero mount offsets:
# - camera +z (forward) -> robot +x (forward)
# - camera +x (right)   -> robot -y (right is negative-left)
# - camera +y (down)    -> robot -z (down is negative-up)
CAM_R_ROBOT_FROM_CAMERA_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)

# Mount attitude relative to the nominal forward-facing camera mount.
CAM_R_MOUNT = (
    _rotz(np.deg2rad(CAM_YAW_DEG))
    @ _rotx(np.deg2rad(CAM_PITCH_DEG))
    @ _roty(np.deg2rad(CAM_ROLL_DEG))
)

CAM_R_ROBOT_FROM_CAMERA = CAM_R_ROBOT_FROM_CAMERA_BASE @ CAM_R_MOUNT
CAM_T_ROBOT_FROM_CAMERA_M = np.array(
    [CAM_OFFSET_X_M, CAM_OFFSET_Y_M, CAM_OFFSET_Z_M],
    dtype=float,
)
T_ROBOT_FROM_CAMERA = np.eye(4, dtype=float)
T_ROBOT_FROM_CAMERA[:3, :3] = CAM_R_ROBOT_FROM_CAMERA
T_ROBOT_FROM_CAMERA[:3, 3] = CAM_T_ROBOT_FROM_CAMERA_M

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.2


# Arena landmarks
RECHARGE_TAG_IDS = {34, 38}
SMALL_GOAL_TAG_IDS = {11, 41}
LARGE_GOAL_TAG_IDS = {45, 19}
GOAL_TAG_SAME_BLOCK_PAIRS = (
    (11, 19),
    (45, 41),
)
ALL_LANDMARK_TAG_IDS = RECHARGE_TAG_IDS | SMALL_GOAL_TAG_IDS | LARGE_GOAL_TAG_IDS


# YOLO class IDs
CLASS_CONE = 0
CLASS_BOX = 1
CLASS_SMALL_BRICK = 3
CLASS_LARGE_BRICK = 2


# Battery management
BATTERY_START_PCT = 60.0
BATTERY_LARGE_BRICK_COST = 40.0
BATTERY_SMALL_BRICK_COST = 30.0
BATTERY_RECHARGE_LEVEL = 100.0
BATTERY_RESERVE_PCT = 5.0


# Tower detection
DEFAULT_DETECT_CONF = 0.45
DEFAULT_STOP_METRIC = "top_y"
DEFAULT_DESIRED_H_PX = 170.0
DEFAULT_TARGET_TOP_Y_RATIO = 0.70


# Tower alignment
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


# Navigation and sweep behavior
MOVE_SPEED_MPS = 1.0
TURN_SPEED_DPS = 35.0
SWEEP_STEP_DEG = 15.0
SWEEP_SETTLE_S = 0.6


# Obstacle avoidance
OBS_CLEAR_DIST_M = 0.3
OBS_APPROACH_DIST_M = 0.35
OBS_SLIDE_SPEED_MPS = 0.12
OBS_FWD_SPEED_MPS = 0.14
OBSTACLE_1_FALLBACK_POSITION = (0.9, 1.7)
OBSTACLE_2_FALLBACK_POSITION = (2.1, 1.8)
DOCK_FALLBACK_POSITION = (2.46, 0.48)
RECHARGE_ALIGNMENT_TAG_ID = 34


# Recharge approach
RECHARGE_APPROACH_DIST_M = 0.30
RECHARGE_STOP_DIST_M = 0.05
RECHARGE_HOLD_S = 5.0


# AprilTag servo
TAG_SERVO_CENTER_TOL_PX = 20.0
TAG_SERVO_DIST_TOL_M = 0.05
TAG_SERVO_K_YAW = 0.08
TAG_SERVO_K_FWD = 0.6
TAG_SERVO_MAX_YAW_DPS = 35.0
TAG_SERVO_MAX_V = 0.18
TAG_SERVO_STEP_S = 0.12
TAG_DIST_TOL_M = TAG_SERVO_DIST_TOL_M


# Brick servo
BRICK_SERVO_CENTER_TOL_PX = 24.0
BRICK_SERVO_TOP_Y_RATIO = 0.70
BRICK_SERVO_TOP_TOL_PX = 18.0
BRICK_SERVO_K_FWD = 0.005
BRICK_SERVO_K_LAT = 0.010
BRICK_SERVO_MAX_V = 0.16
BRICK_SERVO_STEP_S = 0.20
BRICK_SERVO_STABLE_THRESH = 4


# Loading dock color segmentation
# PURPLE_HSV_LO = np.array([125, 60, 40], dtype=np.uint8)
# PURPLE_HSV_HI = np.array([155, 255, 255], dtype=np.uint8)
# PURPLE_MIN_AREA_PX = 1500


# Robotic arm positions
DEFAULT_ARM_X = 135
DEFAULT_ARM_Y = 45
DEFAULT_APPROACH_Y = 30
DEFAULT_LOWER_Y = -30
DEFAULT_RAISED_Y = 100


# Gripper control
DEFAULT_GRIPPER_POWER = 50
DEFAULT_GRIPPER_WAIT_SECONDS = 1.0
