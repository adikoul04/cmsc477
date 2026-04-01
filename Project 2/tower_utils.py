import time

import robomaster
from robomaster import robot

from capture_robot_images import DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN


DEFAULT_ARM_X = 180
DEFAULT_APPROACH_Y = 30
DEFAULT_LOWER_Y = -50
DEFAULT_RAISED_Y = 100
DEFAULT_GRIPPER_POWER = 50
DEFAULT_GRIPPER_WAIT_SECONDS = 1.0


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

        ep_arm.moveto(x=arm_x, y=approach_y).wait_for_completed()
        ep_arm.moveto(x=arm_x, y=lower_y).wait_for_completed()

        ep_gripper.open(power=gripper_power)
        time.sleep(grip_wait_seconds)
        ep_gripper.pause()

        ep_arm.moveto(x=arm_x, y=raised_y).wait_for_completed()
        return ep_robot
    finally:
        if owns_robot:
            ep_robot.close()
