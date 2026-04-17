#!/usr/bin/env python3
"""Command-line tester for RoboMaster chassis, arm, and gripper movement.

Examples:
  python "Project 2/test_robot_movement.py" --rotate 90
  python "Project 2/test_robot_movement.py" --move-position 0.3 -0.2
  python "Project 2/test_robot_movement.py" --move-velocity 0.15 0.0
  python "Project 2/test_robot_movement.py" --move-arm-absolute 180 30
  python "Project 2/test_robot_movement.py" --move-arm-relative 20 -10
  python "Project 2/test_robot_movement.py" --gripper-open
  python "Project 2/test_robot_movement.py" --gripper-close
"""

import argparse

from tower_utils import DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN, connect_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test RoboMaster movement with one explicit action per run."
    )

    connection = parser.add_argument_group("connection")
    connection.add_argument("--conn-type", default="sta", choices=["sta", "ap"], help="Robot connection mode.")
    connection.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Robot IP address for STA mode.")
    connection.add_argument("--sn", default=DEFAULT_ROBOT_SN, help="Robot serial number.")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--rotate", type=float, metavar="DEG", help="Rotate the chassis by DEG degrees.")
    action.add_argument(
        "--move-position",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Move the chassis by relative X/Y position in meters.",
    )
    action.add_argument(
        "--move-velocity",
        nargs=2,
        type=float,
        metavar=("VX", "VY"),
        help="Drive the chassis with X/Y velocity in m/s for a fixed timeout.",
    )
    action.add_argument(
        "--move-arm-absolute",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Move the robotic arm to an absolute X/Y position in mm.",
    )
    action.add_argument(
        "--move-arm-relative",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        help="Move the robotic arm relative to its current position in mm.",
    )
    action.add_argument("--gripper-open", action="store_true", help="Open the gripper.")
    action.add_argument("--gripper-close", action="store_true", help="Close the gripper.")

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--xy-speed", type=float, default=0.20, help="Speed used for chassis position moves.")
    tuning.add_argument(
        "--velocity-timeout",
        type=float,
        default=2.0,
        help="Timeout used for chassis velocity moves.",
    )
    tuning.add_argument(
        "--gripper-power",
        type=int,
        default=50,
        help="Power used when opening or closing the gripper.",
    )
    tuning.add_argument(
        "--gripper-pause",
        type=float,
        default=1.0,
        help="Seconds to wait after gripper open/close before pausing it.",
    )
    tuning.add_argument(
        "--rotate-speed",
        type=float,
        default=45.0,
        help="Degrees/second used when rotating the chassis.",
    )

    return parser.parse_args()


def run_action(ep_robot, args: argparse.Namespace) -> None:
    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper

    if args.rotate is not None:
        degrees = args.rotate
        print(f"Rotating chassis by {degrees:.2f} degrees at {args.rotate_speed:.2f} deg/s")
        ep_chassis.move(x=0.0, y=0.0, z=degrees, z_speed=abs(args.rotate_speed)).wait_for_completed()
        return

    if args.move_position is not None:
        x_m, y_m = args.move_position
        print(f"Moving chassis by x={x_m:.3f} m, y={y_m:.3f} m at {args.xy_speed:.2f} m/s")
        ep_chassis.move(x=x_m, y=y_m, z=0.0, xy_speed=args.xy_speed).wait_for_completed()
        return

    if args.move_velocity is not None:
        vx, vy = args.move_velocity
        print(
            f"Driving chassis with vx={vx:.3f} m/s, vy={vy:.3f} m/s "
            f"for {args.velocity_timeout:.2f} s"
        )
        ep_chassis.drive_speed(x=vx, y=vy, z=0.0, timeout=args.velocity_timeout)
        return

    if args.move_arm_absolute is not None:
        x_mm, y_mm = args.move_arm_absolute
        print(f"Moving arm to absolute position x={x_mm:.1f} mm, y={y_mm:.1f} mm")
        ep_arm.moveto(x=x_mm, y=y_mm).wait_for_completed()
        return

    if args.move_arm_relative is not None:
        dx_mm, dy_mm = args.move_arm_relative
        print(f"Moving arm relative by dx={dx_mm:.1f} mm, dy={dy_mm:.1f} mm")
        ep_arm.move(x=dx_mm, y=dy_mm).wait_for_completed()
        return

    if args.gripper_open:
        print(f"Opening gripper at power {args.gripper_power}")
        ep_gripper.open(power=args.gripper_power)
        ep_gripper.pause()
        return

    if args.gripper_close:
        print(f"Closing gripper at power {args.gripper_power}")
        ep_gripper.close(power=args.gripper_power)
        ep_gripper.pause()
        return

    raise RuntimeError("No action was selected.")


def main() -> None:
    args = parse_args()
    ep_robot = connect_robot(conn_type=args.conn_type, robot_ip=args.robot_ip, sn=args.sn)

    try:
        run_action(ep_robot, args)
        print("Action complete.")
    finally:
        ep_robot.close()


if __name__ == "__main__":
    main()
