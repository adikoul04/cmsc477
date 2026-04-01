import argparse

from tower_utils import (
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
    connect_robot,
    pick_up_tower,
    place_down_tower,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test tower utility functions from the command line."
    )
    parser.add_argument(
        "command",
        choices=["1", "2"],
        help="Use '1' to pick up the tower or '2' to place it down.",
    )
    parser.add_argument(
        "--conn-type",
        default="sta",
        choices=["sta", "ap"],
        help="Robot connection mode.",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help="Robot IP address used in STA mode.",
    )
    parser.add_argument(
        "--sn",
        default=DEFAULT_ROBOT_SN,
        help="Robot serial number.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ep_robot = connect_robot(
        conn_type=args.conn_type,
        robot_ip=args.robot_ip,
        sn=args.sn,
    )

    try:
        if args.command == "1":
            print("Running pick_up_tower()")
            pick_up_tower(ep_robot=ep_robot)
        else:
            print("Running place_down_tower()")
            place_down_tower(ep_robot=ep_robot)
    finally:
        ep_robot.close()


if __name__ == "__main__":
    main()
