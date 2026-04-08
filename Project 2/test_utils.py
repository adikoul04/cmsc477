import argparse

from tower_utils import (
    DEFAULT_DETECT_CONF,
    DEFAULT_MODEL_PATH,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_SN,
    connect_robot,
    go_to_tower,
    load_model,
    pick_up_tower,
    place_down_tower,
    start_camera_stream,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test tower utility functions from the command line."
    )
    parser.add_argument(
        "command",
        choices=["1", "2", "3"],
        help="Use '1' to pick up the tower, '2' to place it down, or '3' to go to a tower.",
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
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YOLO weights used by go_to_tower().",
    )
    parser.add_argument(
        "--resolution",
        default="360p",
        choices=["360p", "720p"],
        help="Camera stream resolution used by go_to_tower().",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ep_robot = connect_robot(
        conn_type=args.conn_type,
        robot_ip=args.robot_ip,
        sn=args.sn,
    )
    model = load_model(args.model_path)
    ep_camera = start_camera_stream(ep_robot, resolution=args.resolution)

    try:
        if args.command == "1":
            print("Running pick_up_tower()")
            pick_up_tower(ep_robot=ep_robot)
        else:
            if args.command == "2":
                print("Running place_down_tower()")
                place_down_tower(ep_robot=ep_robot)
            else:
                print("Running go_to_tower()")
                go_to_tower(
                    ep_robot=ep_robot,
                    model=model,
                    ep_camera=ep_camera,
                    conf_thresh=DEFAULT_DETECT_CONF,
                    show=True,
                )
    finally:
        ep_camera.stop_video_stream()
        ep_robot.close()


if __name__ == "__main__":
    main()
