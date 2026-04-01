from pathlib import Path
import argparse
import time
import traceback

import cv2
from queue import Empty
from robomaster import camera
from robomaster import robot
import robomaster


DEFAULT_ROBOT_IP = "192.168.50.117"
DEFAULT_ROBOT_SN = "3JKCH8800100RC"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the RoboMaster camera and save images on demand."
    )
    parser.add_argument(
        "--output-dir",
        default="captures",
        help="Directory where captured images will be saved.",
    )
    parser.add_argument(
        "--prefix",
        default="capture",
        help="Filename prefix for saved images.",
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
        "--resolution",
        default="360p",
        choices=["360p", "720p"],
        help="Camera stream resolution.",
    )
    return parser.parse_args()


def resolve_resolution(name):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.conn_type == "sta":
        robomaster.config.ROBOT_IP_STR = args.robot_ip

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=args.conn_type, sn=args.sn)
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(
        display=False,
        resolution=resolve_resolution(args.resolution),
    )

    capture_count = 0

    print("Live camera preview started.")
    print("Press 'c' to capture an image.")
    print("Press 'q' to quit.")
    print(f"Saving images to: {output_dir}")

    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                time.sleep(0.01)
                continue

            if frame is None:
                continue

            cv2.imshow("RoboMaster Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{args.prefix}_{timestamp}_{capture_count:03d}.jpg"
                image_path = output_dir / filename

                if cv2.imwrite(str(image_path), frame):
                    capture_count += 1
                    print(f"Saved: {image_path}")
                else:
                    print(f"Failed to save: {image_path}")
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        print("Shutting down RoboMaster camera stream.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()


if __name__ == "__main__":
    main()
