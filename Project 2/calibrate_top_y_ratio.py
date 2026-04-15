#!/usr/bin/env python3
"""Estimate a good top-y ratio target for visual servo stopping.

This script mirrors the live detection workflow in bounding_box_capture.py and
prints top-of-bounding-box ratio values from RoboMaster camera frames.
"""

import argparse
from collections import deque
from pathlib import Path
from queue import Empty
import statistics

import cv2
from ultralytics import YOLO

from robomaster import camera

from tower_utils import DEFAULT_MODEL_PATH, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_SN, connect_robot, move_arm_to_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate top-y ratio from live RoboMaster detections")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YOLO model weights.",
    )
    parser.add_argument("--conn-type", default="sta", choices=["sta", "ap"], help="Robot connection mode.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Robot IP address for STA mode.")
    parser.add_argument("--sn", default=DEFAULT_ROBOT_SN, help="Robot serial number.")
    parser.add_argument("--resolution", default="360p", choices=["360p", "720p"], help="Camera stream resolution.")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold.")
    parser.add_argument("--target-class", type=int, default=None, help="Optional class id to filter detections.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Number of recent ratio samples for rolling median/mean.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print rolling stats every N accepted frames.",
    )
    return parser.parse_args()


def resolve_resolution(name: str):
    if name == "720p":
        return camera.STREAM_720P
    return camera.STREAM_360P


def pick_detection(result, target_class):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = None
    best_conf = -1.0

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if target_class is not None and cls != target_class:
            continue
        if conf > best_conf:
            best_conf = conf
            best = box

    return best


def main() -> None:
    args = parse_args()

    model = YOLO(args.model_path)

    ep_robot = connect_robot(conn_type=args.conn_type, robot_ip=args.robot_ip, sn=args.sn)
    ep_camera = ep_robot.camera
    move_arm_to_default(ep_robot)
    ep_camera.start_video_stream(display=False, resolution=resolve_resolution(args.resolution))

    ratios = []
    rolling = deque(maxlen=max(1, args.window_size))
    accepted_frames = 0

    print("Live calibration started.")
    print("Place the robot at your desired pickup stop distance and angle.")
    print("Press 'q' to quit.")

    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                continue

            if frame is None:
                continue

            result = model.predict(source=frame, show=False, conf=args.conf, verbose=False)[0]
            selected = pick_detection(result=result, target_class=args.target_class)

            if selected is not None:
                xyxy = selected.xyxy.cpu().numpy().flatten()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                frame_h = frame.shape[0]

                top_y_ratio = y1 / max(frame_h, 1)
                ratios.append(top_y_ratio)
                rolling.append(top_y_ratio)
                accepted_frames += 1

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color=(0, 0, 255),
                    thickness=2,
                )

                label = f"top_y_ratio={top_y_ratio:.3f} conf={float(selected.conf.item()):.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if accepted_frames % max(1, args.print_every) == 0:
                    rolling_median = statistics.median(rolling)
                    rolling_mean = statistics.fmean(rolling)
                    print(
                        f"samples={accepted_frames} "
                        f"last={top_y_ratio:.3f} "
                        f"rolling_median={rolling_median:.3f} "
                        f"rolling_mean={rolling_mean:.3f}"
                    )
            else:
                cv2.putText(
                    frame,
                    "No detection",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("top_y_ratio_calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        ep_robot.close()

    if not ratios:
        print("No valid detections collected.")
        return

    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)

    def percentile(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return ratios_sorted[idx]

    print("\nCalibration summary")
    print(f"samples={n}")
    print(f"median={statistics.median(ratios):.3f}")
    print(f"mean={statistics.fmean(ratios):.3f}")
    print(f"p25={percentile(25):.3f} p75={percentile(75):.3f}")
    print(f"p10={percentile(10):.3f} p90={percentile(90):.3f}")
    print(
        "Recommended --target-top-y-ratio is usually the median or slightly lower "
        "(about 0.02 to 0.05) if you want a bit more forward approach before stopping."
    )


if __name__ == "__main__":
    main()
