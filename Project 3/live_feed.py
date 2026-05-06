#!/usr/bin/env python3
"""Live video feed with YOLO and AprilTag overlays.

Usage: run this script while the robot is powered and on the same Wi‑Fi.
Press 'q' in the window to quit.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from queue import Empty

import cv2
import numpy as np
import pupil_apriltags
from robomaster import camera as rm_camera
from robomaster import robot
import robomaster
from ultralytics import YOLO


# Load project config (absolute path)
CONFIG_PATH = "./config.py"
spec = importlib.util.spec_from_file_location("proj3_config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


CLASS_NAMES = {
    config.CLASS_CONE: "cone",
    config.CLASS_BOX: "box",
    config.CLASS_SMALL_BRICK: "small_brick",
    config.CLASS_LARGE_BRICK: "large_brick",
}


def make_apriltag_detector(K, family: str = "tag36h11", threads: int = 2, marker_size_m: float = 0.16):
    camera_params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    detector = pupil_apriltags.Detector(families=family, nthreads=threads,
                                        quad_decimate=2.0, quad_sigma=0.0,
                                        refine_edges=1, decode_sharpening=0.25, debug=0)

    def find_tags(gray):
        return detector.detect(gray, estimate_tag_pose=True,
                               camera_params=camera_params, tag_size=marker_size_m)

    return find_tags


def draw_yolo_boxes(frame, boxes):
    """boxes: list of (x1,y1,x2,y2,cls,conf)"""
    for x1, y1, x2, y2, cls, conf in boxes:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_apriltags(frame, detections):
    for det in detections:
        corners = np.array(det.corners, dtype=np.int32).reshape((-1, 2))
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        # draw cross lines
        cv2.line(frame, tuple(corners[0]), tuple(corners[2]), (0, 0, 255), 1)
        cv2.line(frame, tuple(corners[1]), tuple(corners[3]), (0, 0, 255), 1)
        # id near center
        cx = int(det.center[0]); cy = int(det.center[1])
        cv2.putText(frame, f"id:{int(det.tag_id)}", (cx + 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def extract_yolo_boxes(result) -> list:
    """Robust extraction of boxes from an ultralytics Results object."""
    boxes_out = []
    try:
        r = result
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            # try a couple of common APIs
            try:
                xy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
            except Exception:
                try:
                    xy = r.boxes.xyxy.numpy()
                    cls = r.boxes.cls.numpy().astype(int)
                    conf = r.boxes.conf.numpy()
                except Exception:
                    return []

            for (x1, y1, x2, y2), c, cf in zip(xy, cls, conf):
                boxes_out.append((int(x1), int(y1), int(x2), int(y2), int(c), float(cf)))
    except Exception:
        return []
    return boxes_out


def main():
    # Robot connection
    ep_robot = robot.Robot()
    # make sure user config is applied
    try:
        robomaster.config.ROBOT_IP_STR = str(config.ROBOT_IP)
    except Exception:
        pass
    ep_robot.initialize(conn_type="sta", sn=str(getattr(config, 'ROBOT_SN', '')))

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_720P)

    # Load YOLO model
    print(f"Loading YOLO model from {config.MODEL_PATH}")
    model = YOLO(str(config.MODEL_PATH))

    # AprilTag detector
    find_tags = make_apriltag_detector(config.K_CAM, family=config.TAG_FAMILY, marker_size_m=config.TAG_SIZE_M)

    window_name = "Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
            except Empty:
                time.sleep(0.01)
                continue
            if frame is None:
                time.sleep(0.01)
                continue

            # YOLO expects RGB in many builds; convert to be safe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, verbose=False)
            # Use first result (single image)
            boxes = []
            if len(results) > 0:
                boxes = extract_yolo_boxes(results[0])

            # Draw YOLO boxes
            draw_yolo_boxes(frame, boxes)

            # AprilTag detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            try:
                detections = find_tags(gray)
            except Exception:
                detections = []

            draw_apriltags(frame, detections)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down")
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
