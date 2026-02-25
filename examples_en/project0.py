import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
import robomaster

class AprilTagDetector:
    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(family, threads)

    def find_tags(self, frame_gray):
        detections = self.detector.detect(frame_gray, estimate_tag_pose=True,
            camera_params=self.camera_params, tag_size=self.marker_size_m)
        return detections

def get_pose_apriltag_in_camera_frame(detection):
    R_ca = detection.pose_R
    t_ca = detection.pose_t
    return t_ca.flatten(), R_ca

def draw_detections(frame, detections):
    for detection in detections:
        pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        top_left = tuple(pts[0][0])  # First corner
        top_right = tuple(pts[1][0])  # Second corner
        bottom_right = tuple(pts[2][0])  # Third corner
        bottom_left = tuple(pts[3][0])  # Fourth corner
        cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)

        if len(detections) > 0:
            assert len(detections) == 1 # Assume there is only one AprilTag to track
            detection = detections[0]

            t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
            print('t_ca', t_ca)
            move_to_tag(t_ca)

        draw_detections(img, detections)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
def move_to_tag(t_ca):
    x, y, measured_position = t_ca
    turn_angle = 0
    desired_position = 0.4
    K = 1.8
    K_turn = -150
    error_z = max(0, measured_position - desired_position)
    velocity = K * error_z
    if (x < -0.1):
        # turn_angle = 15
        print("Turn left")
        ep_chassis.move(x=0, y=0, z=K_turn*x, z_speed=90).wait_for_completed()
    elif (x > 0.1):
        # turn_angle = -15
        print("Turn right")
        ep_chassis.move(x=0, y=0, z=K_turn*x, z_speed=90).wait_for_completed()
    else:
        ep_chassis.drive_speed(x=velocity, y=0, z=0, timeout=5)
        time.sleep(0.5)
        print("Moving forward")

    print(f"X-Error: {x} Z-Error: {error_z}")
    print("\n")




if __name__ == '__main__':
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    robomaster.config.ROBOT_IP_STR = "192.168.50.117"
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100RC")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
    marker_size_m = 0.153 # Size of the AprilTag in meters
    apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

    try:
        detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Waiting for robomaster shutdown')
        ep_camera.stop_video_stream()
        ep_robot.close()