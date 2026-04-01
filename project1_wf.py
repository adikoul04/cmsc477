import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera
import robomaster

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

matrix = [[0 for _ in range(11)] for _ in range(9)]
cur_r, cur_c = 3, 0  # starting grid cell (row, col)
heading_deg = 0  # 0 = +x (right), 90 = +y (down), 180 = -x, 270 = -y (up)
matrix[cur_r][cur_c] = 1  # mark starting position as visited
status_text = "idle"

def mark_visited():
    global matrix, cur_r, cur_c
    if 0 <= cur_r < len(matrix) and 0 <= cur_c < len(matrix[0]):
        matrix[cur_r][cur_c] = 1

def mark_tag_front():
    global matrix, cur_r, cur_c, heading_deg
    tr, tc = cur_r, cur_c

    # Mark the estimated tag location in front of the robot, not the rover's current cell.
    if heading_deg % 360 == 0:
        tc += 1
    elif heading_deg % 360 == 90:
        tr += 1
    elif heading_deg % 360 == 180:
        tc -= 1
    else:  # 270
        tr -= 1

    if 0 <= tr < len(matrix) and 0 <= tc < len(matrix[0]):
        matrix[tr][tc] = 'X'

def print_matrix():
    for r in range(len(matrix)):
        print(' '.join(str(x) for x in matrix[r]))
    print('\n')

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


def draw_minimap_overlay(frame):
    """Draw a small grid minimap on the top-right of the frame showing OCC, visited and robot."""
    h, w = frame.shape[:2]
    map_h = 120
    map_w = 165
    scale_x = map_w // len(matrix[0])
    scale_y = map_h // len(matrix)
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8) + 50

    # draw visited cells
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            v = matrix[r][c]
            if v == 1:
                cv2.rectangle(
                    map_img,
                    (c * scale_x + 1, r * scale_y + 1),
                    ((c + 1) * scale_x - 2, (r + 1) * scale_y - 2),
                    (0, 200, 0),
                    thickness=-1,
                )
            elif v == 'X':
                cv2.rectangle(
                    map_img,
                    (c * scale_x + 1, r * scale_y + 1),
                    ((c + 1) * scale_x - 2, (r + 1) * scale_y - 2),
                    (0, 0, 200),
                    thickness=-1,
                )

    # draw grid lines
    for r in range(len(matrix) + 1):
        y = r * scale_y
        cv2.line(map_img, (0, y), (map_w, y), (120, 120, 120), 1)
    for c in range(len(matrix[0]) + 1):
        x = c * scale_x
        cv2.line(map_img, (x, 0), (x, map_h), (120, 120, 120), 1)

    # draw robot as circle
    rr = int((cur_r + 0.5) * scale_y)
    cc = int((cur_c + 0.5) * scale_x)
    cv2.circle(map_img, (cc, rr), min(scale_x, scale_y) // 3, (0, 255, 255), -1)

    # put status text above minimap
    cv2.putText(map_img, status_text, (5, map_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # overlay onto top-right of the frame
    x0 = w - map_w - 10
    y0 = 10
    roi = frame[y0 : y0 + map_h, x0 : x0 + map_w]
    if roi.shape[0] == map_img.shape[0] and roi.shape[1] == map_img.shape[1]:
        blended = cv2.addWeighted(roi, 0.4, map_img, 0.6, 0)
        frame[y0 : y0 + map_h, x0 : x0 + map_w] = blended


def overlay_status_text(frame, txt):
    cv2.putText(frame, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 

def move_to_tag(t_ca, ep_chassis, ep_camera, apriltag):
    global cur_r, cur_c, heading_deg, status_text
    x, y, measured_position = t_ca
    desired_position = 0.4
    K_turn = 150
    
    # Center the tag
    ok = center_tag_until_aligned(ep_chassis, ep_camera, apriltag, tol_px=10, timeout_s=8.0)
    if not ok:
        print("Centering failed or timed out; aborting approach")
        return

    # After centering, refresh detection to get current measured_position
    try:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        dets = apriltag.find_tags(gray)
    except Exception:
        dets = []

    if dets:
        # pick closest
        det = min(dets, key=lambda d: np.linalg.norm(np.array(d.pose_t).reshape(3)))
        t_ca, R_ca = get_pose_apriltag_in_camera_frame(det)
        x, y, measured_position = t_ca
    else:
        print("No detection after centering; aborting approach")
        return

    # Move forward toward the tag in discrete cube-sized steps until we reach desired distance
    # Each forward step is one cube (0.266 m) in robot-forward direction.
    while measured_position > desired_position:
        ep_chassis.move(x=0.266, y=0, z=0, xy_speed=0.7).wait_for_completed()
        # update grid cell according to current heading
        if heading_deg % 360 == 0:
            cur_c += 1
        elif heading_deg % 360 == 90:
            cur_r += 1
        elif heading_deg % 360 == 180:
            cur_c -= 1
        else:  # 270
            cur_r -= 1

        # clamp to bounds
        cur_r = max(0, min(len(matrix) - 1, cur_r))
        cur_c = max(0, min(len(matrix[0]) - 1, cur_c))

        mark_visited()
        status_text = "moving forward"
        print("Moving forward")
        print_matrix()

        # Refresh tag pose from camera to get updated measured_position
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            dets = apriltag.find_tags(gray)
        except Exception:
            dets = []

        if dets:
            # choose the first detection (caller assumed single tag)
            t_ca, R_ca = get_pose_apriltag_in_camera_frame(dets[0])
            x, y, measured_position = t_ca
        else:
            # If we can't see the tag after moving, stop attempting further approach.
            print("Lost tag after move; stopping forward approach")
            break

        # If we've reached within desired_position, mark the current cell with 'X'.
        if measured_position <= desired_position:
            mark_tag_front()
            print("At tag front; marking X")
            print_matrix()
            break


def center_tag_until_aligned(ep_chassis, ep_camera, apriltag, tol_px=10, timeout_s=None):
    """Rotate in-place until the detected tag is centered in the camera image.

    Returns True if centered, False on timeout or lost tag.
    """
    global status_text
    t0 = time.time()
    cx_des = 320.0
    last_err = None
    status_text = "centering"

    while True:
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            status_text = "center timeout"
            return False

        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Exception:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        dets = apriltag.find_tags(gray)
        draw_detections(img, dets)

        det = None
        if dets:
            # pick detection closest in camera frame
            det = min(dets, key=lambda d: np.linalg.norm(np.array(d.pose_t).reshape(3)))

        if det is None:
            status_text = "looking for tag"
            overlay_status_text(img, status_text)
            draw_minimap_overlay(img)
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                return False
            continue

        err = float(det.center[0] - cx_des)
        filt_err = err if last_err is None else 0.65 * last_err + 0.35 * err
        last_err = filt_err

        # show status on frame
        overlay_status_text(img, f"centering err={filt_err:+.1f}px")
        draw_minimap_overlay(img)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            return False

        if abs(filt_err) <= tol_px:
            status_text = "centered"
            overlay_status_text(img, "centered")
            draw_minimap_overlay(img)
            cv2.imshow("img", img)
            cv2.waitKey(1)
            print("Centered")
            return True

        # Compute conservative yaw rate and issue drive command for rotation
        z_mag = clamp(0.05 * abs(filt_err), 3.0, 10.0)
        # Left (negative error) -> counterclockwise (+z)
        z_cmd = (z_mag if filt_err < 0 else -z_mag)
        ep_chassis.drive_speed(x=0.0, y=0.0, z=z_cmd, timeout=0.2)



def sweep_lateral_until_tag(ep_chassis, ep_camera, apriltag, y_val=0.266):
    """
    Move laterally to the right in cube-sized steps until an AprilTag is detected.
    If the robot leaves the map bounds moving right, reverse direction and move left instead.
    Print the map before each decision to move.
    """
    global cur_r, cur_c
    cols = len(matrix[0])
    rows = len(matrix)

    # Try moving right until tag seen or out of bounds
    steps = 0
    max_steps = cols + 2
    direction = 1  # 1 means right (increase column), -1 means left

    while steps < max_steps:
        # Decide to move right one cube
        print("Decision: move right")
        print_matrix()
        # perform the lateral move
        ep_chassis.move(x=0, y=y_val, z=0, xy_speed=0.7).wait_for_completed()

        # update grid column in world coordinates (assume lateral +y increments column)
        cur_c += 1
        cur_c = max(0, min(cols - 1, cur_c))
        mark_visited()

        # check for tag after move
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            dets = apriltag.find_tags(gray)
        except Exception:
            dets = []

        if dets:
            print("Tag found during right sweep")
            return True

        # If we've reached the right edge of the map, reverse direction and move left
        if cur_c >= cols - 1:
            print("Out of map frame on right; reversing direction")
            # Try moving left until tag or until we return to left edge
            for _ in range(max_steps):
                print("Decision: move left")
                print_matrix()
                ep_chassis.move(x=0, y=-y_val, z=0, xy_speed=0.7).wait_for_completed()
                cur_c -= 1
                cur_c = max(0, min(cols - 1, cur_c))
                mark_visited()

                try:
                    img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                    dets = apriltag.find_tags(gray)
                except Exception:
                    dets = []

                if dets:
                    print("Tag found during left sweep")
                    return True

                if cur_c <= 0:
                    print("Returned to left edge; stopping sweep")
                    return False

            return False

        steps += 1

    return False


def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):
    global heading_deg, rotation_accum_deg, status_text
    while True:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray.astype(np.uint8)

        detections = apriltag.find_tags(gray)
        draw_detections(img, detections)
        # overlay current status and minimap before showing
        overlay_status_text(img, status_text)
        draw_minimap_overlay(img)
        cv2.imshow("img", img)

        if len(detections) > 0:
            # If multiple detections are present, pick the closest tag (smallest camera-frame distance)
            def _det_dist(d):
                t = np.array(d.pose_t, dtype=float).reshape(3)
                return np.linalg.norm(t)

            detection = min(detections, key=_det_dist)

            t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
            print('t_ca', t_ca, f"(num_detections={len(detections)})")
            # reset rotation accumulator when a tag is seen
            rotation_accum_deg = 0
            move_to_tag(t_ca, ep_chassis, ep_camera, apriltag)
        else:
            status_text = "looking for tag"
            print("No tag detected")
            # Decide to rotate 90deg and continue searching. Print map before moving.
            print("Decision: rotate 90 deg")
            print_matrix()
            ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed()
            heading_deg = (heading_deg + 90) % 360
            rotation_accum_deg += 90

            # Only perform lateral sweep after a full 360 rotation without detections
            if rotation_accum_deg >= 360:
                print("Completed 360deg search; performing lateral sweep")
                found = sweep_lateral_until_tag(ep_chassis, ep_camera, apriltag, y_val=0.266)
                rotation_accum_deg = 0
                if not found:
                    print("Sweep did not find tag; continuing rotation search")

        if cv2.waitKey(1) == ord('q'):
            break

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
