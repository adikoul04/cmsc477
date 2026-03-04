#!/usr/bin/env python3
"""
Project 1: AprilTag-based localization + path planning + closed-loop navigation.

Design choices aligned to project requirements:
- Known map and known AprilTag world poses.
- A* shortest path planning on occupancy grid.
- Obstacle inflation for robot footprint (robot is 1.5 cubes long, 1.0 cube wide).
- Continuous velocity control using pose estimated from AprilTags.
- At turn waypoints, use critical tags (aligned with robot heading) to re-center orientation.

Coordinate convention:
- Top-left origin for grid indexing.
- x increases right, y increases down.
"""

import math
import time
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import pupil_apriltags
except ModuleNotFoundError:
    pupil_apriltags = None

try:
    import robomaster
    from robomaster import camera as rm_camera
    from robomaster import robot
except ModuleNotFoundError:
    robomaster = None
    rm_camera = None
    robot = None

# =========================
# Known constants
# =========================

K = np.array(
    [
        [314, 0, 320],
        [0, 314, 180],
        [0, 0, 1],
    ],
    dtype=float,
)

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.200
CELL_SIZE_M = 0.266

# Top-left origin occupancy map (0=free, 1=obstacle)
OCC = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)

# Start/goal in original grid cells (top-left origin)
START_CELL = (3, 0)
GOAL_CELL = (3, 10)

# Controller settings
V_MAX = 0.22
W_MAX_DEG = 65.0
K_RHO = 0.9
K_ALPHA = 2.2
WAYPOINT_TOL_M = 0.08


# =========================
# Helpers
# =========================

def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def T_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def double_grid_resolution(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    doubled = np.zeros((2 * H, 2 * W), dtype=int)
    for r in range(H):
        for c in range(W):
            doubled[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = grid[r, c]
    return doubled


def inflate_obstacles(grid: np.ndarray, inflation_cells: int) -> np.ndarray:
    """Simple binary dilation without scipy dependency."""
    H, W = grid.shape
    out = grid.copy()
    occ = np.argwhere(grid == 1)
    for r, c in occ:
        r0 = max(0, r - inflation_cells)
        r1 = min(H, r + inflation_cells + 1)
        c0 = max(0, c - inflation_cells)
        c1 = min(W, c + inflation_cells + 1)
        out[r0:r1, c0:c1] = 1
    return out


def doubled_node_to_world(r: int, c: int, doubled_cell_size_m: float) -> Tuple[float, float]:
    """
    Convert a doubled-grid node index to world coordinates.

    For doubled planning, nodes represent half-cell lattice points, so world
    coordinates are index * doubled_cell_size (not cell centers).
    """
    return (c * doubled_cell_size_m, r * doubled_cell_size_m)


# =========================
# Data classes
# =========================

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class TagWorldPose:
    x: float
    y: float
    yaw: float


# =========================
# Mapping + planning
# =========================

class GridMap:
    def __init__(self, occ: np.ndarray, cell_size_m: float, origin_xy=(0.0, 0.0)):
        self.occ = occ.astype(int)
        self.H, self.W = self.occ.shape
        self.cell = float(cell_size_m)
        self.ox, self.oy = origin_xy

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.occ[r, c] == 0

    def neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        cand = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        return [(rr, cc) for rr, cc in cand if self.is_free(rr, cc)]

    def grid_to_world_center(self, r: int, c: int) -> Tuple[float, float]:
        x = self.ox + (c + 0.5) * self.cell
        y = self.oy + (r + 0.5) * self.cell
        return (x, y)


def astar(grid: GridMap, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    import heapq

    def h(rc):
        r, c = rc
        gr, gc = goal_rc
        return abs(r - gr) + abs(c - gc)

    open_heap = []
    heapq.heappush(open_heap, (h(start_rc), 0.0, start_rc))
    came_from = {start_rc: None}
    gscore = {start_rc: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal_rc:
            break
        for nb in grid.neighbors4(*cur):
            ng = g + 1.0
            if nb not in gscore or ng < gscore[nb]:
                gscore[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))

    if goal_rc not in came_from:
        return []

    path = []
    cur = goal_rc
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def remove_collinear(path_rc: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Keep only corners + endpoints."""
    if len(path_rc) <= 2:
        return path_rc[:]
    out = [path_rc[0]]
    for i in range(1, len(path_rc) - 1):
        r0, c0 = path_rc[i - 1]
        r1, c1 = path_rc[i]
        r2, c2 = path_rc[i + 1]
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            out.append(path_rc[i])
    out.append(path_rc[-1])
    return out


def turn_waypoint_indices(path_rc: List[Tuple[int, int]]) -> List[int]:
    idx = []
    for i in range(1, len(path_rc) - 1):
        d1 = (path_rc[i][0] - path_rc[i - 1][0], path_rc[i][1] - path_rc[i - 1][1])
        d2 = (path_rc[i + 1][0] - path_rc[i][0], path_rc[i + 1][1] - path_rc[i][1])
        if d1 != d2:
            idx.append(i)
    return idx


# =========================
# AprilTag localization
# =========================

class AprilTagDetector:
    def __init__(self, Kmat, family=TAG_FAMILY, threads=2, marker_size_m=TAG_SIZE_M):
        if pupil_apriltags is None:
            raise ImportError("pupil_apriltags is required for AprilTag detection.")
        self.camera_params = [Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]]
        self.detector = pupil_apriltags.Detector(
            families=family,
            nthreads=threads,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.marker_size_m = marker_size_m

    def find_tags(self, gray_img: np.ndarray):
        return self.detector.detect(
            gray_img,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )


def draw_detections(img: np.ndarray, detections) -> None:
    if cv2 is None:
        return
    for det in detections:
        corners = det.corners.astype(int)
        for i in range(4):
            p0 = tuple(corners[i])
            p1 = tuple(corners[(i + 1) % 4])
            cv2.line(img, p0, p1, (0, 255, 0), 2)
        c = det.center.astype(int)
        cv2.circle(img, tuple(c), 4, (0, 0, 255), -1)
        cv2.putText(img, str(det.tag_id), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def build_tag_world_map() -> Dict[int, TagWorldPose]:
    """Tag map from Figure 1 using top-left world convention."""
    m: Dict[int, TagWorldPose] = {}

    def pos(row_offset: float, col_offset: float) -> Tuple[float, float]:
        return (col_offset * CELL_SIZE_M, row_offset * CELL_SIZE_M)

    m[30] = TagWorldPose(*pos(1.5, 2.0), yaw=math.pi)
    m[31] = TagWorldPose(*pos(1.5, 3.0), yaw=0.0)
    m[32] = TagWorldPose(*pos(3.5, 2.0), yaw=math.pi)
    m[33] = TagWorldPose(*pos(3.5, 3.0), yaw=0.0)
    m[34] = TagWorldPose(*pos(5.0, 2.5), yaw=math.pi / 2)

    m[35] = TagWorldPose(*pos(1.0, 4.5), yaw=math.pi / 2)
    m[36] = TagWorldPose(*pos(1.0, 6.5), yaw=math.pi / 2)

    m[37] = TagWorldPose(*pos(4.0, 5.5), yaw=-math.pi / 2)
    m[38] = TagWorldPose(*pos(5.5, 5.0), yaw=math.pi)
    m[39] = TagWorldPose(*pos(5.5, 6.0), yaw=0.0)
    m[40] = TagWorldPose(*pos(7.5, 5.0), yaw=math.pi)
    m[41] = TagWorldPose(*pos(7.5, 6.0), yaw=0.0)

    m[42] = TagWorldPose(*pos(1.5, 8.0), yaw=math.pi)
    m[43] = TagWorldPose(*pos(1.5, 9.0), yaw=0.0)
    m[44] = TagWorldPose(*pos(3.5, 8.0), yaw=math.pi)
    m[45] = TagWorldPose(*pos(3.5, 9.0), yaw=0.0)
    m[46] = TagWorldPose(*pos(5.0, 8.5), yaw=math.pi / 2)

    return m


def build_T_rc() -> np.ndarray:
    """
    Camera pose in robot frame.
    Identity is used here because x/y are centered with robot body center and only z differs.
    """
    return np.eye(4, dtype=float)


class Localizer:
    def __init__(self, tag_world_map: Dict[int, TagWorldPose], T_rc: np.ndarray):
        self.tag_world_map = tag_world_map
        self.T_rc = T_rc
        self.last_pose: Optional[Pose2D] = None
        self.alpha = 0.35

    def estimate_robot_pose(self, detection) -> Optional[Pose2D]:
        tag_id = int(detection.tag_id)
        if tag_id not in self.tag_world_map:
            return None

        t_ca = np.array(detection.pose_t, dtype=float).reshape(3)
        R_ca = np.array(detection.pose_R, dtype=float).reshape(3, 3)
        T_ca = T_from_R_t(R_ca, t_ca)  # tag -> camera

        tw = self.tag_world_map[tag_id]
        T_wa = T_from_R_t(rotz(tw.yaw), np.array([tw.x, tw.y, 0.0], dtype=float))  # tag -> world

        T_wc = T_wa @ inv_T(T_ca)  # camera -> world
        T_wr = T_wc @ inv_T(self.T_rc)  # robot -> world

        pose = Pose2D(
            x=float(T_wr[0, 3]),
            y=float(T_wr[1, 3]),
            yaw=yaw_from_R(T_wr[:3, :3]),
        )
        return self._smooth(pose)

    def _smooth(self, pose: Pose2D) -> Pose2D:
        if self.last_pose is None:
            self.last_pose = pose
            return pose

        lp = self.last_pose
        a = self.alpha

        x = (1 - a) * lp.x + a * pose.x
        y = (1 - a) * lp.y + a * pose.y

        u0 = np.array([math.cos(lp.yaw), math.sin(lp.yaw)])
        u1 = np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        u = (1 - a) * u0 + a * u1
        yaw = math.atan2(u[1], u[0])

        self.last_pose = Pose2D(x=x, y=y, yaw=yaw)
        return self.last_pose


# =========================
# Critical tags at turns
# =========================

def find_critical_tags(
    path_rc: List[Tuple[int, int]],
    tag_map: Dict[int, TagWorldPose],
    grid: GridMap,
) -> Dict[int, List[int]]:
    """
    For each turn waypoint, select tags that are in-line and in-front before turn.
    Key: waypoint index in path_rc, Value: candidate tag IDs.
    """

    def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
        c = int(round((wx - grid.ox) / grid.cell))
        r = int(round((wy - grid.oy) / grid.cell))
        return r, c

    def line_of_sight_clear(rx: float, ry: float, tx: float, ty: float) -> bool:
        # Sample the segment in world coordinates and reject if it passes through
        # occupied original-map cells before reaching the tag location.
        dist = math.hypot(tx - rx, ty - ry)
        if dist < 1e-6:
            return True
        steps = max(2, int(dist / (CELL_SIZE_M / 8.0)))
        for k in range(1, steps):
            a = k / steps
            if a > 0.92:
                break  # ignore final short segment near the tag itself
            x = rx + a * (tx - rx)
            y = ry + a * (ty - ry)
            cc = int(x / CELL_SIZE_M)
            rr = int(y / CELL_SIZE_M)
            if 0 <= rr < OCC.shape[0] and 0 <= cc < OCC.shape[1] and OCC[rr, cc] == 1:
                return False
        return True

    out: Dict[int, List[int]] = {}

    for i in turn_waypoint_indices(path_rc):
        r_prev, c_prev = path_rc[i - 1]
        r_curr, c_curr = path_rc[i]

        dr = r_curr - r_prev
        dc = c_curr - c_prev

        if dc > 0:
            heading = 0.0
            horizontal = True
        elif dc < 0:
            heading = math.pi
            horizontal = True
        elif dr > 0:
            heading = math.pi / 2
            horizontal = False
        else:
            heading = -math.pi / 2
            horizontal = False

        rx, ry = doubled_node_to_world(r_curr, c_curr, grid.cell)
        ids: List[int] = []

        for tag_id, tag_pose in tag_map.items():
            tr, tc = world_to_grid(tag_pose.x, tag_pose.y)
            if horizontal and tr != r_curr:
                continue
            if (not horizontal) and tc != c_curr:
                continue

            dx = tag_pose.x - rx
            dy = tag_pose.y - ry
            dist = math.hypot(dx, dy)
            if dist < 0.05 or dist > 1.2:
                continue

            angle_to_tag = math.atan2(dy, dx)
            if abs(wrap_to_pi(angle_to_tag - heading)) > math.radians(30):
                continue

            # Tag should face toward the robot.
            tag_to_robot = wrap_to_pi(angle_to_tag + math.pi)
            if abs(wrap_to_pi(tag_pose.yaw - tag_to_robot)) > math.radians(60):
                continue

            if not line_of_sight_clear(rx, ry, tag_pose.x, tag_pose.y):
                continue

            ids.append(tag_id)

        if ids:
            out[i] = sorted(ids)

    return out


def choose_detection(detections, valid_ids: Optional[set] = None):
    cand = []
    for d in detections:
        if valid_ids is not None and int(d.tag_id) not in valid_ids:
            continue
        margin = getattr(d, "decision_margin", 0.0)
        area = cv2.contourArea(d.corners.astype(np.float32))
        cand.append((margin, area, d))
    if not cand:
        return None
    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return cand[0][2]


def center_tag_in_view(
    ep_chassis,
    ep_camera,
    detector: AprilTagDetector,
    target_ids: List[int],
    timeout_s: float = 1.5,
    tol_px: float = 24.0,
) -> bool:
    """Rotate in place to center target tag in camera frame."""
    t0 = time.time()
    target_set = set(target_ids)
    cx_des = 320.0

    while time.time() - t0 < timeout_s:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.2)
        except Empty:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        detections = detector.find_tags(gray)
        det = choose_detection(detections, valid_ids=target_set)

        if det is None:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=12.0, timeout=1)
            continue

        err = float(det.center[0] - cx_des)
        if abs(err) <= tol_px:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
            return True

        z_cmd = clamp(-0.10 * err, -25.0, 25.0)
        ep_chassis.drive_speed(x=0.0, y=0.0, z=z_cmd, timeout=1)

    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
    return False


# =========================
# Main
# =========================

def main() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required to run project1_nav main().")
    if robomaster is None or robot is None or rm_camera is None:
        raise ImportError("robomaster SDK is required to run project1_nav main().")

    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    print("\n=== Planning ===")
    print(f"Original grid: {OCC.shape}, cell={CELL_SIZE_M:.3f} m")

    doubled_occ = double_grid_resolution(OCC)
    doubled_cell = CELL_SIZE_M / 2.0

    # 1 doubled-cell inflation == 0.133 m == 0.5 cube guardrail in original scale.
    inflated = inflate_obstacles(doubled_occ, inflation_cells=1)

    start_d = (2 * START_CELL[0] + 1, 2 * START_CELL[1] + 1)
    goal_d = (2 * GOAL_CELL[0] + 1, 2 * GOAL_CELL[1] + 1)

    grid = GridMap(inflated, cell_size_m=doubled_cell, origin_xy=(0.0, 0.0))
    path_rc = astar(grid, start_d, goal_d)
    if not path_rc:
        print("No path found.")
        return

    path_rc = remove_collinear(path_rc)
    path_xy = [doubled_node_to_world(r, c, doubled_cell) for (r, c) in path_rc]

    print(f"Path waypoints: {len(path_xy)}")
    for i, (r, c) in enumerate(path_rc):
        x, y = path_xy[i]
        print(f"  {i:2d}: rc=({r:2d},{c:2d}) -> ({x:.3f}, {y:.3f})")

    tag_world_map = build_tag_world_map()
    critical_tags = find_critical_tags(path_rc, tag_world_map, grid)
    print("Critical tags at turn waypoints:")
    if not critical_tags:
        print("  none")
    else:
        for idx, ids in critical_tags.items():
            print(f"  wp {idx}: {ids}")

    print("\n=== Robot Init ===")
    ep_robot = robot.Robot()
    robomaster.config.ROBOT_IP_STR = "192.168.50.117"
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100RC")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=True, resolution=rm_camera.STREAM_360P)
    detector = AprilTagDetector(K)

    cv2.namedWindow("Robot Camera", cv2.WINDOW_NORMAL)
    print("Press 'q' to stop.")

    # Open-loop execution by known segment distances.
    # Assumes initial robot heading is +x (to the right on map).
    expected_yaw = 0.0
    turn_speed_deg = 30.0
    move_speed_mps = 0.15
    completed = False

    def desired_yaw_from_step(dr: int, dc: int) -> float:
        if dc > 0:
            return 0.0
        if dc < 0:
            return math.pi
        if dr > 0:
            return math.pi / 2
        return -math.pi / 2

    try:
        for i in range(len(path_rc) - 1):
            # If this waypoint is a turn point, correct heading by centering critical tag.
            if i in critical_tags:
                print(f"Waypoint {i}: centering critical tags {critical_tags[i]}")
                center_tag_in_view(ep_chassis, ep_camera, detector, critical_tags[i])

            r0, c0 = path_rc[i]
            r1, c1 = path_rc[i + 1]
            dr, dc = (r1 - r0), (c1 - c0)
            desired_yaw = desired_yaw_from_step(dr, dc)

            # Turn to desired heading (open-loop timing).
            turn_rad = wrap_to_pi(desired_yaw - expected_yaw)
            turn_deg = math.degrees(turn_rad)
            if abs(turn_deg) > 1.0:
                z_cmd = turn_speed_deg if turn_deg > 0 else -turn_speed_deg
                t_turn = abs(turn_deg) / turn_speed_deg
                print(f"Segment {i}: turn {turn_deg:.1f} deg")
                ep_chassis.drive_speed(x=0.0, y=0.0, z=z_cmd, timeout=3)
                time.sleep(t_turn)
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
                time.sleep(0.2)
            expected_yaw = desired_yaw

            # Move exact known segment distance from map geometry.
            x0, y0 = path_xy[i]
            x1, y1 = path_xy[i + 1]
            dist = math.hypot(x1 - x0, y1 - y0)
            t_move = dist / move_speed_mps if move_speed_mps > 1e-6 else 0.0
            print(f"Segment {i}: move {dist:.3f} m")
            ep_chassis.drive_speed(x=move_speed_mps, y=0.0, z=0.0, timeout=5)
            time.sleep(t_move)
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
            time.sleep(0.2)

            # Allow user stop after each segment.
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.2)
                cv2.imshow("Robot Camera", img)
            except Exception:
                pass
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User stop requested.")
                break

        else:
            completed = True

        ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
        if completed:
            print("Reached goal.")
        else:
            print("Stopped before goal.")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
        except Exception:
            pass
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        ep_robot.close()
        cv2.destroyAllWindows()
        print(f"Traveled pose samples: {len(traveled)}")


if __name__ == "__main__":
    main()
