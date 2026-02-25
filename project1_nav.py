#!/usr/bin/env python3
"""
Project 1 — Navigation + Localization using AprilTags (single-file, based on your apriltag.py)

What is ALREADY filled in (from your files + Figure 1 screenshot):
- Camera intrinsics (same as apriltag.py):
    K = [[314, 0, 320],
         [0, 314, 180],
         [0,   0,   1]]
- AprilTag family: tag36h11
- AprilTag printed size: 200 mm => TAG_SIZE_M = 0.200
- Grid cell size (storage cubes): 26.6 cm => CELL_SIZE_M = 0.266
- Occupancy grid shape and blocked cells extracted from your Figure 1 screenshot:
    Grid is 11 columns x 9 rows, origin at bottom-left.
- Start/Goal cell indices extracted from your Figure 1 screenshot:
    start_rc = (5, 0), goal_rc = (5, 10)   (r=0 bottom, c=0 left)

What you STILL must fill to make this fully correct on the real robot:
1) TAG_WORLD_MAP (tag_id -> TagWorldPose(x,y,yaw)):
   - Enter the true world pose of each tag in Figure 1.
   - Tags are on cube faces → often offset by ±CELL_SIZE_M/2 from cube centers.
2) Camera-to-robot transform T_rc:
   - Measure camera offset relative to robot base frame (even rough is better than identity).
3) Verify world origin + axis convention:
   - origin is bottom-left corner of grid
   - x right, y up
4) Tune controller params (k_v, k_w, limits).

Run:
  python3 project1_nav.py
Stop:
  press 'q' in the OpenCV window or Ctrl+C
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from queue import Empty

import pupil_apriltags
from robomaster import robot, camera as rm_camera
import robomaster


# =========================
# Constants we KNOW
# =========================

K = np.array([[314, 0, 320],
              [0, 314, 180],
              [0,   0,   1]], dtype=float)

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.200     # 200 mm tags for THIS project
CELL_SIZE_M = 0.266    # 26.6 cm storage cube grid

# Occupancy from Figure 1 screenshot (r=0 TOP row, c=0 left column)
# Origin is TOP-LEFT to match numpy indexing
# Row 0 is top, row 8 is bottom; Column 0 is left, column 10 is right
OCC = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Row 0 (top)
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 1
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 2
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 3
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # Row 4 (middle)
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 5
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 6
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 7
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=int)  # Row 8 (bottom)

# Start / goal cells - using TOP-LEFT origin
# Start is bottom-left area, Goal is bottom-right area
START_RC = (3, 0)   
GOAL_RC  = (3, 10)


# =========================
# AprilTag code copied from your apriltag.py (modified tag size)
# =========================

class AprilTagDetector:
    def __init__(self, K, family=TAG_FAMILY, threads=2, marker_size_m=TAG_SIZE_M):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m
        self.detector = pupil_apriltags.Detector(
            families=family,
            nthreads=threads,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def find_tags(self, gray_img: np.ndarray):
        return self.detector.detect(
            gray_img,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m,
        )

def get_pose_apriltag_in_camera_frame(detection):
    return detection.pose_t, detection.pose_R

def draw_detections(img, detections):
    for det in detections:
        corners = det.corners.astype(int)
        for i in range(4):
            p0 = tuple(corners[i])
            p1 = tuple(corners[(i+1) % 4])
            cv2.line(img, p0, p1, (0, 255, 0), 2)
        c = det.center.astype(int)
        cv2.circle(img, tuple(c), 4, (0, 0, 255), -1)
        cv2.putText(img, str(det.tag_id), (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


# =========================
# Math helpers
# =========================

def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

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


# =========================
# Data structures
# =========================

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float  # radians

@dataclass
class TagWorldPose:
    x: float
    y: float
    yaw: float  # radians (direction the TAG faces in world)

@dataclass
class ControlParams:
    v_max: float
    w_max_deg: float
    k_v: float
    k_w: float
    waypoint_tol: float
    lookahead: float


# =========================
# Grid map + planning
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
        cand = [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]
        return [(rr,cc) for rr,cc in cand if self.is_free(rr,cc)]

    def grid_to_world_center(self, r: int, c: int) -> Tuple[float, float]:
        """Convert grid cell (r,c) to world (x,y) at cell center.
        Top-left origin: x increases right, y increases down.
        """
        x = self.ox + (c + 0.5) * self.cell
        y = self.oy + (r + 0.5) * self.cell
        return (x, y)


def astar(grid: GridMap, start_rc: Tuple[int,int], goal_rc: Tuple[int,int]) -> List[Tuple[int,int]]:
    import heapq

    def h(rc):
        r,c = rc
        gr,gc = goal_rc
        return abs(r-gr) + abs(c-gc)

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

def densify(points: List[Tuple[float,float]], step_m: float = 0.08) -> List[Tuple[float,float]]:
    if len(points) <= 1:
        return points[:]
    out = [points[0]]
    for i in range(1, len(points)):
        x0,y0 = out[-1]
        x1,y1 = points[i]
        dx,dy = x1-x0, y1-y0
        dist = math.hypot(dx,dy)
        if dist < 1e-9:
            continue
        n = int(dist // step_m)
        for k in range(1, n+1):
            a = (k * step_m) / dist
            out.append((x0 + a*dx, y0 + a*dy))
        out.append((x1,y1))
    return out


# =========================
# Tag world map + localization
# =========================

def build_tag_world_map() -> Dict[int, TagWorldPose]:
    """
    Tag placements from Figure 1.

    Coordinate convention (TOP-LEFT ORIGIN):
      - origin (0,0) at top-left of grid
      - x increases right, y increases down
      - cell (r,c) center: x = (c+0.5)*CELL, y = (r+0.5)*CELL
      - yaw: 0=right, π/2=down, π=left, -π/2=up
    """
    m: Dict[int, TagWorldPose] = {}

    # Fill like:
    # m[30] = TagWorldPose(x=..., y=..., yaw=...)
    
    HALF = 0.133  # CELL_SIZE_M / 2

    # -----------------------
    # Top horizontal bar (TOP of Figure 1)
    # -----------------------

    # Tag 30
    # Block at (r=1, c=2) → y = (1+0.5)*0.266 = 0.399
    # Facing LEFT
    m[30] = TagWorldPose(x=0.665 - HALF, y=0.399, yaw=math.pi)

    # Tag 31
    # Block at (r=1, c=2)
    # Facing RIGHT
    m[31] = TagWorldPose(x=0.665 + HALF, y=0.399, yaw=0)

    # Tag 32
    # Block at (r=3, c=2) → y = (3+0.5)*0.266 = 0.931
    # Facing LEFT
    m[32] = TagWorldPose(x=0.665 - HALF, y=0.931, yaw=math.pi)

    # Tag 33
    # Block at (r=3, c=2)
    # Facing RIGHT
    m[33] = TagWorldPose(x=0.665 + HALF, y=0.931, yaw=0)

    # Tag 34
    # Block at (r=4, c=2) → y_center = 1.197, facing DOWN (π/2)
    m[34] = TagWorldPose(x=0.665, y=1.197 + HALF, yaw=math.pi/2)


    # -----------------------
    # Middle horizontal bars
    # -----------------------

    # Tag 35
    # Block at (r=0, c=4) → y = (0+0.5)*0.266 = 0.133, facing DOWN (π/2)
    m[35] = TagWorldPose(x=1.197, y=0.133 + HALF, yaw=math.pi/2)

    # Tag 36
    # Block at (r=0, c=6) → y = 0.133, facing DOWN (π/2)
    m[36] = TagWorldPose(x=1.729, y=0.133 + HALF, yaw=math.pi/2)


    # -----------------------
    # Center vertical stem (MIDDLE of Figure 1)
    # -----------------------

    # Tag 37
    # Block at (r=4, c=5) → y = 1.197, facing UP (-π/2)
    m[37] = TagWorldPose(x=1.463, y=1.197 - HALF, yaw=-math.pi/2)

    # Tag 38
    # Block at (r=5, c=5) → y = 1.463, facing LEFT
    m[38] = TagWorldPose(x=1.463 - HALF, y=1.463, yaw=math.pi)

    # Tag 39
    # Block at (r=5, c=5) → facing RIGHT
    m[39] = TagWorldPose(x=1.463 + HALF, y=1.463, yaw=0)

    # Tag 40
    # Block at (r=7, c=5) → y = 1.995, facing LEFT
    m[40] = TagWorldPose(x=1.463 - HALF, y=1.995, yaw=math.pi)

    # Tag 41
    # Block at (r=7, c=5) → facing RIGHT
    m[41] = TagWorldPose(x=1.463 + HALF, y=1.995, yaw=0)


    # -----------------------
    # Bottom vertical pillars (BOTTOM of Figure 1)
    # -----------------------

    # Tag 42
    # Block at (r=1, c=8) → y = 0.399, facing LEFT
    m[42] = TagWorldPose(x=2.261 - HALF, y=0.399, yaw=math.pi)

    # Tag 43
    # Block at (r=1, c=8) → facing RIGHT
    m[43] = TagWorldPose(x=2.261 + HALF, y=0.399, yaw=0)

    # Tag 44
    # Block at (r=3, c=8) → y = 0.931, facing LEFT
    m[44] = TagWorldPose(x=2.261 - HALF, y=0.931, yaw=math.pi)

    # Tag 45
    # Block at (r=3, c=8) → facing RIGHT
    m[45] = TagWorldPose(x=2.261 + HALF, y=0.931, yaw=0)

    # Tag 46
    # Block at (r=4, c=8) → y = 1.197, facing DOWN (π/2)
    m[46] = TagWorldPose(x=2.261, y=1.197 + HALF, yaw=math.pi/2)
    return m


def build_T_rc() -> np.ndarray:
    """
    TODO: Camera pose in robot frame (T_rc).
    Start with identity, then measure offsets and update.
    """
    T = np.eye(4)
    return T


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

        t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
        t_ca = np.array(t_ca, dtype=float).reshape(3)
        R_ca = np.array(R_ca, dtype=float).reshape(3,3)
        T_ca = T_from_R_t(R_ca, t_ca)  # tag -> camera

        tw = self.tag_world_map[tag_id]
        T_wa = T_from_R_t(rotz(tw.yaw), np.array([tw.x, tw.y, 0.0], dtype=float))  # tag -> world

        T_wc = T_wa @ inv_T(T_ca)       # camera -> world
        T_wr = T_wc @ inv_T(self.T_rc)  # robot  -> world

        pose = Pose2D(
            x=float(T_wr[0,3]),
            y=float(T_wr[1,3]),
            yaw=yaw_from_R(T_wr[:3,:3]),
        )
        return self._smooth(pose)

    def _smooth(self, pose: Pose2D) -> Pose2D:
        if self.last_pose is None:
            self.last_pose = pose
            return pose
        lp = self.last_pose
        a = self.alpha

        x = (1-a)*lp.x + a*pose.x
        y = (1-a)*lp.y + a*pose.y

        u0 = np.array([math.cos(lp.yaw), math.sin(lp.yaw)])
        u1 = np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        u = (1-a)*u0 + a*u1
        yaw = math.atan2(u[1], u[0])

        self.last_pose = Pose2D(x=x, y=y, yaw=yaw)
        return self.last_pose


# =========================
# Path following control
# =========================

class WaypointFollower:
    def __init__(self, path_xy: List[Tuple[float,float]], p: ControlParams):
        self.path = path_xy
        self.p = p
        self.i = 0

    def done(self) -> bool:
        return self.i >= len(self.path)

    def _advance_if_close(self, pose: Pose2D):
        while self.i < len(self.path):
            tx, ty = self.path[self.i]
            if math.hypot(tx - pose.x, ty - pose.y) <= self.p.waypoint_tol:
                self.i += 1
            else:
                break

    def _lookahead(self, pose: Pose2D) -> Optional[Tuple[float,float]]:
        if self.done():
            return None
        for j in range(self.i, len(self.path)):
            tx, ty = self.path[j]
            if math.hypot(tx - pose.x, ty - pose.y) >= self.p.lookahead:
                return (tx, ty)
        return self.path[-1]

    def command(self, pose: Pose2D) -> Tuple[float, float]:
        self._advance_if_close(pose)
        if self.done():
            return 0.0, 0.0

        target = self._lookahead(pose)
        if target is None:
            return 0.0, 0.0

        tx, ty = target
        dx, dy = tx - pose.x, ty - pose.y
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        alpha = wrap_to_pi(desired - pose.yaw)

        v = clamp(self.p.k_v * dist, 0.0, self.p.v_max)
        w_deg = clamp((self.p.k_w * alpha) * 180.0 / math.pi, -self.p.w_max_deg, self.p.w_max_deg)
        return v, w_deg


def recovery_spin(ep_chassis, spin_deg_s: float = 15.0):
    ep_chassis.drive_speed(x=0.0, y=0.0, z=spin_deg_s, timeout=1)


# =========================
# Odometry-based pose tracking
# =========================

class OdometryTracker:
    """Track robot pose using odometry with AprilTag corrections"""
    def __init__(self, initial_pose: Pose2D):
        self.pose = initial_pose
        self.last_update_time = time.time()
    
    def update_odometry(self, v: float, w_deg: float, dt: float):
        """Update pose using velocity commands (simple odometry model)"""
        w_rad = w_deg * math.pi / 180.0
        
        # Simple motion model: assumes constant velocity over dt
        if abs(w_rad) < 1e-6:  # Moving straight
            dx = v * dt * math.cos(self.pose.yaw)
            dy = v * dt * math.sin(self.pose.yaw)
            self.pose.x += dx
            self.pose.y += dy
        else:  # Turning
            # Arc motion model
            R = v / w_rad  # Radius of curvature
            dtheta = w_rad * dt
            
            # Center of rotation in current frame
            self.pose.x += R * (math.sin(self.pose.yaw + dtheta) - math.sin(self.pose.yaw))
            self.pose.y += R * (-math.cos(self.pose.yaw + dtheta) + math.cos(self.pose.yaw))
            self.pose.yaw = wrap_to_pi(self.pose.yaw + dtheta)
    
    def correct_with_tag(self, tag_pose: Pose2D, alpha: float = 0.5):
        """Correct odometry pose using AprilTag observation"""
        # Blend odometry with tag observation
        self.pose.x = (1 - alpha) * self.pose.x + alpha * tag_pose.x
        self.pose.y = (1 - alpha) * self.pose.y + alpha * tag_pose.y
        
        # Handle angle wrapping for yaw blending
        u_odom = np.array([math.cos(self.pose.yaw), math.sin(self.pose.yaw)])
        u_tag = np.array([math.cos(tag_pose.yaw), math.sin(tag_pose.yaw)])
        u_blend = (1 - alpha) * u_odom + alpha * u_tag
        self.pose.yaw = math.atan2(u_blend[1], u_blend[0])
    
    def get_pose(self) -> Pose2D:
        return self.pose


# =========================
# Main
# =========================

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    # Build grid and plan
    grid = GridMap(OCC, cell_size_m=CELL_SIZE_M, origin_xy=(0.0, 0.0))
    path_rc = astar(grid, START_RC, GOAL_RC)
    if not path_rc:
        print("No path found on the current occupancy grid.")
        return
    
    # Debug: print the planned path
    print(f"\n=== Planned Path (TOP-LEFT origin) ===")
    print(f"Start: row={START_RC[0]}, col={START_RC[1]}")
    print(f"Goal:  row={GOAL_RC[0]}, col={GOAL_RC[1]}")
    print(f"Path has {len(path_rc)} waypoints:")
    for i, (r, c) in enumerate(path_rc):
        wx, wy = grid.grid_to_world_center(r, c)
        print(f"  {i:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})")
    print(f"======================================\n")
    
    path_xy = densify([grid.grid_to_world_center(r,c) for (r,c) in path_rc], step_m=0.08)

    # AprilTag + localization
    tag_world_map = build_tag_world_map()
    T_rc = build_T_rc()
    localizer = Localizer(tag_world_map, T_rc)

    # Controller params (TODO tune)
    p = ControlParams(
        v_max=0.20,
        w_max_deg=60.0,
        k_v=0.8,
        k_w=2.0,
        waypoint_tol=0.12,
        lookahead=0.25,
    )
    follower = WaypointFollower(path_xy, p)

    # Robot setup
    ep_robot = robot.Robot()
    robomaster.config.ROBOT_IP_STR = "192.168.50.117"
    # ep_robot.initialize(conn_type="ap")
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100RC")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_360P)

    detector = AprilTagDetector(K=K, family=TAG_FAMILY, threads=2, marker_size_m=TAG_SIZE_M)

    traveled: List[Tuple[float,float]] = []
    dt = 1.0 / 15.0
    
    # Wait for first tag to initialize
    odom_tracker: Optional[OdometryTracker] = None
    last_v, last_w_deg = 0.0, 0.0
    
    print("Waiting for first AprilTag to initialize position...")

    try:
        while True:
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            detections = detector.find_tags(gray)

            draw_detections(img, detections)
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break

            # Initialize odometry with first tag detection
            if odom_tracker is None:
                if detections:
                    det = detections[0]
                    initial_pose = localizer.estimate_robot_pose(det)
                    if initial_pose is not None:
                        odom_tracker = OdometryTracker(initial_pose)
                        print(f"Initialized at: x={initial_pose.x:.3f}, y={initial_pose.y:.3f}, yaw={math.degrees(initial_pose.yaw):.1f}°")
                        print(f"Detected tag ID: {det.tag_id}")
                else:
                    # Spin slowly to find a tag
                    recovery_spin(ep_chassis, spin_deg_s=20.0)
                    time.sleep(dt)
                continue
            
            # Update odometry with last commanded velocities
            odom_tracker.update_odometry(last_v, last_w_deg, dt)
            
            # If we see a tag, correct the odometry
            if detections:
                det = detections[0]
                tag_pose = localizer.estimate_robot_pose(det)
                if tag_pose is not None:
                    odom_tracker.correct_with_tag(tag_pose, alpha=0.6)
                    print(f"Tag {det.tag_id} correction: x={tag_pose.x:.3f}, y={tag_pose.y:.3f}")
            
            # Get current pose from odometry (corrected if tag was seen)
            current_pose = odom_tracker.get_pose()
            traveled.append((current_pose.x, current_pose.y))
            
            # Compute control commands based on path
            v, w_deg = follower.command(current_pose)
            ep_chassis.drive_speed(x=v, y=0.0, z=w_deg, timeout=1)
            
            # Store commands for next odometry update
            last_v, last_w_deg = v, w_deg

            if follower.done():
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
                print("Reached goal.")
                break

            time.sleep(dt)

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

        # TODO: save logs for report plot
        # np.savetxt("planned.csv", np.array(path_xy), delimiter=",", header="x,y", comments="")
        # np.savetxt("traveled.csv", np.array(traveled), delimiter=",", header="x,y", comments="")
        print("Planned points:", len(path_xy), "traveled samples:", len(traveled))


if __name__ == "__main__":
    main()
