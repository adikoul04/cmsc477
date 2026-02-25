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

# Positions stored as (row+0.5, col+0.5) - cell centers in grid units
START_RC = (3.5, 0.5)
GOAL_RC  = (3.5, 10.5)


def double_grid_resolution(grid: np.ndarray) -> np.ndarray:
    """
    Double the grid resolution by expanding each cell into a 2x2 block.
    This allows for half-cell precision in path planning and obstacle inflation.
    
    Args:
        grid: Original occupancy grid (H x W)
    
    Returns:
        Doubled grid (2H x 2W) where each original cell becomes 2x2
    """
    H, W = grid.shape
    doubled = np.zeros((2*H, 2*W), dtype=int)
    
    for r in range(H):
        for c in range(W):
            # Each cell (r,c) becomes a 2x2 block at (2r:2r+2, 2c:2c+2)
            doubled[2*r:2*r+2, 2*c:2*c+2] = grid[r, c]
    
    return doubled


def inflate_obstacles(grid: np.ndarray, inflation_cells: int = 1) -> np.ndarray:
    """
    Inflate obstacles by adding a safety margin around them.
    This accounts for robot dimensions in path planning.
    
    Args:
        grid: Binary occupancy grid (0=free, 1=occupied)
        inflation_cells: Number of cells to inflate in all directions
    
    Returns:
        Inflated grid with same shape
    """
    from scipy.ndimage import binary_dilation
    
    # Create structuring element for inflation
    struct = np.ones((2*inflation_cells+1, 2*inflation_cells+1), dtype=int)
    
    # Dilate obstacles
    inflated = binary_dilation(grid, structure=struct).astype(int)
    
    return inflated


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
    """Build the AprilTag world map
    
    Tags are positioned on cube faces with 0.5 offset from cell centers.
    Grid cell centers are at (row+0.5, col+0.5) * CELL_SIZE_M
    """
    m: Dict[int, TagWorldPose] = {}
    
    # Helper function to convert grid position to world coordinates
    def pos(row_offset: float, col_offset: float) -> Tuple[float, float]:
        return (col_offset * CELL_SIZE_M, row_offset * CELL_SIZE_M)
    
    # Top horizontal bar - Tags on cells in row 1, 3, and 4, column 2
    m[30] = TagWorldPose(*pos(1.5, 2.0), yaw=math.pi)      # Left face of (1,2)
    m[31] = TagWorldPose(*pos(1.5, 3.0), yaw=0)            # Right face of (1,2)
    m[32] = TagWorldPose(*pos(3.5, 2.0), yaw=math.pi)      # Left face of (3,2)
    m[33] = TagWorldPose(*pos(3.5, 3.0), yaw=0)            # Right face of (3,2)
    m[34] = TagWorldPose(*pos(5.0, 2.5), yaw=math.pi/2)    # Bottom face of (4,2)

    # Middle horizontal bars - row 0
    m[35] = TagWorldPose(*pos(1.0, 4.5), yaw=math.pi/2)    # Bottom face of (0,4)
    m[36] = TagWorldPose(*pos(1.0, 6.5), yaw=math.pi/2)    # Bottom face of (0,6)

    # Center vertical stem - column 5
    m[37] = TagWorldPose(*pos(4.0, 5.5), yaw=-math.pi/2)   # Top face of (4,5)
    m[38] = TagWorldPose(*pos(5.5, 5.0), yaw=math.pi)      # Left face of (5,5)
    m[39] = TagWorldPose(*pos(5.5, 6.0), yaw=0)            # Right face of (5,5)
    m[40] = TagWorldPose(*pos(7.5, 5.0), yaw=math.pi)      # Left face of (7,5)
    m[41] = TagWorldPose(*pos(7.5, 6.0), yaw=0)            # Right face of (7,5)

    # Bottom vertical pillars - column 8
    m[42] = TagWorldPose(*pos(1.5, 8.0), yaw=math.pi)      # Left face of (1,8)
    m[43] = TagWorldPose(*pos(1.5, 9.0), yaw=0)            # Right face of (1,8)
    m[44] = TagWorldPose(*pos(3.5, 8.0), yaw=math.pi)      # Left face of (3,8)
    m[45] = TagWorldPose(*pos(3.5, 9.0), yaw=0)            # Right face of (3,8)
    m[46] = TagWorldPose(*pos(5.0, 8.5), yaw=math.pi/2)    # Bottom face of (4,8)
    
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
# Grid-based path execution
# =========================

@dataclass
class PathSegment:
    """A single move or turn command"""
    cmd_type: str  # "turn" or "move"
    value: float   # degrees for turn, meters for move
    start_rc: Tuple[int, int]  # grid cell at start of segment
    end_rc: Tuple[int, int]    # grid cell at end of segment

def path_to_segments(path_rc: List[Tuple[int,int]], cell_size_m: float) -> List[PathSegment]:
    """Convert grid path to turn/move segments with exact distances"""
    if len(path_rc) < 2:
        return []
    
    segments = []
    current_yaw = 0.0  # Start facing right (0°)
    
    for i in range(len(path_rc) - 1):
        r1, c1 = path_rc[i]
        r2, c2 = path_rc[i + 1]
        
        # Determine direction of movement
        dr, dc = r2 - r1, c2 - c1
        
        # Calculate desired yaw: 0=right, π/2=down, π=left, -π/2=up
        if dc > 0:  # Moving right
            desired_yaw = 0.0
        elif dc < 0:  # Moving left
            desired_yaw = math.pi
        elif dr > 0:  # Moving down
            desired_yaw = math.pi / 2
        else:  # Moving up
            desired_yaw = -math.pi / 2
        
        # Calculate turn needed
        turn_angle = wrap_to_pi(desired_yaw - current_yaw)
        
        # Add turn segment if needed
        if abs(turn_angle) > 0.01:  # More than ~0.5 degrees
            segments.append(PathSegment(
                cmd_type="turn",
                value=math.degrees(turn_angle),
                start_rc=(r1, c1),
                end_rc=(r1, c1)
            ))
            current_yaw = desired_yaw
        
        # Add move segment (one cell)
        distance = cell_size_m * math.sqrt(dr*dr + dc*dc)
        segments.append(PathSegment(
            cmd_type="move",
            value=distance,
            start_rc=(r1, c1),
            end_rc=(r2, c2)
        ))
    
    return segments

def find_critical_tags(path_rc: List[Tuple[int,int]], tag_map: Dict[int, TagWorldPose], 
                       grid: GridMap) -> Dict[int, List[int]]:
    """
    Find critical tags at turning waypoints.
    Critical tags are directly in front of the robot at points where it needs to turn
    after completing a straight segment. Used for alignment before executing the turn.
    
    Grid alignment requirement:
    - If robot facing left/right: tag must be in same row
    - If robot facing up/down: tag must be in same column
    
    Returns: waypoint_index -> list of tag IDs that should be visible for alignment
    """
    def world_to_grid_cell(wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates back to grid cell (row, col)"""
        c = int((wx - grid.ox) / grid.cell)
        r = int((wy - grid.oy) / grid.cell)
        return r, c
    
    critical_tags = {}
    
    for i in range(1, len(path_rc) - 1):  # Skip start and goal
        r_prev, c_prev = path_rc[i-1]
        r_curr, c_curr = path_rc[i]
        r_next, c_next = path_rc[i+1]
        
        # Check if this is a turning point
        dr_in = r_curr - r_prev
        dc_in = c_curr - c_prev
        dr_out = r_next - r_curr
        dc_out = c_next - c_curr
        
        # If direction changes, this is a turning waypoint
        if (dr_in, dc_in) != (dr_out, dc_out):
            # Robot's current heading (from previous segment)
            if dc_in > 0:  # Came from left, facing right
                robot_yaw = 0.0
                is_horizontal = True
            elif dc_in < 0:  # Came from right, facing left
                robot_yaw = math.pi
                is_horizontal = True
            elif dr_in > 0:  # Came from above, facing down
                robot_yaw = math.pi / 2
                is_horizontal = False
            else:  # Came from below, facing up
                robot_yaw = -math.pi / 2
                is_horizontal = False
            
            # Robot position at turning point
            rx, ry = grid.grid_to_world_center(r_curr, c_curr)
            
            # Find tags directly in front (before the turn)
            tags_in_front = []
            max_view_distance = 1.0  # meters
            angle_tolerance = math.radians(30)  # ±30 degrees for "directly in front"
            
            for tag_id, tag_pose in tag_map.items():
                # Convert tag position to grid coordinates
                tag_r, tag_c = world_to_grid_cell(tag_pose.x, tag_pose.y)
                
                # Check grid alignment
                if is_horizontal:
                    # Facing left/right: tag must be in same row
                    if tag_r != r_curr:
                        continue
                else:
                    # Facing up/down: tag must be in same column
                    if tag_c != c_curr:
                        continue
                
                # Vector from robot to tag
                dx = tag_pose.x - rx
                dy = tag_pose.y - ry
                distance = math.hypot(dx, dy)
                
                if distance < 0.05 or distance > max_view_distance:
                    continue
                
                # Angle to tag from robot's current heading
                angle_to_tag = math.atan2(dy, dx)
                angle_diff = abs(wrap_to_pi(angle_to_tag - robot_yaw))
                
                # Tag is "in front" if aligned with current heading
                if angle_diff < angle_tolerance:
                    # Verify tag is facing toward robot
                    tag_to_robot_direction = wrap_to_pi(angle_to_tag + math.pi)
                    facing_diff = abs(wrap_to_pi(tag_pose.yaw - tag_to_robot_direction))
                    
                    if facing_diff < math.radians(60):
                        tags_in_front.append(tag_id)
            
            if tags_in_front:
                critical_tags[i] = sorted(tags_in_front)
    
    return critical_tags


# =========================
# Main
# =========================

def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    # Double grid resolution for half-cell precision
    print("\n=== Preparing Grid for Path Planning ===")
    print(f"Original grid: {OCC.shape[0]} x {OCC.shape[1]} (cell size: {CELL_SIZE_M:.3f}m)")
    
    doubled_grid = double_grid_resolution(OCC)
    doubled_cell_size = CELL_SIZE_M / 2.0  # Half the cell size
    print(f"Doubled grid: {doubled_grid.shape[0]} x {doubled_grid.shape[1]} (cell size: {doubled_cell_size:.3f}m)")
    
    # Inflate by 1 cell in doubled grid (= 0.5 cell in original)
    # Inflate obstacles - 1 cell provides 0.133m lateral clearance
    # Robot: 1.5 boxes long × 1.0 box wide (half-width = 0.133m) ✓
    inflated_grid = inflate_obstacles(doubled_grid, inflation_cells=1)
    print(f"Inflated obstacles by 1 cell in doubled grid (0.133m = robot half-width) ✓")
    print(f"Obstacles: {np.sum(doubled_grid)} -> {np.sum(inflated_grid)} cells")
    print("============================================\n")
    
    # Convert start/goal to doubled coordinates
    # Simply multiply by 2 since positions are stored as (row+0.5, col+0.5)
    doubled_start = (int(START_RC[0] * 2), int(START_RC[1] * 2))
    doubled_goal = (int(GOAL_RC[0] * 2), int(GOAL_RC[1] * 2))
    
    # Build grid and plan with doubled resolution
    grid = GridMap(inflated_grid, cell_size_m=doubled_cell_size, origin_xy=(0.0, 0.0))
    path_rc_doubled = astar(grid, doubled_start, doubled_goal)
    
    if not path_rc_doubled:
        print("No path found on the current occupancy grid.")
        print("Try adjusting start/goal positions or reducing inflation.")
        return
    
    # Convert doubled path back to original grid coordinates for display
    # (divide by 2, but path execution uses world coordinates anyway)
    path_rc = [(r//2, c//2) for r, c in path_rc_doubled]
    
    # Debug: print the planned path using DOUBLED resolution for accurate world coords
    print(f"\n=== Planned Path (TOP-LEFT origin, doubled resolution) ===")
    print(f"Start: row={START_RC[0]}, col={START_RC[1]} (doubled: {doubled_start})")
    print(f"Goal:  row={GOAL_RC[0]}, col={GOAL_RC[1]} (doubled: {doubled_goal})")
    print(f"Path has {len(path_rc_doubled)} waypoints in doubled grid:")
    for i, (r, c) in enumerate(path_rc_doubled[::2]):  # Show every other point to avoid clutter
        wx, wy = grid.grid_to_world_center(r, c)
        print(f"  {i*2:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})")
    print(f"===========================================================\n")
    
    # Convert to discrete segments using DOUBLED resolution for proper distances
    segments = path_to_segments(path_rc_doubled, doubled_cell_size)
    print(f"\n=== Path Segments ===")
    for i, seg in enumerate(segments):
        if seg.cmd_type == "turn":
            print(f"  {i:2d}: TURN {seg.value:6.1f}° at {seg.start_rc}")
        else:
            print(f"  {i:2d}: MOVE {seg.value:.3f}m from {seg.start_rc} to {seg.end_rc}")
    print(f"=====================\n")
    
    # Find critical tags at vertices using doubled resolution path
    tag_world_map = build_tag_world_map()
    critical_tags = find_critical_tags(path_rc_doubled, tag_world_map, grid)
    print(f"\n=== Critical Tags at Vertices ===")
    for waypoint_idx, tag_ids in critical_tags.items():
        r, c = path_rc_doubled[waypoint_idx]
        print(f"  Waypoint {waypoint_idx} (doubled r={r}, c={c}): Tags {tag_ids}")
    print(f"=================================\n")
    for waypoint_idx, tag_ids in critical_tags.items():
        r, c = path_rc[waypoint_idx]
        print(f"  Waypoint {waypoint_idx} (r={r}, c={c}): Tags {tag_ids}")
    print(f"=================================\n")

    # AprilTag + localization
    T_rc = build_T_rc()
    localizer = Localizer(tag_world_map, T_rc)

    # Robot setup
    ep_robot = robot.Robot()
    robomaster.config.ROBOT_IP_STR = "192.168.50.117"
    # ep_robot.initialize(conn_type="ap")
    ep_robot.initialize(conn_type="sta", sn="3JKCH8800100RC")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_360P)

    detector = AprilTagDetector(K=K, family=TAG_FAMILY, threads=2, marker_size_m=TAG_SIZE_M)
    
    # Create OpenCV window
    cv2.namedWindow("Robot Camera", cv2.WINDOW_NORMAL)
    print("Camera window opened - press 'q' to quit")

    traveled: List[Tuple[float,float]] = []
    
    # Initialize with first tag
    print("Looking for first AprilTag to initialize...")
    initial_pose = None
    while initial_pose is None:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            detections = detector.find_tags(gray)
            
            # Draw detections on display image
            display_img = img.copy()
            draw_detections(display_img, detections)
            cv2.imshow("Robot Camera", display_img)
            
            if detections:
                det = detections[0]
                initial_pose = localizer.estimate_robot_pose(det)
                if initial_pose is not None:
                    print(f"Initialized with Tag {det.tag_id}: x={initial_pose.x:.3f}, y={initial_pose.y:.3f}, yaw={math.degrees(initial_pose.yaw):.1f}°")
            else:
                ep_chassis.drive_speed(x=0.0, y=0.0, z=20.0, timeout=1)
                time.sleep(0.1)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit during initialization")
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
                ep_camera.stop_video_stream()
                ep_robot.close()
                cv2.destroyAllWindows()
                return
        except Empty:
            time.sleep(0.001)
    
    ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
    time.sleep(0.5)
    
    # Execute segments
    current_pose = initial_pose
    waypoint_idx = 0
    
    try:
        for seg_idx, seg in enumerate(segments):
            print(f"\n--- Executing Segment {seg_idx} ---")
            
            if seg.cmd_type == "turn":
                # Execute turn
                print(f"Turning {seg.value:.1f}°...")
                turn_speed = 30.0 if seg.value > 0 else -30.0
                turn_duration = abs(seg.value) / abs(turn_speed)
                
                ep_chassis.drive_speed(x=0.0, y=0.0, z=turn_speed, timeout=5)
                time.sleep(turn_duration)
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
                
                # Update expected pose
                current_pose.yaw = wrap_to_pi(current_pose.yaw + math.radians(seg.value))
                time.sleep(0.3)
                
            else:  # move
                # Execute straight move
                print(f"Moving {seg.value:.3f}m forward...")
                move_speed = 0.15
                move_duration = seg.value / move_speed
                
                ep_chassis.drive_speed(x=move_speed, y=0.0, z=0.0, timeout=5)
                time.sleep(move_duration)
                ep_chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=1)
                
                # Update expected pose
                current_pose.x += seg.value * math.cos(current_pose.yaw)
                current_pose.y += seg.value * math.sin(current_pose.yaw)
                
                waypoint_idx += 1
                time.sleep(0.3)
                
                # Check for critical tags at this waypoint
                if waypoint_idx in critical_tags:
                    print(f"At waypoint {waypoint_idx}, checking for critical tags: {critical_tags[waypoint_idx]}")
                    
                    # Look for tags
                    tag_found = False
                    for attempt in range(10):  # Try for ~0.5 seconds
                        try:
                            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                            detections = detector.find_tags(gray)
                            
                            # Draw detections on display image
                            display_img = img.copy()
                            draw_detections(display_img, detections)
                            cv2.putText(display_img, f"Looking for tags: {critical_tags[waypoint_idx]}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.imshow("Robot Camera", display_img)
                            cv2.waitKey(1)
                            
                            if detections:
                                det = detections[0]
                                if det.tag_id in critical_tags[waypoint_idx]:
                                    tag_pose = localizer.estimate_robot_pose(det)
                                    if tag_pose is not None:
                                        print(f"  ✓ Found Tag {det.tag_id}")
                                        print(f"  Expected: x={current_pose.x:.3f}, y={current_pose.y:.3f}, yaw={math.degrees(current_pose.yaw):.1f}°")
                                        print(f"  Observed: x={tag_pose.x:.3f}, y={tag_pose.y:.3f}, yaw={math.degrees(tag_pose.yaw):.1f}°")
                                        
                                        # Correct pose to align with grid
                                        current_pose = tag_pose
                                        tag_found = True
                                        break
                        except Empty:
                            pass
                        time.sleep(0.05)
                    
                    if not tag_found:
                        print(f"  ⚠ Warning: Expected tag not found, continuing with odometry")
                
                traveled.append((current_pose.x, current_pose.y))
        
        print("\n✓ Reached goal!")

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

        print(f"Traveled points: {len(traveled)}")


if __name__ == "__main__":
    main()
