#!/usr/bin/env python3
"""Show the complete path output for the current graph configuration"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

# Constants
CELL_SIZE_M = 0.266

OCC = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=int)

# Positions stored as (row+0.5, col+0.5) - cell centers in grid units
START_RC = (3.5, 0.5)
GOAL_RC = (3.5, 10.5)

def double_grid_resolution(grid: np.ndarray) -> np.ndarray:
    """Double the grid resolution by expanding each cell into a 2x2 block."""
    H, W = grid.shape
    doubled = np.zeros((2*H, 2*W), dtype=int)
    for r in range(H):
        for c in range(W):
            doubled[2*r:2*r+2, 2*c:2*c+2] = grid[r, c]
    return doubled

def inflate_obstacles(grid: np.ndarray, inflation_cells: int = 1) -> np.ndarray:
    """Inflate obstacles by adding a safety margin."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((2*inflation_cells+1, 2*inflation_cells+1), dtype=int)
    inflated = binary_dilation(grid, structure=struct).astype(int)
    return inflated

@dataclass
class TagWorldPose:
    tag_id: int
    x: float
    y: float
    yaw: float

def build_tag_world_map() -> Dict[int, TagWorldPose]:
    """Build the AprilTag world map
    
    Tags are positioned on cube faces with 0.5 offset from cell centers.
    Grid cell centers are at (row+0.5, col+0.5) * CELL_SIZE_M
    """
    m: Dict[int, TagWorldPose] = {}
    
    # Helper function to convert grid position to world coordinates
    def pos(row_offset: float, col_offset: float, tag_id: int, yaw: float) -> TagWorldPose:
        return TagWorldPose(tag_id, col_offset * CELL_SIZE_M, row_offset * CELL_SIZE_M, yaw)
    
    # Top horizontal bar - Tags on cells in row 1, 3, and 4, column 2
    m[30] = pos(1.5, 2.0, 30, math.pi)       # Left face of (1,2)
    m[31] = pos(1.5, 3.0, 31, 0)             # Right face of (1,2)
    m[32] = pos(3.5, 2.0, 32, math.pi)       # Left face of (3,2)
    m[33] = pos(3.5, 3.0, 33, 0)             # Right face of (3,2)
    m[34] = pos(5.0, 2.5, 34, math.pi/2)     # Bottom face of (4,2)

    # Middle horizontal bars - row 0
    m[35] = pos(1.0, 4.5, 35, math.pi/2)     # Bottom face of (0,4)
    m[36] = pos(1.0, 6.5, 36, math.pi/2)     # Bottom face of (0,6)

    # Center vertical stem - column 5
    m[37] = pos(4.0, 5.5, 37, -math.pi/2)    # Top face of (4,5)
    m[38] = pos(5.5, 5.0, 38, math.pi)       # Left face of (5,5)
    m[39] = pos(5.5, 6.0, 39, 0)             # Right face of (5,5)
    m[40] = pos(7.5, 5.0, 40, math.pi)       # Left face of (7,5)
    m[41] = pos(7.5, 6.0, 41, 0)             # Right face of (7,5)

    # Bottom vertical pillars - column 8
    m[42] = pos(1.5, 8.0, 42, math.pi)       # Left face of (1,8)
    m[43] = pos(1.5, 9.0, 43, 0)             # Right face of (1,8)
    m[44] = pos(3.5, 8.0, 44, math.pi)       # Left face of (3,8)
    m[45] = pos(3.5, 9.0, 45, 0)             # Right face of (3,8)
    m[46] = pos(5.0, 8.5, 46, math.pi/2)     # Bottom face of (4,8)
    
    return m

@dataclass
class PathSegment:
    cmd_type: str
    start_rc: Tuple[int, int]
    end_rc: Optional[Tuple[int, int]]
    value: float

class GridMap:
    def __init__(self, occupancy_grid: np.ndarray, cell_size_m: float,
                 origin_xy: Tuple[float, float] = (0.0, 0.0)):
        self.grid = occupancy_grid
        self.cell_size = cell_size_m
        self.origin_x, self.origin_y = origin_xy
        self.rows, self.cols = self.grid.shape

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r, c] == 0

    def grid_to_world(self, r: float, c: float) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates.
        Handles both integer grid indices and centered positions (with 0.5 offset)."""
        wx = self.origin_x + c * self.cell_size
        wy = self.origin_y + r * self.cell_size
        return wx, wy

def astar(grid_map: GridMap, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    start_r, start_c = start_rc
    goal_r, goal_c = goal_rc
    
    def h(r, c):
        return abs(r - goal_r) + abs(c - goal_c)
    
    q = deque()
    q.append((h(start_r, start_c), 0, start_r, start_c))
    
    g_cost = {start_rc: 0}
    parent = {}
    
    while q:
        _, g, r, c = q.popleft()
        
        if (r, c) == goal_rc:
            path = []
            cur = goal_rc
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            return path
        
        current_g = g_cost.get((r, c), float('inf'))
        if g > current_g:
            continue
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not grid_map.is_free(nr, nc):
                continue
            new_g = g + 1
            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                parent[(nr, nc)] = (r, c)
                f = new_g + h(nr, nc)
                q.append((f, new_g, nr, nc))
        
        q = deque(sorted(q, key=lambda x: x[0]))
    
    return []

def path_to_segments(path_rc: List[Tuple[int, int]], cell_size_m: float) -> List[PathSegment]:
    if len(path_rc) < 2:
        return []
    
    def direction_yaw(dr: int, dc: int) -> float:
        if dc == 1: return 0.0
        if dc == -1: return math.pi
        if dr == 1: return math.pi / 2
        if dr == -1: return -math.pi / 2
        return 0.0
    
    segments = []
    current_yaw = 0.0
    
    for i in range(len(path_rc) - 1):
        r0, c0 = path_rc[i]
        r1, c1 = path_rc[i + 1]
        
        dr = r1 - r0
        dc = c1 - c0
        
        target_yaw = direction_yaw(dr, dc)
        turn_angle = target_yaw - current_yaw
        
        while turn_angle > math.pi:
            turn_angle -= 2 * math.pi
        while turn_angle < -math.pi:
            turn_angle += 2 * math.pi
        
        if abs(turn_angle) > 0.01:
            segments.append(PathSegment('turn', (r0, c0), None, math.degrees(turn_angle)))
        
        dist = abs(dr) + abs(dc)
        move_dist = dist * cell_size_m
        segments.append(PathSegment('move', (r0, c0), (r1, c1), move_dist))
        
        current_yaw = target_yaw
    
    return segments

def find_critical_tags(path_rc: List[Tuple[int, int]], tag_map: Dict[int, TagWorldPose],
                      grid_map: GridMap) -> Dict[int, List[int]]:
    """
    Find critical tags at turning waypoints.
    Critical tags are directly in front of the robot at points where it needs to turn
    after completing a straight segment. Used for alignment before executing the turn.
    
    Grid alignment requirement:
    - If robot facing left/right: tag must be in same row
    - If robot facing up/down: tag must be in same column
    
    Returns: waypoint_index -> list of tag IDs that should be visible for alignment
    """
    def wrap_to_pi(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
    
    def world_to_grid_cell(wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates back to grid cell (row, col)"""
        c = int((wx - grid_map.origin_x) / grid_map.cell_size)
        r = int((wy - grid_map.origin_y) / grid_map.cell_size)
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
            rx, ry = grid_map.grid_to_world(r_curr, c_curr)
            
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

# Double grid resolution and inflate obstacles
print('\n=== Preparing Grid for Path Planning ===')
print(f'Original grid: {OCC.shape[0]} x {OCC.shape[1]} (cell size: {CELL_SIZE_M:.3f}m)')

doubled_grid = double_grid_resolution(OCC)
doubled_cell_size = CELL_SIZE_M / 2.0
print(f'Doubled grid: {doubled_grid.shape[0]} x {doubled_grid.shape[1]} (cell size: {doubled_cell_size:.3f}m)')

# Inflate by 1 cell in doubled grid (= 0.5 cell in original)
# Inflate obstacles - 1 cell provides 0.133m lateral clearance
# Robot: 1.5 boxes long × 1.0 box wide (half-width = 0.133m) ✓
inflated_grid = inflate_obstacles(doubled_grid, inflation_cells=1)
print(f'Inflated obstacles by 1 cell (0.133m = robot half-width) ✓')
print(f'Obstacles: {np.sum(doubled_grid)} -> {np.sum(inflated_grid)} cells')
print('==========================================\n')

# Convert start/goal to doubled coordinates
# Simply multiply by 2 since positions are stored as (row+0.5, col+0.5)
doubled_start = (int(START_RC[0] * 2), int(START_RC[1] * 2))
doubled_goal = (int(GOAL_RC[0] * 2), int(GOAL_RC[1] * 2))

# Build grid and plan with doubled resolution
grid = GridMap(inflated_grid, cell_size_m=doubled_cell_size, origin_xy=(0.0, 0.0))
path_rc = astar(grid, doubled_start, doubled_goal)

print('=' * 60)
print('COMPLETE PATH OUTPUT')
print('=' * 60)

print('\n=== Grid Path ===')
print(f'Start: row={START_RC[0]}, col={START_RC[1]}')
print(f'Goal:  row={GOAL_RC[0]}, col={GOAL_RC[1]}')
print(f'Path has {len(path_rc)} waypoints (doubled resolution):\n')

for i, (r, c) in enumerate(path_rc[::2]):  # Show every other point
    wx, wy = grid.grid_to_world(r, c)
    print(f'  {i*2:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})')

print('\\n=== Discrete Movement Segments ===')
segments = path_to_segments(path_rc, doubled_cell_size)
print(f'Total segments: {len(segments)}\\n')

for i, seg in enumerate(segments):
    if seg.cmd_type == 'turn':
        print(f'{i:2d}: TURN  {seg.value:7.1f}° at cell {seg.start_rc}')
    else:
        print(f'{i:2d}: MOVE  {seg.value:.3f}m from {seg.start_rc} to {seg.end_rc}')

print('\n=== Critical Tags at Waypoints ===')
tag_world_map = build_tag_world_map()
critical_tags = find_critical_tags(path_rc, tag_world_map, grid)

if critical_tags:
    for waypoint_idx, tag_ids in sorted(critical_tags.items()):
        r, c = path_rc[waypoint_idx]
        wx, wy = grid.grid_to_world(r, c)
        print(f'Waypoint {waypoint_idx:2d} (r={r}, c={c:2d}) at ({wx:.3f}, {wy:.3f}): Tags {tag_ids}')
else:
    print('No critical tags found along path')

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Total waypoints: {len(path_rc)}')
print(f'Total segments:  {len(segments)}')
print(f'Turns:  {sum(1 for s in segments if s.cmd_type == "turn")}')
print(f'Moves:  {sum(1 for s in segments if s.cmd_type == "move")}')
total_distance = sum(s.value for s in segments if s.cmd_type == "move")
print(f'Total distance: {total_distance:.3f}m')
print('=' * 60)
