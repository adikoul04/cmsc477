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

START_RC = (3, 0)
GOAL_RC = (3, 10)

@dataclass
class TagWorldPose:
    tag_id: int
    x: float
    y: float
    yaw: float

def build_tag_world_map() -> Dict[int, TagWorldPose]:
    """
    Tag placements from Figure 1.
    
    Coordinate convention (TOP-LEFT ORIGIN):
      - origin (0,0) at top-left of grid
      - x increases right, y increases down
      - cell (r,c) center: x = (c+0.5)*CELL, y = (r+0.5)*CELL
      - yaw: 0=right, π/2=down, π=left, -π/2=up
    """
    m = {}
    HALF = CELL_SIZE_M / 2  # 0.133

    # Tag 30: Block at (r=1, c=2), facing LEFT
    m[30] = TagWorldPose(30, 0.665 - HALF, 0.399, math.pi)
    
    # Tag 31: Block at (r=1, c=2), facing RIGHT
    m[31] = TagWorldPose(31, 0.665 + HALF, 0.399, 0.0)
    
    # Tag 32: Block at (r=3, c=2), facing LEFT
    m[32] = TagWorldPose(32, 0.665 - HALF, 0.931, math.pi)
    
    # Tag 33: Block at (r=3, c=2), facing RIGHT
    m[33] = TagWorldPose(33, 0.665 + HALF, 0.931, 0.0)
    
    # Tag 34: Block at (r=4, c=2), facing DOWN
    m[34] = TagWorldPose(34, 0.665, 1.197 + HALF, math.pi/2)
    
    # Tag 35: Block at (r=0, c=4), facing DOWN
    m[35] = TagWorldPose(35, 1.197, 0.133 + HALF, math.pi/2)
    
    # Tag 36: Block at (r=0, c=6), facing DOWN
    m[36] = TagWorldPose(36, 1.729, 0.133 + HALF, math.pi/2)
    
    # Tag 37: Block at (r=4, c=5), facing UP
    m[37] = TagWorldPose(37, 1.463, 1.197 - HALF, -math.pi/2)
    
    # Tag 38: Block at (r=5, c=5), facing LEFT
    m[38] = TagWorldPose(38, 1.463 - HALF, 1.463, math.pi)
    
    # Tag 39: Block at (r=5, c=5), facing RIGHT
    m[39] = TagWorldPose(39, 1.463 + HALF, 1.463, 0.0)
    
    # Tag 40: Block at (r=7, c=5), facing LEFT
    m[40] = TagWorldPose(40, 1.463 - HALF, 1.995, math.pi)
    
    # Tag 41: Block at (r=7, c=5), facing RIGHT
    m[41] = TagWorldPose(41, 1.463 + HALF, 1.995, 0.0)
    
    # Tag 42: Block at (r=1, c=8), facing LEFT
    m[42] = TagWorldPose(42, 2.261 - HALF, 0.399, math.pi)
    
    # Tag 43: Block at (r=1, c=8), facing RIGHT
    m[43] = TagWorldPose(43, 2.261 + HALF, 0.399, 0.0)
    
    # Tag 44: Block at (r=3, c=8), facing LEFT
    m[44] = TagWorldPose(44, 2.261 - HALF, 0.931, math.pi)
    
    # Tag 45: Block at (r=3, c=8), facing RIGHT
    m[45] = TagWorldPose(45, 2.261 + HALF, 0.931, 0.0)
    
    # Tag 46: Block at (r=4, c=8), facing DOWN
    m[46] = TagWorldPose(46, 2.261, 1.197 + HALF, math.pi/2)
    
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

    def grid_to_world_center(self, r: int, c: int) -> Tuple[float, float]:
        half = self.cell_size / 2.0
        wx = self.origin_x + c * self.cell_size + half
        wy = self.origin_y + r * self.cell_size + half
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
            rx, ry = grid_map.grid_to_world_center(r_curr, c_curr)
            
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

# Run path planning
grid = GridMap(OCC, cell_size_m=CELL_SIZE_M, origin_xy=(0.0, 0.0))
path_rc = astar(grid, START_RC, GOAL_RC)

print('=' * 60)
print('COMPLETE PATH OUTPUT')
print('=' * 60)

print('\n=== Grid Path ===')
print(f'Start: row={START_RC[0]}, col={START_RC[1]}')
print(f'Goal:  row={GOAL_RC[0]}, col={GOAL_RC[1]}')
print(f'Path has {len(path_rc)} waypoints:\n')

for i, (r, c) in enumerate(path_rc):
    wx, wy = grid.grid_to_world_center(r, c)
    print(f'  {i:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})')

print('\n=== Discrete Movement Segments ===')
segments = path_to_segments(path_rc, CELL_SIZE_M)
print(f'Total segments: {len(segments)}\n')

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
        wx, wy = grid.grid_to_world_center(r, c)
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
