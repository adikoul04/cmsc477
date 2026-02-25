#!/usr/bin/env python3
"""
Visualize the occupancy grid, AprilTag positions, and A* path
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import deque

# Constants from project1_nav.py
CELL_SIZE_M = 0.266

OCC = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Row 0 (top)
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 1
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 2
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Row 3
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # Row 4 (middle)
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 5
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 6
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Row 7
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=int)  # Row 8 (bottom)

START_RC = (3, 0)
GOAL_RC = (3, 10)

@dataclass
class TagWorldPose:
    x: float
    y: float
    yaw: float

class GridMap:
    def __init__(self, occ: np.ndarray, cell_size_m: float, origin_xy: Tuple[float, float]):
        self.occ = occ
        self.cell = cell_size_m
        self.ox, self.oy = origin_xy
        self.rows, self.cols = occ.shape

    def is_free(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return (self.occ[r, c] == 0)

    def grid_to_world_center(self, r: int, c: int) -> Tuple[float, float]:
        x = self.ox + (c + 0.5) * self.cell
        y = self.oy + (r + 0.5) * self.cell
        return (x, y)

def astar(grid: GridMap, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* pathfinding on the grid"""
    from heapq import heappush, heappop
    
    if not grid.is_free(*start_rc) or not grid.is_free(*goal_rc):
        return []
    
    def heuristic(r, c):
        return abs(r - goal_rc[0]) + abs(c - goal_rc[1])
    
    open_set = []
    heappush(open_set, (heuristic(*start_rc), 0, start_rc))
    
    came_from = {}
    g_score = {start_rc: 0}
    
    while open_set:
        _, current_g, current = heappop(open_set)
        
        if current == goal_rc:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_rc)
            return list(reversed(path))
        
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            if not grid.is_free(*neighbor):
                continue
            
            tentative_g = current_g + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(*neighbor)
                heappush(open_set, (f_score, tentative_g, neighbor))
                came_from[neighbor] = current
    
    return []

def build_tag_world_map() -> Dict[int, TagWorldPose]:
    """Build the AprilTag world map"""
    m: Dict[int, TagWorldPose] = {}
    HALF = 0.133  # CELL_SIZE_M / 2

    # Top horizontal bar
    m[30] = TagWorldPose(x=0.665 - HALF, y=0.399, yaw=math.pi)
    m[31] = TagWorldPose(x=0.665 + HALF, y=0.399, yaw=0)
    m[32] = TagWorldPose(x=0.665 - HALF, y=0.931, yaw=math.pi)
    m[33] = TagWorldPose(x=0.665 + HALF, y=0.931, yaw=0)
    m[34] = TagWorldPose(x=0.665, y=1.197 + HALF, yaw=math.pi/2)

    # Middle horizontal bars
    m[35] = TagWorldPose(x=1.197, y=0.133 + HALF, yaw=math.pi/2)
    m[36] = TagWorldPose(x=1.729, y=0.133 + HALF, yaw=math.pi/2)

    # Center vertical stem
    m[37] = TagWorldPose(x=1.463, y=1.197 - HALF, yaw=-math.pi/2)
    m[38] = TagWorldPose(x=1.463 - HALF, y=1.463, yaw=math.pi)
    m[39] = TagWorldPose(x=1.463 + HALF, y=1.463, yaw=0)
    m[40] = TagWorldPose(x=1.463 - HALF, y=1.995, yaw=math.pi)
    m[41] = TagWorldPose(x=1.463 + HALF, y=1.995, yaw=0)

    # Bottom vertical pillars
    m[42] = TagWorldPose(x=2.261 - HALF, y=0.399, yaw=math.pi)
    m[43] = TagWorldPose(x=2.261 + HALF, y=0.399, yaw=0)
    m[44] = TagWorldPose(x=2.261 - HALF, y=0.931, yaw=math.pi)
    m[45] = TagWorldPose(x=2.261 + HALF, y=0.931, yaw=0)
    m[46] = TagWorldPose(x=2.261, y=1.197 + HALF, yaw=math.pi/2)
    
    return m

def visualize_grid_and_path():
    """Create a comprehensive visualization of the grid, tags, and path"""
    
    # Get the grid and plan path
    grid = GridMap(OCC, cell_size_m=CELL_SIZE_M, origin_xy=(0.0, 0.0))
    path_rc = astar(grid, START_RC, GOAL_RC)
    
    if not path_rc:
        print("No path found!")
        return
    
    # Get tag positions
    tag_map = build_tag_world_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    rows, cols = OCC.shape
    
    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            # Cell bounds in world coordinates
            x = c * CELL_SIZE_M
            y = r * CELL_SIZE_M
            
            # Color based on occupancy
            if OCC[r, c] == 1:
                color = 'black'
                alpha = 0.8
            else:
                color = 'white'
                alpha = 0.3
            
            # Draw cell
            rect = patches.Rectangle(
                (x, y), CELL_SIZE_M, CELL_SIZE_M,
                linewidth=0.5, edgecolor='gray', 
                facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            
            # Add grid coordinates as text (small)
            cx = x + CELL_SIZE_M / 2
            cy = y + CELL_SIZE_M / 2
            ax.text(cx, cy, f'{r},{c}', 
                   ha='center', va='center', 
                   fontsize=6, color='gray', alpha=0.5)
    
    # Draw AprilTags
    for tag_id, tag_pose in tag_map.items():
        x, y, yaw = tag_pose.x, tag_pose.y, tag_pose.yaw
        
        # Draw tag position as a circle
        circle = plt.Circle((x, y), radius=0.02, color='red', zorder=10)
        ax.add_patch(circle)
        
        # Draw orientation arrow
        arrow_length = 0.08
        dx = arrow_length * math.cos(yaw)
        dy = arrow_length * math.sin(yaw)
        
        arrow = FancyArrow(
            x, y, dx, dy,
            width=0.015, head_width=0.04, head_length=0.03,
            color='red', alpha=0.7, zorder=9
        )
        ax.add_patch(arrow)
        
        # Label with tag ID
        label_offset = 0.06
        label_x = x + label_offset * math.cos(yaw + math.pi/2)
        label_y = y + label_offset * math.sin(yaw + math.pi/2)
        ax.text(label_x, label_y, str(tag_id), 
               ha='center', va='center', 
               fontsize=8, fontweight='bold',
               color='red', 
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.7),
               zorder=11)
    
    # Draw path
    if path_rc:
        path_xy = [grid.grid_to_world_center(r, c) for r, c in path_rc]
        path_x = [p[0] for p in path_xy]
        path_y = [p[1] for p in path_xy]
        
        # Draw path line
        ax.plot(path_x, path_y, 
               color='blue', linewidth=3, 
               marker='o', markersize=6,
               label='A* Path', zorder=8, alpha=0.7)
        
        # Mark start and goal
        start_xy = grid.grid_to_world_center(*START_RC)
        goal_xy = grid.grid_to_world_center(*GOAL_RC)
        
        ax.plot(start_xy[0], start_xy[1], 
               marker='s', markersize=15, 
               color='green', label='Start',
               zorder=12, markeredgecolor='black', markeredgewidth=2)
        
        ax.plot(goal_xy[0], goal_xy[1], 
               marker='*', markersize=20, 
               color='gold', label='Goal',
               zorder=12, markeredgecolor='black', markeredgewidth=2)
    
    # Set axis properties
    ax.set_xlim(-0.1, cols * CELL_SIZE_M + 0.1)
    ax.set_ylim(-0.1, rows * CELL_SIZE_M + 0.1)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # To match top-left origin
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Occupancy Grid with AprilTags and A* Path\n(Top-Left Origin)', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid info text
    info_text = f'Grid: {rows} rows × {cols} cols\n'
    info_text += f'Cell size: {CELL_SIZE_M:.3f} m\n'
    info_text += f'Start: row {START_RC[0]}, col {START_RC[1]}\n'
    info_text += f'Goal: row {GOAL_RC[0]}, col {GOAL_RC[1]}\n'
    info_text += f'Path length: {len(path_rc)} waypoints'
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Print path details
    print("\n=== A* Path Details ===")
    print(f"Start: row={START_RC[0]}, col={START_RC[1]}")
    print(f"Goal:  row={GOAL_RC[0]}, col={GOAL_RC[1]}")
    print(f"\nPath ({len(path_rc)} waypoints):")
    for i, (r, c) in enumerate(path_rc):
        wx, wy = grid.grid_to_world_center(r, c)
        print(f"  {i:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})")
    
    print(f"\n=== AprilTag Positions ===")
    for tag_id in sorted(tag_map.keys()):
        tag = tag_map[tag_id]
        yaw_deg = math.degrees(tag.yaw)
        print(f"  Tag {tag_id}: x={tag.x:.3f}, y={tag.y:.3f}, yaw={yaw_deg:6.1f}°")
    
    plt.tight_layout()
    plt.savefig('/Users/adikoul/Desktop/CMSC477/grid_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: grid_visualization.png")
    # plt.show()  # Comment out to avoid hanging


if __name__ == '__main__':
    visualize_grid_and_path()
