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

# Positions stored as (row+0.5, col+0.5) - cell centers in grid units
# This makes doubling straightforward: just multiply by 2
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

    def grid_to_world(self, r: float, c: float) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates.
        Handles both integer grid indices and centered positions (with 0.5 offset)."""
        x = self.ox + c * self.cell
        y = self.oy + r * self.cell
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

def visualize_grid_and_path():
    """Create a comprehensive visualization of the grid, tags, and path"""
    
    # Double grid resolution and inflate
    print("Doubling grid resolution and inflating obstacles...")
    doubled_grid = double_grid_resolution(OCC)
    doubled_cell_size = CELL_SIZE_M / 2.0
    
    print(f"Original obstacles: {np.sum(OCC)} cells")
    print(f"Doubled obstacles: {np.sum(doubled_grid)} cells")
    
    # Inflate obstacles in doubled grid (adds safety margin)
    # Robot: 1.5 boxes long × 1.0 box wide (0.399m × 0.266m)
    # Half-width = 0.133m requires 1 doubled cell inflation ✓
    inflated_grid = inflate_obstacles(doubled_grid, inflation_cells=1)
    print(f"After inflation: {np.sum(inflated_grid)} cells (added {np.sum(inflated_grid) - np.sum(doubled_grid)} cells)")
    
    # Convert start/goal to doubled coordinates
    # Simply multiply by 2 since positions are stored as (row+0.5, col+0.5)
    doubled_start = (int(START_RC[0] * 2), int(START_RC[1] * 2))
    doubled_goal = (int(GOAL_RC[0] * 2), int(GOAL_RC[1] * 2))
    
    # Get the grid and plan path using doubled resolution
    grid = GridMap(inflated_grid, cell_size_m=doubled_cell_size, origin_xy=(0.0, 0.0))
    path_rc_doubled = astar(grid, doubled_start, doubled_goal)
    
    if not path_rc_doubled:
        print("No path found!")
        return
    
    # Convert path back to world coordinates for plotting
    # Use grid_to_world (without centering) since original cell centers fall on doubled grid boundaries
    path_world = [grid.grid_to_world(r, c) for r, c in path_rc_doubled]
    
    # Get tag positions
    tag_map = build_tag_world_map()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    rows, cols = OCC.shape
    doubled_rows, doubled_cols = doubled_grid.shape
    
    # First, draw all cells as white background
    for r in range(rows):
        for c in range(cols):
            x = c * CELL_SIZE_M
            y = r * CELL_SIZE_M
            
            rect = patches.Rectangle(
                (x, y), CELL_SIZE_M, CELL_SIZE_M,
                linewidth=0.5, edgecolor='gray', 
                facecolor='white', alpha=0.3
            )
            ax.add_patch(rect)
            
            # Add grid coordinates as text (small)
            cx = x + CELL_SIZE_M / 2
            cy = y + CELL_SIZE_M / 2
            ax.text(cx, cy, f'{r},{c}', 
                   ha='center', va='center', 
                   fontsize=6, color='gray', alpha=0.5)
    
    # Draw inflated areas in translucent red (half-cell resolution)
    for dr in range(doubled_rows):
        for dc in range(doubled_cols):
            # Only draw if this is an inflated area (not an original obstacle)
            is_inflated = inflated_grid[dr, dc] == 1
            is_original_obstacle = doubled_grid[dr, dc] == 1
            
            if is_inflated and not is_original_obstacle:
                # This is an inflated safety margin cell
                x = dc * doubled_cell_size
                y = dr * doubled_cell_size
                
                rect = patches.Rectangle(
                    (x, y), doubled_cell_size, doubled_cell_size,
                    linewidth=0, edgecolor='none', 
                    facecolor='red', alpha=0.3, zorder=2
                )
                ax.add_patch(rect)
    
    # Draw original obstacles as black on top
    for r in range(rows):
        for c in range(cols):
            if OCC[r, c] == 1:
                x = c * CELL_SIZE_M
                y = r * CELL_SIZE_M
                
                rect = patches.Rectangle(
                    (x, y), CELL_SIZE_M, CELL_SIZE_M,
                    linewidth=0.5, edgecolor='gray', 
                    facecolor='black', alpha=0.9, zorder=3
                )
                ax.add_patch(rect)
    
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
    if path_world:
        path_x = [p[0] for p in path_world]
        path_y = [p[1] for p in path_world]
        
        # Draw path line
        ax.plot(path_x, path_y, 
               color='blue', linewidth=3, 
               marker='o', markersize=6,
               label='A* Path', zorder=8, alpha=0.7)
        
        # Mark start and goal using stored centered positions
        # Positions are already stored as (row+0.5, col+0.5) in grid units
        start_xy = (START_RC[1] * CELL_SIZE_M, START_RC[0] * CELL_SIZE_M)
        goal_xy = (GOAL_RC[1] * CELL_SIZE_M, GOAL_RC[0] * CELL_SIZE_M)
        
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
    ax.set_title('Occupancy Grid with AprilTags and A* Path\n(Top-Left Origin, Red=Safety Margin)', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid info text
    info_text = f'Original: {rows} rows × {cols} cols\n'
    info_text += f'Doubled: {doubled_grid.shape[0]} × {doubled_grid.shape[1]}\n'
    info_text += f'Cell size: {CELL_SIZE_M:.3f} m (doubled: {doubled_cell_size:.3f} m)\n'
    info_text += f'Inflation: 1 cell (0.133m lateral clearance)\n'
    info_text += f'Robot: 1.5×1.0 boxes (0.399×0.266m), half-width=0.133m ✓\n'
    info_text += f'Start: row {START_RC[0]:.1f}, col {START_RC[1]:.1f}\n'
    info_text += f'Goal: row {GOAL_RC[0]:.1f}, col {GOAL_RC[1]:.1f}\n'
    info_text += f'Path length: {len(path_rc_doubled)} waypoints (doubled res)'
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Print path details
    print("\n=== A* Path Details ===")
    print(f"Start: grid=({START_RC[0]:.1f}, {START_RC[1]:.1f}), world=({start_xy[0]:.3f}, {start_xy[1]:.3f})")
    print(f"Goal:  grid=({GOAL_RC[0]:.1f}, {GOAL_RC[1]:.1f}), world=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")
    print(f"\nPath ({len(path_rc_doubled)} waypoints in doubled grid):")
    for i, (r, c) in enumerate(path_rc_doubled[::4]):  # Show every 4th point
        wx, wy = grid.grid_to_world(r, c)
        print(f"  {i*4:2d}: (r={r}, c={c:2d}) -> world ({wx:.3f}, {wy:.3f})")
    
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
