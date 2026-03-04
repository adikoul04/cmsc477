#!/usr/bin/env python3
"""
Visualize the exact path and critical turn points used by project1_nav.py.

This script imports planning logic from project1_nav so the plotted path matches
what the robot follows at runtime.
"""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrow

from project1_nav import (
    CELL_SIZE_M,
    GOAL_CELL,
    INFLATION_SUBCELLS,
    OCC,
    PLANNING_SCALE,
    START_CELL,
    GridMap,
    astar,
    build_tag_world_map,
    double_grid_resolution,
    doubled_node_to_world,
    find_critical_tags,
    inflate_obstacles,
    remove_collinear,
)


def compute_runtime_plan() -> tuple[
    np.ndarray,
    np.ndarray,
    GridMap,
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[float, float]],
    dict[int, list[int]],
]:
    """Compute the same planning products used in project1_nav.main()."""
    doubled_occ = double_grid_resolution(OCC, scale=PLANNING_SCALE)
    doubled_cell = CELL_SIZE_M / float(PLANNING_SCALE)
    inflated = inflate_obstacles(doubled_occ, inflation_cells=INFLATION_SUBCELLS)

    center_offset = PLANNING_SCALE // 2
    start_d = (PLANNING_SCALE * START_CELL[0] + center_offset, PLANNING_SCALE * START_CELL[1] + center_offset)
    goal_d = (PLANNING_SCALE * GOAL_CELL[0] + center_offset, PLANNING_SCALE * GOAL_CELL[1] + center_offset)

    grid = GridMap(inflated, cell_size_m=doubled_cell, origin_xy=(0.0, 0.0))
    path_full = astar(grid, start_d, goal_d)
    if not path_full:
        raise RuntimeError("No path found on inflated grid")

    path_follow = remove_collinear(path_full)
    path_xy = [doubled_node_to_world(r, c, doubled_cell) for (r, c) in path_follow]

    tag_map = build_tag_world_map()
    critical_tags = find_critical_tags(path_follow, tag_map, grid)

    return doubled_occ, inflated, grid, path_full, path_follow, path_xy, critical_tags


def draw_grid(ax, occ: np.ndarray, cell_size: float, color_free="white", alpha=0.25):
    rows, cols = occ.shape
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            face = "black" if occ[r, c] == 1 else color_free
            a = 0.9 if occ[r, c] == 1 else alpha
            rect = patches.Rectangle(
                (x, y),
                cell_size,
                cell_size,
                linewidth=0.4,
                edgecolor="gray",
                facecolor=face,
                alpha=a,
                zorder=1,
            )
            ax.add_patch(rect)


def visualize_grid_and_path() -> None:
    doubled_occ, inflated, grid, path_full, path_follow, path_xy, critical_tags = compute_runtime_plan()
    tag_map = build_tag_world_map()

    rows, cols = OCC.shape
    doubled_cell = CELL_SIZE_M / float(PLANNING_SCALE)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw original map (black obstacles)
    draw_grid(ax, OCC, CELL_SIZE_M)

    # Draw inflated-only safety region (red translucent) at subcell resolution
    doubled_rows, doubled_cols = inflated.shape
    for r in range(doubled_rows):
        for c in range(doubled_cols):
            if inflated[r, c] == 1 and doubled_occ[r, c] == 0:
                x = c * doubled_cell
                y = r * doubled_cell
                rect = patches.Rectangle(
                    (x, y),
                    doubled_cell,
                    doubled_cell,
                    linewidth=0,
                    facecolor="red",
                    alpha=0.28,
                    zorder=2,
                )
                ax.add_patch(rect)

    # Draw all tags
    for tag_id, tag in tag_map.items():
        x, y, yaw = tag.x, tag.y, tag.yaw
        ax.add_patch(plt.Circle((x, y), radius=0.02, color="red", zorder=7))
        ax.add_patch(
            FancyArrow(
                x,
                y,
                0.08 * math.cos(yaw),
                0.08 * math.sin(yaw),
                width=0.012,
                head_width=0.035,
                head_length=0.028,
                color="red",
                alpha=0.8,
                zorder=6,
            )
        )
        ax.text(
            x + 0.06 * math.cos(yaw + math.pi / 2),
            y + 0.06 * math.sin(yaw + math.pi / 2),
            str(tag_id),
            ha="center",
            va="center",
            fontsize=8,
            color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.75),
            zorder=8,
        )

    # Draw full A* path (debug path before simplification)
    full_xy = [doubled_node_to_world(r, c, doubled_cell) for (r, c) in path_full]
    ax.plot(
        [p[0] for p in full_xy],
        [p[1] for p in full_xy],
        color="steelblue",
        linewidth=1.5,
        alpha=0.55,
        label=f"A* Full Path ({PLANNING_SCALE}x grid)",
        zorder=3,
    )

    # Draw robot-follow path (the actual waypoint path)
    ax.plot(
        [p[0] for p in path_xy],
        [p[1] for p in path_xy],
        color="blue",
        linewidth=3.0,
        marker="o",
        markersize=7,
        label="Robot Follow Path",
        zorder=5,
    )

    # Start/goal markers
    sx, sy = path_xy[0]
    gx, gy = path_xy[-1]
    ax.plot(sx, sy, marker="s", markersize=15, color="green", markeredgecolor="black", markeredgewidth=2, label="Start", zorder=9)
    ax.plot(gx, gy, marker="*", markersize=20, color="gold", markeredgecolor="black", markeredgewidth=1.5, label="Goal", zorder=9)

    # Mark critical turn points
    for idx, ids in sorted(critical_tags.items()):
        x, y = path_xy[idx]
        ax.plot(x, y, marker="X", markersize=12, color="purple", markeredgecolor="black", markeredgewidth=1.2, zorder=10)
        ax.text(
            x + 0.03,
            y - 0.03,
            f"wp{idx}: {ids}",
            fontsize=8,
            color="purple",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lavender", alpha=0.85),
            zorder=10,
        )

    ax.set_xlim(-0.1, cols * CELL_SIZE_M + 0.1)
    ax.set_ylim(-0.1, rows * CELL_SIZE_M + 0.1)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # top-left origin visual

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Project 1 Runtime Path + Critical Turn Points (Top-Left Origin)")
    ax.legend(loc="upper right", fontsize=9)

    info = []
    info.append(f"Original grid: {rows}x{cols}, cell={CELL_SIZE_M:.3f}m")
    info.append(f"Scaled grid: {doubled_occ.shape[0]}x{doubled_occ.shape[1]} ({PLANNING_SCALE}x), cell={doubled_cell:.3f}m")
    info.append(f"Inflation: {INFLATION_SUBCELLS} subcells ({INFLATION_SUBCELLS/PLANNING_SCALE:.2f} block = {INFLATION_SUBCELLS*doubled_cell:.3f}m)")
    info.append(f"A* full path points: {len(path_full)}")
    info.append(f"Robot follow waypoints: {len(path_follow)}")
    info.append(f"Critical turn points: {len(critical_tags)}")

    ax.text(
        0.02,
        0.98,
        "\n".join(info),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    out_path = "/Users/adikoul/Desktop/CMSC477/grid_visualization.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    print("=== Visualization Summary ===")
    print(f"Saved: {out_path}")
    print(f"Start subcell: {path_follow[0]}, world: ({sx:.3f}, {sy:.3f})")
    print(f"Goal subcell:  {path_follow[-1]}, world: ({gx:.3f}, {gy:.3f})")
    print(f"A* full points: {len(path_full)}")
    print(f"Robot follow waypoints: {len(path_follow)}")
    if critical_tags:
        print("Critical waypoints:")
        for i, ids in sorted(critical_tags.items()):
            x, y = path_xy[i]
            print(f"  wp {i:2d} at ({x:.3f}, {y:.3f}) -> tags {ids}")
    else:
        print("Critical waypoints: none")


if __name__ == "__main__":
    visualize_grid_and_path()
