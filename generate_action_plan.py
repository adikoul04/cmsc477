#!/usr/bin/env python3
"""
Generate a text action plan from project1_nav path + manual tag settings.

Output includes:
- startup alignment,
- turns,
- forward moves,
- manual alignment events.
"""

import math
from pathlib import Path

from project1_nav import (
    CELL_SIZE_M,
    GOAL_CELL,
    INFLATION_SUBCELLS,
    MANUAL_ALIGNMENT_TAGS,
    OCC,
    PLANNING_SCALE,
    START_ALIGNMENT_TAGS,
    START_CELL,
    GridMap,
    astar,
    build_execution_path_with_manual_nodes,
    build_tag_world_map,
    double_grid_resolution,
    doubled_node_to_world,
    find_manual_alignment_plan,
    inflate_obstacles,
    manual_waypoint_map,
    remove_collinear,
    wrap_to_pi,
)


def desired_yaw_from_step(dr: int, dc: int) -> float:
    if dc > 0:
        return 0.0
    if dc < 0:
        return math.pi
    if dr > 0:
        return math.pi / 2
    return -math.pi / 2


def deg(rad: float) -> float:
    return math.degrees(rad)


def main() -> None:
    out_path = Path("robot_action_plan.txt")

    # Build planning grid exactly as project1_nav main.
    scaled_occ = double_grid_resolution(OCC, scale=PLANNING_SCALE)
    subcell = CELL_SIZE_M / float(PLANNING_SCALE)
    inflated = inflate_obstacles(scaled_occ, inflation_cells=INFLATION_SUBCELLS)

    center_offset = PLANNING_SCALE // 2
    start_d = (PLANNING_SCALE * START_CELL[0] + center_offset, PLANNING_SCALE * START_CELL[1] + center_offset)
    goal_d = (PLANNING_SCALE * GOAL_CELL[0] + center_offset, PLANNING_SCALE * GOAL_CELL[1] + center_offset)

    grid = GridMap(inflated, cell_size_m=subcell, origin_xy=(0.0, 0.0))
    path_full = astar(grid, start_d, goal_d)
    if not path_full:
        raise RuntimeError("No path found on inflated planning grid.")

    path_base = remove_collinear(path_full)
    tag_map = build_tag_world_map()

    manual_plan = find_manual_alignment_plan(path_full, MANUAL_ALIGNMENT_TAGS, tag_map, grid)
    manual_nodes = [node for _, node, _ in manual_plan]
    path_exec = build_execution_path_with_manual_nodes(path_full, path_base, manual_nodes)
    manual_wp_tags = manual_waypoint_map(path_exec, manual_plan)

    path_xy = [doubled_node_to_world(r, c, subcell) for (r, c) in path_exec]

    lines = []
    lines.append("Robot Action Plan")
    lines.append("=")
    lines.append(f"Planning scale: {PLANNING_SCALE}x")
    lines.append(f"Inflation: {INFLATION_SUBCELLS} subcells ({INFLATION_SUBCELLS/PLANNING_SCALE:.2f} block)")
    lines.append(f"Start subcell: {start_d}")
    lines.append(f"Goal subcell:  {goal_d}")
    lines.append(f"Manual tags:   {MANUAL_ALIGNMENT_TAGS}")
    lines.append(f"Waypoints:     {len(path_exec)}")
    lines.append("")

    # Startup alignment phase.
    lines.append("[PHASE] STARTUP_ALIGNMENT")
    lines.append(f"ACTION: MANUAL_ALIGN tags={START_ALIGNMENT_TAGS}")

    if START_ALIGNMENT_TAGS:
        startup_tag = START_ALIGNMENT_TAGS[0]
        expected_yaw = wrap_to_pi(tag_map[startup_tag].yaw + math.pi)  # robot faces opposite tag yaw
        lines.append(
            f"NOTE: after startup alignment, assume heading = opposite(Tag {startup_tag} yaw) = {deg(expected_yaw):.1f} deg"
        )
    else:
        expected_yaw = 0.0
        lines.append("NOTE: no startup tag configured, default heading = 0.0 deg")
    lines.append("")

    total_forward = 0.0
    total_turn_abs = 0.0

    # Segment-by-segment execution plan.
    for i in range(len(path_exec) - 1):
        lines.append(f"[SEGMENT {i}] from {path_exec[i]} to {path_exec[i+1]}")

        # Manual alignment waypoint actions before continuing path.
        if i in manual_wp_tags:
            for tag_id in manual_wp_tags[i]:
                tag_yaw = tag_map[tag_id].yaw
                facing_yaw = wrap_to_pi(tag_yaw + math.pi)
                path_heading_at_wp = expected_yaw

                pre_turn = wrap_to_pi(facing_yaw - expected_yaw)
                pre_deg = deg(pre_turn)
                if abs(pre_deg) > 1.0:
                    lines.append(f"ACTION: TURN pre-align for tag {tag_id}: {pre_deg:+.1f} deg")
                    total_turn_abs += abs(pre_deg)
                    expected_yaw = facing_yaw
                else:
                    lines.append(f"ACTION: TURN pre-align for tag {tag_id}: skipped (already facing tag)")

                lines.append(f"ACTION: MANUAL_ALIGN tag={tag_id}")

                back_turn = wrap_to_pi(path_heading_at_wp - expected_yaw)
                back_deg = deg(back_turn)
                if abs(back_deg) > 1.0:
                    lines.append(f"ACTION: TURN return-to-path after tag {tag_id}: {back_deg:+.1f} deg")
                    total_turn_abs += abs(back_deg)
                else:
                    lines.append(f"ACTION: TURN return-to-path after tag {tag_id}: skipped")
                expected_yaw = path_heading_at_wp

        # Path heading turn.
        r0, c0 = path_exec[i]
        r1, c1 = path_exec[i + 1]
        dr, dc = (r1 - r0), (c1 - c0)
        desired_yaw = desired_yaw_from_step(dr, dc)
        turn = wrap_to_pi(desired_yaw - expected_yaw)
        turn_deg = deg(turn)
        if abs(turn_deg) > 1.0:
            lines.append(f"ACTION: TURN path heading: {turn_deg:+.1f} deg")
            total_turn_abs += abs(turn_deg)
        else:
            lines.append("ACTION: TURN path heading: skipped")
        expected_yaw = desired_yaw

        # Forward move distance.
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        total_forward += dist
        lines.append(f"ACTION: MOVE_FORWARD {dist:.3f} m")
        lines.append("")

    lines.append("[SUMMARY]")
    lines.append(f"Total forward distance: {total_forward:.3f} m")
    lines.append(f"Total absolute turn:    {total_turn_abs:.1f} deg")
    lines.append(f"Manual alignment count: {sum(len(v) for v in manual_wp_tags.values())}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
