# Project1 Nav: Components and Workflow

This document explains how `project1_nav.py` works, key components, and the execution flow.

## Purpose
`project1_nav.py` solves Project 1 end-to-end:
1. Plans a shortest path on a known occupancy map.
2. Uses known block geometry to execute segment distances open-loop.
3. Uses user-defined manual tag sequence to insert alignment waypoints on the path.
4. Repeats turn-then-move segments until the goal is reached.

## Coordinate System and Map
- Origin: **top-left** of the map.
- Axes: `x` increases right, `y` increases down.
- Base occupancy map `OCC` is `9 x 11` with `CELL_SIZE_M = 0.266 m`.

## Critical Components

### 1) Grid preparation and safety inflation
- `double_grid_resolution(grid, scale=10)` expands each cell into a 10x10 subcell block.
- `inflate_obstacles(grid, inflation_cells=9)` inflates obstacles by 9 subcells.

Why this matters:
- Robot footprint is approximately `1.5 x 1.0` cubes.
- Inflating by 9 subcells at 10x scale gives a `0.9 block` barrier (`0.2394 m`) around obstacles.

### 2) Path planner
- `GridMap`: occupancy access, bounds checks, and grid->world conversion.
- `astar(grid, start, goal)`: shortest-path planning with 4-connectivity.
- `remove_collinear(path)`: simplifies dense A* path to corner waypoints actually used for tracking.
- `doubled_node_to_world(r, c, subcell_size)`: converts scaled-grid nodes to world coordinates using lattice points (`index * subcell_size`), which keeps tag/path alignment correct.

### 3) AprilTag map and localization
- `build_tag_world_map()`: known map of tag id -> `(x, y, yaw)` in world frame.
- `AprilTagDetector`: wrapper over `pupil_apriltags` detection + pose estimation.
- `center_tag_in_view(...)`:
  - detects target manual-alignment tag IDs,
  - rotates in place to center the chosen tag in the image,
  - provides heading correction only.

Detailed behavior of `center_tag_in_view(...)`:
- Inputs:
  - `ep_chassis`: robot chassis command interface.
  - `ep_camera`: camera stream interface.
  - `detector`: AprilTag detector.
  - `target_ids`: allowed tag IDs for this alignment step.
  - `timeout_s`: max correction time.
  - `tol_px`: acceptable pixel error from image center.
- Image-space target:
  - Uses horizontal image center `cx_des` (for 360p this is near pixel `x=320`).
  - Only horizontal alignment is corrected (yaw); no forward/backward distance correction is done.
- Loop logic:
  1. Read newest camera frame.
  2. Detect tags in the frame.
  3. Keep only detections whose `tag_id` is in `target_ids`.
  4. Select the best candidate detection (highest detection quality/size).
  5. Compute horizontal error: `err = detected_center_x - cx_des`.
  6. If `|err| <= tol_px`, stop rotation and return success.
  7. Otherwise command a small yaw rate proportional to error:
     - `z_cmd = clamp(-k * err, -z_max, z_max)`.
     - Tag left of center -> turn one way; tag right of center -> turn the other way.
  8. Repeat until centered or timeout.
- Fallback behavior:
  - If no target tag is visible in a loop iteration, apply a slow search spin to reacquire.
  - On timeout, stop yaw motion and return failure.

Why this is enough for this project design:
- The map provides known move distances from grid geometry.
- `center_tag_in_view` is used specifically to reduce heading drift at turns.
- Distance is intentionally not corrected from tag pose in this navigation mode.

Note:
- The script keeps localization utilities in the file, but runtime navigation now
  uses tags for orientation correction rather than continuous position tracking.

### 4) Camera-to-robot transform
- `build_T_rc()` currently returns identity.
- In this setup, this is intentional because camera and robot center are assumed aligned in `x/y` (difference is mainly `z`).

### 5) Critical turn tags
### 5) Manual alignment waypoints
- Configure `MANUAL_ALIGNMENT_TAGS` in `project1_nav.py` as an ordered list of tag IDs.
- For each tag in this list, `find_manual_alignment_plan(...)` finds the first point on the full path where:
  - if tag faces up/down: robot is vertically aligned (same subcell column),
  - if tag faces left/right: robot is horizontally aligned (same subcell row),
  - and line-of-sight is clear (not blocked by walls).
- These points are injected into the simplified execution path with
  `build_execution_path_with_manual_nodes(...)`.

### 6) Open-loop distance + heading correction
Runtime execution is segment-based:
1. Plan a simplified waypoint path (corner points).
2. For each segment:
   - if current waypoint is a manual alignment waypoint:
     - if already facing opposite the tag yaw (robot_yaw = tag_yaw + pi): skip pre-turn,
     - otherwise turn to that facing direction (90°/180° supported),
     - center that manual tag in view,
     - turn back to path heading,
   - compute desired heading from segment direction and perform timed turn,
   - compute segment distance from map geometry and perform timed forward move.
3. Continue to next segment until goal.

This design intentionally avoids using tag-estimated position for distance control.

## Runtime Workflow (High-Level)
1. Build 10x scaled and inflated grid.
2. Convert start/goal to scaled-grid node indices (`10*r+5, 10*c+5`) and plan on that lattice.
3. Run A*.
4. Simplify path to corner waypoints.
5. Build tag map and manual alignment plan.
6. Initialize RoboMaster and video stream.
7. Perform startup alignment by centering Tag 32 (known start-facing tag).
   - after this step, runtime assumes heading is opposite Tag 32 yaw.
8. Execute segments:
   - apply manual tag alignments at injected waypoints,
   - timed turn to outgoing segment heading,
   - timed forward move for known segment distance.
9. Stop robot, close camera, cleanup.

## Key Tuning Parameters
- `turn_speed_deg`: turn rate used for timed turns.
- `move_speed_mps`: forward rate used for timed distance execution.
- `center_tag_in_view` gains/tolerances:
  - image-center tolerance,
  - yaw command gain and clamp,
  - timeout.

## Outputs and Diagnostics
- Console prints planned waypoint list, manual alignment waypoints, and each segment turn/move command.
- OpenCV window is used during tag-centering and as a stop window (`q` to stop).
- `visualize_grid.py` now reproduces runtime planning path and marks manual alignment waypoints with tag IDs.
