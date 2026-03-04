# Project1 Nav: Components and Workflow

This document explains how `project1_nav.py` works, what parts are critical, and the execution flow.

## Purpose
`project1_nav.py` solves Project 1 end-to-end:
1. Plans a shortest path on a known occupancy map.
2. Uses known block geometry to execute segment distances open-loop.
3. Uses AprilTags at critical turn waypoints to correct heading by image centering.
4. Repeats turn-then-move segments until the goal is reached.

## Coordinate System and Map
- Origin: **top-left** of the map.
- Axes: `x` increases right, `y` increases down.
- Base occupancy map `OCC` is `9 x 11` with `CELL_SIZE_M = 0.266 m`.

## Critical Components

### 1) Grid preparation and safety inflation
- `double_grid_resolution(grid)` expands each cell into a 2x2 block.
- `inflate_obstacles(grid, inflation_cells=1)` inflates obstacles by 1 doubled cell.

Why this matters:
- Robot footprint is approximately `1.5 x 1.0` cubes.
- Inflating by 1 doubled cell gives a `0.133 m` (half-cube) guardrail around obstacles.

### 2) Path planner
- `GridMap`: occupancy access, bounds checks, and grid->world conversion.
- `astar(grid, start, goal)`: shortest-path planning with 4-connectivity.
- `remove_collinear(path)`: simplifies dense A* path to corner waypoints actually used for tracking.
- `doubled_node_to_world(r, c, doubled_cell)`: converts doubled-grid nodes to world coordinates using half-cell lattice points (`index * doubled_cell`), which keeps tag/path alignment correct.

### 3) AprilTag map and localization
- `build_tag_world_map()`: known map of tag id -> `(x, y, yaw)` in world frame.
- `AprilTagDetector`: wrapper over `pupil_apriltags` detection + pose estimation.
- `center_tag_in_view(...)`:
  - detects target critical tag IDs,
  - rotates in place to center the chosen tag in the image,
  - provides heading correction only.

Detailed behavior of `center_tag_in_view(...)`:
- Inputs:
  - `ep_chassis`: robot chassis command interface.
  - `ep_camera`: camera stream interface.
  - `detector`: AprilTag detector.
  - `target_ids`: allowed critical tag IDs for this waypoint.
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
- `find_critical_tags(path_rc, tag_map, grid)` identifies tags that are:
  - aligned with robot heading at turn waypoints,
  - in front of robot,
  - within a reasonable distance range,
  - facing toward the robot,
  - and visible by a line-of-sight check against map obstacles.

- `center_tag_in_view(...)` rotates robot in place so one of these tags is near the image center.

Why this matters:
- Reduces heading drift at corners, improving tracking stability.

### 6) Open-loop distance + heading correction
Runtime execution is segment-based:
1. Plan a simplified waypoint path (corner points).
2. For each segment:
   - if current waypoint is critical, center a critical tag to correct heading,
   - compute desired heading from segment direction and perform timed turn,
   - compute segment distance from map geometry and perform timed forward move.
3. Continue to next segment until goal.

This design intentionally avoids using tag-estimated position for distance control.

## Runtime Workflow (High-Level)
1. Build doubled and inflated grid.
2. Convert start/goal to doubled-grid node indices (`2*r+1, 2*c+1`) and plan on that lattice.
3. Run A*.
4. Simplify path to corner waypoints.
5. Build tag map and critical-tag map.
6. Initialize RoboMaster and video stream.
7. Execute segments:
   - re-center using critical tags at turn waypoints,
   - timed turn to outgoing segment heading,
   - timed forward move for known segment distance.
8. Stop robot, close camera, cleanup.

## Key Tuning Parameters
- `turn_speed_deg`: turn rate used for timed turns.
- `move_speed_mps`: forward rate used for timed distance execution.
- `center_tag_in_view` gains/tolerances:
  - image-center tolerance,
  - yaw command gain and clamp,
  - timeout.

## Outputs and Diagnostics
- Console prints planned waypoint list, critical waypoints, and each segment turn/move command.
- OpenCV window is used during tag-centering and as a stop window (`q` to stop).
- `visualize_grid.py` now reproduces runtime planning path and marks critical turn points with tag IDs.
