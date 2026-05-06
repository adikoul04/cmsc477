# Project 3 Updated Workflow

This document describes the rewritten `project_3_updated.py` workflow that follows the deterministic arena routine requested for the updated setup.

## Overview

The new implementation is not a generic sweep-and-infer pipeline. It assumes a known physical setup and executes a fixed mapping path before starting repeated deliveries.

The robot:

1. Starts near `(1 ft, 1 ft)` in a `10 ft x 10 ft` workspace.
2. Uses the world frame:
   - origin at the top-left corner
   - `+x` to the right
   - `+y` downward
3. Starts facing downward, so initial yaw points along `+y`.
4. Maps:
   - the first visible drop-off zone in front
   - the recharge box after a left turn
   - two centered obstacles while translating left
   - the opposite drop-off zone
   - the loading dock after turning around
5. Repeatedly carries only the brick type that matches the right-side mapped goal.

## Workspace Geometry

- Workspace width: `10 ft = 3.048 m`
- Workspace height: `10 ft = 3.048 m`
- Start position: `(1 ft, 1 ft) = (0.3048 m, 0.3048 m)`
- Start heading: downward / `+y`

These constants live near the top of [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:69>).

## Pose Tracking

The file keeps a continuously updated robot pose:

- `Pose2D.x`
- `Pose2D.y`
- `Pose2D.yaw`

Two motion update paths exist:

- `move_robot(...)`
  - wraps `ep_chassis.move(...)`
  - updates pose after planned translations and rotations
- `integrate_drive_speed(...)`
  - updates pose during short `drive_speed(...)` servo bursts

This logic is in [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:281>) and [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:301>).

## Mapping Sequence

The deterministic startup sequence is implemented in [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:650>).

### Step 1: Map the first drop-off zone

- The robot begins facing a drop-off zone directly in front of it.
- It reads the visible AprilTag.
- The tag ID determines whether the zone is `small_goal` or `large_goal`.
- The measured range and horizontal bearing are converted into world coordinates.

Helper:

- `map_goal_from_view(...)`

### Step 2: Turn left and map the recharge box

- The robot rotates `90` degrees left.
- It expects to see the left face of the recharge box.
- Since the recharge tag may not be visible from this face, it uses YOLO box detection plus the calibrated distance estimator.
- The box is only committed when centered horizontally in the frame.

Helper:

- `map_recharge_from_box(...)`

### Step 3: Return to the original heading and move forward 2 ft

- The robot rotates `90` degrees right to face the original goal again.
- It moves forward `2 ft`.
- This places it in the corridor where the obstacle and opposite-goal scan begins.

### Step 4: Translate left and map the two obstacles

- The robot moves left in small body-frame steps.
- While translating, it checks YOLO box detections.
- An obstacle is only mapped when the box is horizontally centered in the camera frame.
- Nearby repeated measurements are merged into a single obstacle landmark.

Helper:

- `scan_left_and_map_world(...)`

### Step 5: Continue left until the opposite goal is observed

- During the same left-translation phase, the robot looks for the goal whose type is opposite the first mapped goal.
- Once detected and centered, that goal is mapped from AprilTag range and bearing.

### Step 6: Turn around and map the loading dock

- After the two obstacles and both goals are mapped, the robot turns `180` degrees.
- It looks for the tower clump that defines the loading dock.
- It uses YOLO brick detections and averages the resulting tower positions into one dock landmark.

Helper:

- `map_loading_dock(...)`

## Distance Estimation

The rewritten file reuses the calibrated bbox-height distance model from `live_feed.py` for boxes and towers:

- `OBJECT_HEIGHTS_M`
- `RAW_DISTANCE_SCALE`
- `DISTANCE_CORRECTION_SLOPE`
- `DISTANCE_CORRECTION_OFFSET`

These values are in [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:85>).

For AprilTags, distance comes directly from the tag pose estimate produced by `pupil_apriltags`.

## Motion Policy

The user requested:

- use `move(...)` whenever possible
- use `drive_speed(...)` only while going to an ArUco / AprilTag

The implementation follows that policy:

- `move(...)` is used for:
  - turns
  - waypoint navigation
  - left-translation scan
  - brick approach
- `drive_speed(...)` is used only in:
  - `servo_to_visible_tag(...)`

Relevant functions:

- [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:281>)
- [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:492>)

## Delivery Logic

After mapping is complete, the robot chooses the delivery goal on the right side of the workspace.

The current interpretation is:

- the repeated delivery destination is the mapped goal with the largest `x` value

That assumption is documented in code near:

- [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:666>)

The delivery loop is implemented in [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:690>).

The loop does this:

1. Determine the brick class required by the right-side goal.
2. Recharge first if the current battery cannot support another pickup of that class.
3. Navigate to the dock.
4. Approach only that brick class.
5. Pick it up.
6. Deduct battery based on brick type.
7. Navigate to the mapped goal.
8. Servo to the goal tag.
9. Place the brick.
10. Repeat.

## Recharge Logic

Recharge is handled by `recharge_robot(...)`.

It:

1. Navigates to the mapped recharge landmark with `move(...)`.
2. Uses `drive_speed(...)` tag servo to align with the recharge tag.
3. Waits in place to simulate recharge.
4. Restores the battery to `BATTERY_RECHARGE_LEVEL`.

## Main Functions

The most important functions in the rewritten file are:

- `move_robot(...)`
  - planned motion wrapper with pose updates
- `navigate_to_point(...)`
  - turn-then-translate waypoint motion
- `map_goal_from_view(...)`
  - map a goal from a visible AprilTag
- `map_recharge_from_box(...)`
  - map recharge from YOLO box detection
- `scan_left_and_map_world(...)`
  - map two obstacles and the opposite goal during left translation
- `map_loading_dock(...)`
  - map the dock as the centroid of visible towers
- `servo_to_visible_tag(...)`
  - fine alignment loop using `drive_speed(...)`
- `approach_brick_with_move(...)`
  - move-based brick pickup alignment
- `execute_mapping_sequence(...)`
  - deterministic startup routine
- `run_delivery_loop(...)`
  - repeated dock-to-goal transport loop

## Visualization

If requested, the script renders a simple bird's-eye map showing:

- recharge
- small goal
- large goal
- loading dock
- obstacles
- final robot pose

This is implemented in [project_3_updated.py](</c:/Users/dutta/Documents/cmsc477/Project 3/project_3_updated.py:721>).

## Important Assumptions

This version intentionally assumes the updated deterministic setup:

- the first visible front tag is one of the two goal zones
- the recharge station is visible as a box after the initial left turn
- obstacle positions are only trusted when horizontally centered
- the loading dock is visible as a cluster of tower detections after the 180-degree turn
- the repeated delivery goal is the mapped goal on the right side of the arena

If the real setup differs from those assumptions, the code path should be adjusted in `execute_mapping_sequence(...)`.
