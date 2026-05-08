# `project_3_updated.py` Explained

This document explains what [`project_3_updated.py`](/c:/Users/dutta/Documents/cmsc477/Project%203/project_3_updated.py) does, what goal it is trying to accomplish, what each class and function is responsible for, how those pieces work together, and what assumptions must be true for the workflow to succeed.

## Overall Goal

The file implements a deterministic robot mission for Project 3.

The mission has two phases:

1. Build a simple map of the arena by following a fixed startup path.
2. Use that map to repeatedly carry the correct brick type from the loading dock to the chosen goal zone, recharging when needed.

In short, the script is trying to make the RoboMaster:

- connect to the robot and camera
- localize itself in a known 10 ft x 10 ft arena
- detect goals, obstacles, recharge, and loading dock
- choose the intended delivery goal
- pick up the right brick type
- deliver bricks repeatedly
- recharge whenever the simulated battery says it must

## Big-Picture Workflow

The file follows this sequence:

1. Start with a known pose near the top-left of the arena.
2. Detect the front goal tag and map that goal.
3. Turn left and map the recharge station from a box detection.
4. Turn back, move forward 2 ft, and translate left across the arena.
5. While translating left, map up to two obstacles and then the opposite goal.
6. Turn around and map the loading dock from visible brick detections.
7. Choose the right-side goal as the repeated delivery destination.
8. Loop:
   - recharge first if necessary
   - go to the dock
   - approach the needed brick class
   - pick it up
   - go to the goal
   - visually servo to the goal tag
   - place the brick

## Important Data Structures

### `Pose2D`

Stores the robot's estimated global pose:

- `x`: world x position in meters
- `y`: world y position in meters
- `yaw`: heading in radians

This pose is updated after every movement command, so the script always keeps a dead-reckoned estimate of where the robot is.

### `Landmark`

Represents one mapped world object as a 2D point:

- `kind`: what it is, such as `recharge`, `small_goal`, `large_goal`, `dock`, or `obstacle`
- `x`, `y`: world position
- `tag_id`: optional AprilTag ID if the landmark came from a tag

### `TagReference`

Stores a reference observation of a goal tag so the robot can later re-localize itself from a known relay point.

It remembers:

- which tag was seen
- which goal it belongs to
- where that tag is in the world
- what the robot pose was when the reference was captured
- how far the tag was from the robot
- where the tag center was in the image

### `WorldMap`

Holds the persistent mission map:

- `recharge`
- `small_goal`
- `large_goal`
- `dock`
- `intermediate`
- `dropoff_tag_ref`
- `obstacles`

Its methods help the rest of the program:

- `goal_for_kind(...)`: return the goal landmark for a goal label
- `set_goal(...)`: store the current small or large goal
- `add_or_update_obstacle(...)`: merge repeated obstacle observations that are close together
- `right_side_goal(...)`: choose the goal with the largest `x`
- `mapped_block_count(...)`: count mapped landmarks that matter during startup
- `summary(...)`: build a printable summary of the map

## Support Classes

### `BatteryManager`

This is a simple simulated battery model. It does not read the robot's real battery. Instead, it uses constants from [`config.py`](/c:/Users/dutta/Documents/cmsc477/Project%203/config.py) to decide:

- how much battery a small-brick pickup costs
- how much a large-brick pickup costs
- whether the robot can afford another pickup
- what level the battery returns to after recharging

Methods:

- `cost_for_class(...)`: returns battery cost for a brick class
- `can_pick(...)`: checks whether enough battery remains
- `consume(...)`: subtracts battery after a pickup
- `recharge(...)`: restores battery to the configured recharge level

### `AprilTagDetector`

Wraps the `pupil_apriltags` detector so the rest of the file does not need to know detector details.

Methods:

- `find_tags(...)`: run AprilTag detection on a grayscale image
- `tag_distance_m(...)`: compute the 3D distance from camera to detected tag using the estimated tag pose

This class depends on camera intrinsics, tag family, and tag size from `config.py`.

## Utility Functions

These are the low-level math and helper functions the rest of the mission uses.

### `wrap_to_pi(angle_rad)`

Normalizes any angle into the range `[-pi, pi]`.

Why it matters:

- keeps heading math stable
- avoids angle drift when turning repeatedly

### `goal_kind_from_tag(tag_id)`

Maps a detected AprilTag ID to either:

- `"small_goal"`
- `"large_goal"`
- or `None` if the tag is not a configured goal tag

### `is_obstacle_tag(tag_id)`

Returns `True` when a tag is not one of the configured:

- small goal tags
- large goal tags
- recharge tags

In this workflow, any other visible tag is treated as an obstacle tag.

### `brick_class_for_goal(goal)`

Converts a goal type into the YOLO class that should be delivered there:

- small goal -> small brick class
- large goal -> large brick class

### `rotz(yaw_rad)`

Builds a 3x3 rotation matrix for a yaw rotation.

### `transform_from_rt(R, t)`

Builds a 4x4 homogeneous transform from rotation `R` and translation `t`.

### `invert_transform(T)`

Returns the inverse of a 4x4 rigid transform.

### `yaw_from_rotation(R)`

Extracts yaw from a 3x3 rotation matrix.

### `read_frame(ep_camera, timeout=1.0)`

Safely reads a frame from the robot camera. If the camera queue is empty, it returns `None` instead of crashing.

### `pixel_bearing_rad(cx_px, camera_matrix=K_CAM)`

Converts an object's horizontal image position into a horizontal bearing angle.

### `world_from_range_and_bearing(pose, range_m, bearing_rad)`

Converts a robot-relative range and bearing into a world `(x, y)` point using the current pose estimate.

### `copy_pose(pose)`

Creates a separate copy of a pose object.

### `clamp_to_workspace(x, y)`

Clamps a point so it stays inside the known arena bounds.

### `point_to_segment_distance(px, py, ax, ay, bx, by)`

Computes how close a point is to a line segment.

This is used by obstacle-avoidance planning.

## Navigation and Obstacle Functions

### `path_blocked_by_known_obstacle(start, target_x, target_y, world_map)`

Checks whether the straight-line path from the current pose to a target passes too close to any mapped obstacle.

### `plan_navigation_points(start, target_x, target_y, world_map)`

Creates a waypoint list for navigation.

How it works:

- if there are no obstacles, go straight to the target
- if the straight path is clear, go straight
- if an obstacle blocks the path, generate two perpendicular detour candidates
- keep the shorter valid detour path
- otherwise fall back to the direct path

This is very simple path planning, but it lets the robot avoid known obstacles without a full planner.

## Vision-to-World Mapping Functions

### `estimate_bbox_distance_m(detection, cls=None)`

Estimates object distance from its bounding-box height using a calibrated model copied from another project file.

This is used for:

- recharge box mapping
- obstacle mapping
- loading dock estimation from brick detections

### `detection_world_position(detection, pose)`

Combines:

- bounding-box distance
- pixel bearing
- current robot pose

to turn a camera detection into an approximate world position.

## Motion and Pose Update Functions

### `move_robot(...)`

This is the main motion wrapper for planned movement.

It:

- sends a `chassis.move(...)` command
- waits for completion
- converts the commanded body-frame movement into world-frame movement
- updates `pose.x`, `pose.y`, and `pose.yaw`

This function is central to the script because the rest of the mission depends on pose staying consistent.

### `integrate_drive_speed(pose, vx, vy, wz_deg_s, dt_s)`

Updates the pose estimate while the robot is being controlled by short `drive_speed(...)` velocity bursts.

This is only used during close-range tag servoing.

### `turn_to_yaw(ep_chassis, pose, target_yaw_rad)`

Turns the robot to a desired heading by computing the yaw error and sending it through `move_robot(...)`.

### `navigate_to_point(...)`

Simple wrapper that calls `navigate_to_point_with_map(...)` without obstacle information.

### `navigate_to_point_with_map(...)`

Navigates to a target point using the current map.

It:

- asks `plan_navigation_points(...)` for waypoints
- turns toward each waypoint
- moves forward to that waypoint
- optionally stops short by `stop_dist_m`

This is the main coarse navigation function used throughout the mission.

## Detection Combination and Selection Functions

### `detect_tags_and_objects(frame, yolo_model, tag_detector)`

Runs both detectors on the same camera frame:

- AprilTag detector
- YOLO object detector

This keeps mapping and servo logic synchronized to one image.

### `find_best_tag(tags, valid_ids)`

Filters tags by allowed IDs and returns the closest matching tag.

### `center_error_px(cx_px)`

Computes horizontal pixel error relative to the image center.

This error is used for:

- deciding whether a landmark is centered enough to trust
- tag servo steering
- brick approach steering

## Tag-Based Mapping and Relocalization Functions

### `compute_tag_reference(tag, goal_kind, pose)`

Builds a `TagReference` from a seen goal tag.

It uses:

- the tag pose relative to the camera
- the camera-to-robot transform from `config.py`
- the robot's current world pose

to estimate the tag's world pose and store a reusable reference view.

### `landmark_from_tag_detection(tag, kind, pose)`

Converts a detected tag directly into a `Landmark` in world coordinates using:

- the robot's current world pose
- the detected tag distance
- the tag's horizontal image bearing

### `estimate_pose_from_tag_reference(tag, tag_ref)`

Uses a currently visible tag and a previously saved `TagReference` to estimate the robot's current world pose.

### `capture_intermediate_reference_if_needed(world_map, goal, tag, pose)`

Stores two things the first time an opposite goal is mapped:

- an `intermediate` waypoint at the current robot pose
- a `dropoff_tag_ref` for future relocalization

This gives the robot a repeatable relay point between dock, recharge, and goal.

### `relocalize_from_dropoff_tag(...)`

Attempts to correct pose drift by returning to the saved tag reference view.

It:

- turns to the reference heading
- searches for the saved goal tag
- matches image-center offset and distance to the reference
- computes a corrected world pose once the view is close enough

This is the script's main pose-correction strategy after dead reckoning.

### `go_to_intermediate_waypoint(...)`

Moves to the saved intermediate waypoint and optionally calls `relocalize_from_dropoff_tag(...)`.

This function is used as a hub reset before and after deliveries and recharging.

### `try_refine_recharge_from_tag(...)`

Once the robot gets near the coarse recharge location, this function tries to see an actual recharge tag and replace the coarse estimate with a better tag-based position.

### `wait_for_goal_tag(...)`

Repeatedly reads frames until a valid goal tag is seen or a timeout occurs.

## Mapping Functions

### `map_goal_from_view(...)`

Maps a goal directly from an AprilTag observation.

It:

- waits for a valid goal tag
- identifies whether it is a small or large goal
- uses the robot's current pose plus tag distance and bearing
- converts that into a world landmark
- stores the landmark in `world_map`

### `map_recharge_from_box(...)`

Maps recharge from a box detection instead of a tag.

It:

- waits for YOLO box detections
- picks the box closest to the image center
- only accepts it if centered enough
- converts it to world coordinates
- stores it as `world_map.recharge`

This is important because the recharge tag may not be visible at the first recharge view.

### `scan_left_and_map_world(...)`

This is one of the most important functions in the file.

It performs the leftward scan after the robot moves forward 2 ft.

While translating left, it:

- reads frames continuously
- maps any visible tag that is not a goal tag or recharge tag as an obstacle
- merges duplicate obstacle observations, preferably by tag identity
- falls back to centered YOLO box mapping if no obstacle tag is visible in the current frame
- ignores a fallback box if it is probably the recharge box
- searches for the goal opposite the one mapped at startup
- when that opposite goal is centered and visible, maps it from tag pose and returns

This function combines movement, detection, mapping, and stopping conditions in one loop.

### `map_loading_dock(...)`

Maps the loading dock from visible brick detections.

It:

- rotates in place in fixed steps
- collects visible small and large brick detections
- converts those detections into world points
- averages the points
- stores the centroid as the dock landmark

The key assumption is that the visible tower clump corresponds to the dock.

## Fine Alignment and Manipulation Functions

### `servo_to_visible_tag(...)`

This is the only place where `drive_speed(...)` is intentionally used.

It performs fine visual servoing to a tag by:

- rotating slowly if the tag is not visible
- measuring horizontal image error and distance error
- converting those errors into forward velocity and yaw rate
- integrating that motion into the pose estimate
- stopping when the tag is centered and the robot is at the target distance

This function is used for accurate final alignment at both the goal and recharge station.

### `approach_brick_with_move(...)`

Approaches the desired brick class at the loading dock using only `move(...)`.

It:

- looks for the target class in YOLO detections
- makes small search turns if nothing is visible
- uses box center and box height as alignment signals
- nudges forward/back and sideways with small `move(...)` steps
- requires several stable frames before calling `pick_up_tower(...)`

This function is the pickup-side counterpart to tag servoing.

### `align_to_goal_and_drop(...)`

Carries out one delivery placement.

It:

- goes to the intermediate waypoint
- coarsely navigates to the mapped goal
- fine-aligns to the goal tag with `servo_to_visible_tag(...)`
- calls `place_down_tower(...)`
- returns to the intermediate waypoint

## Recharge and Mission Control Functions

### `recharge_robot(...)`

Handles the recharge routine.

It:

- returns to the intermediate waypoint
- navigates near the mapped recharge point
- tries to refine recharge position from a visible recharge tag
- navigates closer
- tag-servos to the recharge tag
- waits 5 seconds to simulate charging
- resets the battery model
- returns to the intermediate waypoint

### `execute_mapping_sequence(...)`

This is the startup mission controller for mapping.

It hard-codes the deterministic route:

1. map the initial front goal
2. turn left and map recharge
3. turn back and move forward 2 ft
4. scan left to map obstacles and opposite goal
5. turn around and map the dock
6. choose the right-side goal as the repeated destination
7. ensure an intermediate waypoint exists

This is the function that converts the assignment's arena description into robot actions.

### `run_delivery_loop(...)`

This is the main repeated delivery controller.

It:

- checks that the dock was mapped
- determines which brick class matches the selected goal
- repeats until `max_deliveries` is reached
- recharges when battery is too low for another pickup
- navigates to the dock
- picks up the needed brick class
- decrements simulated battery
- goes back through the intermediate waypoint
- aligns to the goal and drops the brick

This function ties together navigation, battery logic, pickup, and placement.

## Visualization and Program Entry

### `visualize_map(world_map, robot_pose=None)`

Draws a simple bird's-eye debug map using `matplotlib`.

It can display:

- workspace boundary
- obstacles
- recharge
- small goal
- large goal
- dock
- intermediate waypoint
- robot pose and heading

It also saves the figure as `arena_map.png`.

### `parse_args()`

Parses command-line options such as:

- model path
- robot IP
- serial number
- connection type
- camera resolution
- map-only mode
- show-map mode
- maximum deliveries

### `main()`

This is the program entry point.

It:

- reads CLI arguments
- loads the YOLO model
- creates the AprilTag detector
- initializes pose, map, and battery state
- connects to the RoboMaster robot
- starts the video stream
- moves the arm and opens the gripper
- runs the mapping sequence
- optionally runs the delivery loop
- optionally visualizes the map
- safely stops the robot and closes hardware resources in `finally`

## How the Functions Work Together

The file is easiest to understand as five cooperating layers.

### 1. State layer

- `Pose2D`
- `Landmark`
- `TagReference`
- `WorldMap`
- `BatteryManager`

These store what the robot believes about itself, the arena, and mission readiness.

### 2. Geometry and sensing layer

- `wrap_to_pi(...)`
- `pixel_bearing_rad(...)`
- `world_from_range_and_bearing(...)`
- `estimate_bbox_distance_m(...)`
- `detection_world_position(...)`
- tag transform helpers

These functions convert camera measurements into usable world coordinates.

### 3. Motion layer

- `move_robot(...)`
- `integrate_drive_speed(...)`
- `turn_to_yaw(...)`
- `navigate_to_point_with_map(...)`
- `plan_navigation_points(...)`

These functions move the robot and keep pose updated.

### 4. Mapping and relocalization layer

- `map_goal_from_view(...)`
- `map_recharge_from_box(...)`
- `scan_left_and_map_world(...)`
- `map_loading_dock(...)`
- `capture_intermediate_reference_if_needed(...)`
- `relocalize_from_dropoff_tag(...)`

These create and maintain the world model the mission depends on.

### 5. Mission layer

- `execute_mapping_sequence(...)`
- `approach_brick_with_move(...)`
- `align_to_goal_and_drop(...)`
- `recharge_robot(...)`
- `run_delivery_loop(...)`
- `main()`

These functions sequence the whole project behavior from boot to repeated deliveries.

## Assumptions Required for This File to Work Properly

The script is strongly tied to the updated arena description. These assumptions are important.

### Arena and pose assumptions

- The workspace is 10 ft x 10 ft, or `3.048 m x 3.048 m`.
- The world origin is at the top-left corner.
- World `+x` points right.
- World `+y` points down.
- The robot starts near `(0.20 m, 0.20 m)`.
- The robot starts facing downward, which is modeled as `yaw = pi/2`.

If any of those are wrong, the mapped positions will be wrong.

### Detection assumptions

- YOLO class IDs in `config.py` match the trained model.
- The model file configured by `MODEL_PATH` exists and is correct.
- AprilTag IDs in `config.py` match the real arena tags.
- Any visible tag that is not listed as a goal tag or recharge tag is assumed to belong to an obstacle.
- Camera intrinsics `K_CAM` are close enough to reality.
- Camera extrinsics `T_ROBOT_FROM_CAMERA` are measured correctly.
- The bounding-box distance calibration is still valid for this camera setup.

If the camera or model calibration is off, world coordinates can be noticeably wrong.

### Layout assumptions

- A valid goal tag is visible in front of the robot at startup.
- After the first left turn, the recharge station can be seen as a box.
- During the leftward scan, the robot will encounter the two obstacles and then the opposite goal in a useful order.
- The loading dock appears as a visible cluster of brick detections after the 180-degree turn.
- The intended repeated drop-off destination is the right-side mapped goal, meaning the one with the largest `x`.

That last rule is not universally true in all arenas. It is a mission-specific assumption in this file.

### Motion assumptions

- `chassis.move(...)` is reliable enough for coarse navigation.
- `drive_speed(...)` is reliable enough for short tag-servo corrections.
- Dead-reckoned pose will drift, but not so badly that relocalization from the saved goal tag becomes impossible.

### Manipulation assumptions

- `pick_up_tower(...)` and `place_down_tower(...)` from `tower_utils.py` work correctly.
- The arm default positions from `config.py` are appropriate for the hardware.
- The desired pickup brick class is visible and reachable at the dock.

### Battery assumptions

- Battery management is simulated, not physical.
- The configured battery costs for small and large bricks are reasonable mission rules.
- Recharging is represented by simply waiting and resetting the simulated battery percentage.

## Short Summary

`project_3_updated.py` is a full mission script for a RoboMaster robot. It uses AprilTags, YOLO detections, dead reckoning, simple map building, obstacle-aware waypoint travel, a relay-point relocalization trick, and a simulated battery model to repeatedly deliver the correct brick type from a loading dock to a chosen goal.

The most important functions to understand first are:

- `main()`
- `execute_mapping_sequence(...)`
- `run_delivery_loop(...)`
- `move_robot(...)`
- `navigate_to_point_with_map(...)`
- `scan_left_and_map_world(...)`
- `servo_to_visible_tag(...)`

Those functions define the mission structure and show how the smaller helpers are used.
