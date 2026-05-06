# Project 3: Warehouse Logistics Workflow

This document explains how `project_3_updated.py` works, its key components, the execution flow, and all safety mechanisms including boundary enforcement and automatic start-corner detection.

## Purpose

`project_3_updated.py` solves Project 3 end-to-end:

1. **Initialization**: Seeds dead reckoning position from CLI start-corner flag or detects it automatically during initial sweep.
2. **Startup Sweep**: Scans the arena perimeter to:
   - Locate and map all AprilTag landmarks (dock, goal zones, recharge station).
   - Detect brick piles and obstacles using YOLO.
   - **Auto-detect start corner** via frame-based dock-vs-recharge-like relative depth comparison.
   - Calibrate camera parameters if needed.
3. **World Map Construction**: Builds a persistent world frame with known landmark positions.
4. **Mission Execution** (Phase state machine):
   - **SIDE 1** (y ∈ [0,1]): Pick up brick → Navigate to recharge → Recharge battery.
   - **Cross**: Drive through obstacle field to SIDE 2.
   - **SIDE 2** (y ∈ [2,3]): Drop off brick → Navigate to goal → Drop brick.
   - **Return**: Cross back to SIDE 1 and repeat.
5. **Safety**: Enforce 0.2m boundary margin everywhere—dead-reckoning updates, motion planning, visual servos.

---

## Coordinate System and Arena Layout

### Physical Arena
- **Dimensions**: 3.048 m × 3.048 m (10 ft × 10 ft).
- **Origin**: Top-left corner of the physical arena.
- **Axes**:
  - `x` increases to the right (0 to 3.048).
  - `y` increases downward (0 to 3.048).

### Side Division
- **SIDE 1**: y ∈ [0.0, 1.0] (top 1/3 of arena).
  - Brick pickup zone, loading dock (cluster of bricks).
  - Recharge station (right edge, y ≈ 0.5).
- **SIDE 2**: y ∈ [2.0, 3.0] (bottom 1/3 of arena).
  - Goal drop zones (left and right goal towers).
  - Recharge station (right edge, y ≈ 2.5).
- **Obstacle Field** (middle y-zone): No landmarks; used for navigation between sides.

### Safety Boundaries
- **Safe Arena Interior**: After applying 0.2m boundary margin on all sides:
  - x ∈ [0.2, 2.848], y ∈ [0.2, 2.848].
- **Margin Purpose**: Keep robot 20cm from physical arena edges to prevent collision with cones placed on the perimeter.
- **Enforcement Points**:
  - Dead-reckoning pose integration (`clamp_pose_to_safe_arena`).
  - Motion planning / path following (`navigate_to_world_point`).
  - Visual servo motion commands (`servo_to_tag`, `approach_brick`).
  - Open-loop driving (`open_loop_drive`).

---

## Critical Components

### 1) Dead Reckoning and World State

#### `DeadReckoner` Class
Maintains integrated odometry-based pose (x, y, yaw):
- Initialized from assumed start corner (CLI flag: `--start-corner top-left` or `top-right`).
- Updated incrementally via chassis odometry readings.
- **Boundary-clamped**: After each update, pose is clamped to safe interior [0.2, 2.848] × [0.2, 2.848].

**Absolute Position Mapping**:
- Start corner flag determines initial world frame:
  - `top-left` → dead_reckoner.pose = (0.2, 0.2) (top-left safe corner).
  - `top-right` → dead_reckoner.pose = (2.848, 0.2) (top-right safe corner, x-mirrored arena).

#### `WorldMap` Class
Persistent dictionary of landmarks detected during initial scan:
```python
world_map = {
    "dock": Landmark(...),
    "recharge": Landmark(...),
    "large_goal_0": Landmark(...),
    "large_goal_1": Landmark(...),
    "goal_0": Landmark(...),
    "goal_1": Landmark(...),
}
```
Each landmark stores:
- `world_pos`: (x, y) in absolute arena coordinates.
- `tag_id`: AprilTag ID (if detected).
- `yaw_rad`: Orientation.
- `confidence`: How recently / reliably detected.

---

### 2) Automatic Start-Corner Detection (Frame-Based Inference)

**Problem**: Robot starts at either **top-left** or **top-right** but cannot determine which from the first video frame alone, especially if the recharge station is occluded (wrong face visible, no AprilTag seen).

**Solution**: During the initial sweep, compare relative **depths** (ranges) of the dock cluster and recharge-like obstacles in the **same video frame**. From this comparison, infer the corner and vote on it.

#### `infer_start_corner_from_frame(dock_range_m, recharge_like_range_m)`
**Logic**:
- Compares effective camera distance to dock cluster vs. distance to recharge-like object.
- **If dock_range < recharge_like_range**:
  - Dock is closer (camera sees loading dock first) → Robot is near loading dock side.
  - Verdict: `"top-right"` (in standard orientation, dock is on right side of SIDE 1).
- **If dock_range ≥ recharge_like_range**:
  - Recharge-like object is closer (already at or past dock) → Robot is past the loading dock.
  - Verdict: `"top-left"` (started on left side or swept past dock first).

#### Startup Sweep with Frame Voting
During `startup_sweep()`, for each video frame:
1. **Detect dock cluster**: ≥2 bricks in image, compute centroid and average range.
2. **Detect recharge candidate**:
   - If AprilTag visible (RECHARGE_TAG_IDS = {8, 10}): Use its range.
   - Otherwise, find nearby obstacle (CLASS_BOX) close to dock centroid as proxy for recharge.
3. **Compute frame-based corner**: Call `infer_start_corner_from_frame()`.
4. **Vote**: Append corner hypothesis to `corner_votes` list.
5. **After sweep completes**:
   - Take majority vote from `corner_votes`.
   - If **inferred corner ≠ assumed start corner (CLI flag)**:
     - Call `apply_start_corner_correction()` to shift entire world frame x-coordinate.
     - All landmarks, dead-reckoner pose updated consistently.

#### Fallback Logic
If frame-based voting produces no clear result:
- Use older range-only heuristic: `infer_start_corner()` compares recharge vs. dock raw distances.
- If both fail, keep the seeded corner (CLI flag assumption).

**Why This Works**:
- Relative depth ordering is **frame-invariant**: Whether recharge is 1.5m or 0.8m away, if it's *closer* than dock, corner vote is consistent.
- Dock + recharge guarantees appearance in same frame at start (both on same side of arena).
- Proxy detection (nearby obstacle as recharge stand-in) handles the occlusion case.

---

### 3) Boundary Safety Mechanisms

#### 3a) Pose Clamping
**Function**: `clamp_pose_to_safe_arena(pose)`
```
Input:  pose (x, y, yaw)
Output: (clamped_x, clamped_y, yaw)
where:  x ∈ [0.2, 2.848], y ∈ [0.2, 2.848]
```
- **Applied after every odometry integration** in dead reckoner.
- Prevents pose from drifting into the boundary zone (outside [0.2, 2.848]).

#### 3b) Motion Translation Clamping
**Function**: `clamp_body_translation_to_safe_arena(pose, dx_body, dy_body)`
```
Input:  Current pose (x, y, yaw), desired translation (dx_body, dy_body) in body frame
Output: Scaled translation (scaled_dx, scaled_dy) that stays within arena interior
```
- **Algorithm**:
  1. Convert body-frame translation to world frame using current yaw.
  2. Compute final world position if translation applied unscaled.
  3. If final position exceeds boundaries, compute **scaling factor** (0.0 to 1.0) to just touch boundary.
  4. Return scaled translation.
- **Applied in**:
  - `open_loop_drive()`: Before issuing chassis move command.
  - `servo_to_tag()`: Before incrementing velocity toward tag (x-body direction).
  - `approach_brick()`: Before incrementing velocity during brick approach (x/y-body directions).

#### 3c) Path Target Clamping
**Function**: `navigate_to_world_point(world_target_x, world_target_y, ...)`
- Clamps world target to safe interior before planning path.
- Ensures planner never issues commands toward boundary.

#### 3d) Visual Servo Clamping
**Functions**: `servo_to_tag()`, `approach_brick()`
- Both compute incremental velocity steps (e.g., `vx_step = 0.1 m/s` per iteration).
- Before applying velocity command to chassis, clamp velocity to `clamp_body_translation_to_safe_arena()`.
- Prevents visual servo loops from drifting the robot edge-ward.

---

### 4) Object Detection and Classification

#### YOLO Integration
- **Model**: YOLOv8n or v11n running on each camera frame.
- **Classes**:
  - `CLASS_CONE = 0`: Red/orange cones (on arena perimeter, not tracked).
  - `CLASS_BOX = 1`: Generic boxes/obstacles.
  - `CLASS_SMALL_BRICK = 2`: Small brick units (blocks for transfer).
  - `CLASS_LARGE_BRICK = 3`: Large brick units.
- **Uses**:
  - **Startup sweep**: Detect brick piles (doc cluster), obstacles, and recharge-like boxes.
  - **Mission execution**: Approach and pickup bricks, detect blocking obstacles for navigation.

#### AprilTag Detection
- **Library**: `pupil_apriltags`, tag family 36h11.
- **Calibration**:
  - Camera matrix `K_CAM`: 3×3 intrinsic calibration.
  - Markers expected 0.075 m (7.5 cm) per side.
  - Pose estimation relative to robot camera frame.

#### Tag ID Assignments
- **RECHARGE_TAG_IDS** = {8, 10}: Recharge stations on SIDE 1 and SIDE 2.
- **SMALL_GOAL_TAG_IDS** = {27, 30}: Small goal zones (tag 27 on SIDE 1, tag 30 on SIDE 2).
- **LARGE_GOAL_TAG_IDS** = {34, 38}: Large goal zones (tag 34 on SIDE 1, tag 38 on SIDE 2).
- **DOCK_TAG_ID** (implicit): Dock area (no dedicated tag; identified as cluster of ≥2 bricks).

---

### 5) Visual Servo Loops

#### 5a) Tag Servo: `servo_to_tag(target_tag_id, approach_distance_m)`
**Purpose**: Align robot heading and position to face a specific AprilTag head-on at a target distance.

**Flow**:
1. Scan camera frames for target tag.
2. If found:
   - Extract tag pose relative to camera.
   - Compute heading error (robot yaw vs. tag facing).
   - Drive forward/backward to reach target distance.
   - Rotate in place to align heading.
3. Once heading aligned and at target distance → servo complete.

**Boundary Enforcement**:
- Before each motion command, clamp velocity to stay within arena interior.

#### 5b) Brick Servo: `approach_brick(brick_detection, approach_distance_m)`
**Purpose**: Position gripper to grab a brick from YOLO detection.

**Flow**:
1. Use YOLO brick detection bounding box.
2. Estimate brick 3D position from bounding box height (similar to range estimation).
3. Iteratively move robot to center brick in frame and reach approach distance.
4. Incremental vx, vy commands with visual feedback.
5. Open gripper, back up, close gripper.

**Boundary Enforcement**:
- Every vx, vy step clamped via `clamp_body_translation_to_safe_arena()`.

---

### 6) World Map Correction (Frame Shift)

#### Problem: Correcting for Wrong Start Corner Assumption

If the robot started at `top-left` (from CLI) but startup sweep infers `top-right`:
- All dead-reckoned positions are x-mirrored.
- All detected landmarks need to be shifted to match corrected frame.

#### Solution: `shift_world_map_x(world_map, delta_x)`
- Iterates all landmarks in world_map.
- For each landmark, shifts x-coordinate: `landmark.x += delta_x`.
- Also shifts dead reckoner pose: `pose.x += delta_x`.

#### `apply_start_corner_correction(assumed_corner, inferred_corner, dead_reckoner, world_map)`
1. Determine δx:
   - If assumed = `top-left`, inferred = `top-right`: δx = 2.648 (shift right).
   - If assumed = `top-right`, inferred = `top-left`: δx = -2.648 (shift left).
2. Call `shift_world_map_x(world_map, delta_x)`.
3. Shift dead reckoner pose by δx.
4. Log correction (landmark positions updated).

---

### 7) Obstacle Reconciliation

#### Problem: Recharge Station Mapped as Obstacle

During YOLO detection, if the recharge station is visible but its AprilTag isn't (wrong face facing robot), the recharge block appears as a generic `CLASS_BOX` obstacle in the world map.

#### Solution: `reconcile_recharge_obstacles(world_map)`
- After confirming recharge position from tag detection (or frame-based inference):
  - Get recharge landmark position from world_map.
  - Search all obstacle entries.
  - If any obstacle is within 0.4m distance of recharge, flag it as a false positive.
  - Remove false-positive obstacles.

**Result**: World map maintains only true obstacles; recharge is not double-counted.

---

## Mission Execution Workflow

### Phase 1: Startup and Calibration
1. **Parse CLI Arguments**:
   - `--start-corner {top-left, top-right}`: Seed position hypothesis.
   - `--show`: Display sweep frames for debugging.
2. **Initialize Dead Reckoner**: Position = assumed start corner safe boundary.
3. **Initialize World Map**: Empty, will populate during sweep.

### Phase 2: Initial Sweep
**Entry point**: `startup_sweep()`

1. **Rotate 360° in place**, collecting frames every ~10°:
   - Detect YOLO objects (bricks, boxes, cones).
   - Detect AprilTags.
   - For each frame with ≥2 bricks + recharge candidate:
     - Compute dock and recharge-like ranges.
     - Call `infer_start_corner_from_frame()`.
     - Vote on corner.
2. **Post-sweep**:
   - Majority vote corner (if any votes collected).
   - If **inferred ≠ assumed**: Apply frame shift via `apply_start_corner_correction()`.
   - Build final `world_map` with all detected landmarks.
   - Call `reconcile_recharge_obstacles()` to clean up false obstacles.

**Output**: 
- Corrected dead_reckoner pose (if corner was inferred differently).
- Populated world_map with dock, recharge, goal zones.

### Phase 3: Mission Execution

#### State Machine: Two Sides with Cross-Zone Transitions

**SIDE 1 Phase** (y ∈ [0, 1]):
1. **PICK_UP_BRICK**: 
   - Navigate to dock (cluster of bricks).
   - Use brick servo to position gripper.
   - Grip and lift brick.
2. **APPROACH_RECHARGE**: 
   - Navigate to recharge station (right side, y ≈ 0.5).
   - Use tag servo to align with recharge AprilTag.
   - Grasp charging connector (simulated: back up, pause, move forward).
3. **RECHARGE**: 
   - Hold position for recharge time (e.g., 3 seconds).
4. **CROSS_TO_SIDE2**: 
   - Enter obstacle field to (x ≈ 1.5, y ≈ 1.5).
   - Navigate to SIDE 2 entry point.
   - Transition to SIDE 2.

**SIDE 2 Phase** (y ∈ [2, 3]):
1. **DROP_OFF_BRICK**: 
   - Navigate to goal zone (left: x ≈ 0.5, or right: x ≈ 2.5).
   - Use brick servo to position brick over goal.
   - Open gripper and release.
2. **APPROACH_RECHARGE** (optional): 
   - Navigate to recharge station (right side, y ≈ 2.5).
   - Recharge if needed.
3. **CROSS_TO_SIDE1**: 
   - Navigate back through obstacle field.
   - Re-enter SIDE 1.
4. **Return to PICK_UP_BRICK** state.

#### Navigation Commands
**`navigate_to_world_point(target_x, target_y, ...)`**:
- Plans path from current pose to target.
- Uses open-loop driving with periodic landmark re-localization.
- Clamps target to safe interior.

**`open_loop_drive(distance_m, heading_rad)`**:
- Simple linear motion for a fixed distance at fixed heading.
- Boundary-clamped to prevent edge overrun.

---

## Key Helper Functions

### Position and Boundary Utilities
- **`clamp_pose_to_safe_arena(pose)`**: Enforces [0.2, 2.848] × [0.2, 2.848].
- **`clamp_body_translation_to_safe_arena(pose, dx, dy)`**: Scales motion to stay in bounds.
- **`shift_world_map_x(world_map, delta_x)`**: Corrects all landmarks if corner inference changes frame.

### Corner Inference
- **`infer_start_corner_from_frame(dock_range_m, recharge_like_range_m)`**: Frame-based depth comparison.
  - Returns `"top-right"` if dock_range < recharge_range.
  - Returns `"top-left"` if dock_range ≥ recharge_range.
- **`infer_start_corner(recharge_range_m, dock_range_m)`**: Fallback range-only heuristic.
- **`apply_start_corner_correction(...)`**: Shifts world frame and dead reckoner if needed.

### Obstacle Management
- **`reconcile_recharge_obstacles(world_map)`**: Removes obstacles overlapping recharge landmark (within 0.4m).

### Servo Control
- **`servo_to_tag(target_tag_id, approach_distance_m, timeout_s)`**: Head-on tag alignment.
- **`approach_brick(brick_det, approach_distance_m)`**: Position gripper for brick pickup.

---

## Safety and Edge Cases

### Boundary Enforcement Layers

1. **Dead Reckoning Layer** (highest priority):
   - Post-integration clamping ensures pose never leaves safe interior.
   - Prevents cascading errors in navigation.

2. **Motion Planning Layer**:
   - Target clamping ensures planned destinations are reachable.
   - Translation clamping scales motion to boundary if needed.

3. **Visual Servo Layer** (lowest priority, last defense):
   - Incremental velocity steps checked and scaled.
   - Prevents loop drift from pushing robot into boundary.

### Recharge Station Occlusion

**Scenario**: Robot performs startup sweep, but recharge block presents its non-tagged face to camera.
- **Symptom**: YOLO detects a `CLASS_BOX` obstacle at recharge location, but no AprilTag visible.
- **Detection**: During frame-vote collection, a dock cluster + nearby box in same frame triggers corner vote.
- **Correction**: Frame-based inference votes corner. After sweep, frame shift ensures landmarks match inferred corner.
- **Outcome**: Recharge mapped correctly despite tag occlusion; reconciliation removes the false obstacle entry.

### Start Corner Ambiguity

**Scenario**: Robot seeded at `top-left` (CLI), but startup sweep infers `top-right`.
- **Detection**: Frame votes collect majority `"top-right"`.
- **Action**: `apply_start_corner_correction()` shifts all landmarks by Δx = 2.648.
- **Result**: World map and dead reckoner now consistent with inferred frame.

---

## Configuration Parameters

### Arena Geometry
```python
ARENA_W_M = 3.048  # Width (x-extent)
ARENA_H_M = 3.048  # Height (y-extent)
SAFE_BOUNDARY_MARGIN_M = 0.20  # Safety clearance from all edges
SAFE_INTERIOR_X = [0.2, 2.848]
SAFE_INTERIOR_Y = [0.2, 2.848]
```

### Side Definitions
```python
SIDE1_Y_RANGE = (0.0, 1.0)  # Pickup side
SIDE2_Y_RANGE = (2.0, 3.0)  # Dropoff side
```

### Object Detection
```python
CLASS_CONE = 0
CLASS_BOX = 1
CLASS_SMALL_BRICK = 2
CLASS_LARGE_BRICK = 3
```

### AprilTag IDs
```python
RECHARGE_TAG_IDS = {8, 10}
SMALL_GOAL_TAG_IDS = {27, 30}
LARGE_GOAL_TAG_IDS = {34, 38}
```

### Servo Parameters
```python
TAG_APPROACH_DISTANCE_M = 0.3  # Distance to tag for head-on alignment
BRICK_APPROACH_DISTANCE_M = 0.15  # Distance to brick for gripper approach
SERVO_TIMEOUT_S = 10.0  # Maximum servo time before timeout/fallback
```

---

## Usage

### Basic Execution
```bash
python project_3_updated.py --start-corner top-left
```

### With Visualization
```bash
python project_3_updated.py --start-corner top-left --show
```

### With Manual Corner Override
```bash
python project_3_updated.py --start-corner top-right
```
(Startup sweep will still attempt auto-detection; if inferred corner differs, world frame is corrected.)

---

## Debugging and Monitoring

### Key Logging Points
1. **Startup Sweep**:
   - Frame-by-frame dock detections, recharge candidate detections.
   - Inferred corner per frame and majority vote result.
   - Corner correction applied (if any).

2. **Mission Execution**:
   - Current phase, current state.
   - Pose updates (dead reckoned).
   - Navigation waypoints, landing completion.
   - Servo successes/timeouts.

3. **Boundary Clamping**:
   - Original vs. clamped pose (indicates boundary touch).
   - Scaled motion commands (indicates collision avoidance).

### Visual Debugging (--show flag)
- **Startup sweep frames** annotated with:
  - Detected bricks (boxes) overlaid.
  - Detected AprilTags with pose.
  - Detected recharge-like candidates.
  - Frame-voting result per frame.

---

## Summary

`project_3_updated.py` is a complete warehouse logistics solution that:

✅ **Maintains 0.2m boundary safety** at all times (dead reckoning, motion planning, visual servos).
✅ **Auto-detects start corner** via frame-based dock-vs-recharge depth inference during startup sweep.
✅ **Handles recharge occlusion** by detecting nearby obstacles as proxies when AprilTags aren't visible.
✅ **Corrects world frame** consistently when startup corner inference differs from CLI assumption.
✅ **Executes multi-phase logistics** (pick brick, recharge SIDE 1 → drop brick SIDE 2 → return).
✅ **Provides fallback logic** (range-only heuristic if frame votes fail; seeded corner if all else fails).

All safety and inference mechanisms are integrated and tested for compilation correctness.

