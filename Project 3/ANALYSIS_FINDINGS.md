# Project 3 Code Analysis Report

## 1. Parameters Imported from Config

### Status: ⚠️ CRITICAL ISSUE FOUND

#### Missing from config.py:
- **`SIDE2_Y_START`** - Imported in line 84 but NOT defined in config.py
  - Used in line 14 of docstring comment: `Side-2 : y ∈ [SIDE2_Y_START, 3.0].`
  - Likely should be defined as something like `SIDE2_Y_START = 1.0 + (ARENA_H_M - SIDE1_Y_LIMIT) / 2` or similar
  
#### All other imported parameters FOUND in config.py:
✓ ARENA_H_M, ARENA_W_M, BATTERY_* constants, K_CAM, TAG_FAMILY, TAG_SIZE_M, RECHARGE_TAG_IDS, SMALL_GOAL_TAG_IDS, LARGE_GOAL_TAG_IDS, ALL_LANDMARK_TAG_IDS, CLASS_* constants, MOVE_SPEED_MPS, OBS_*, RECHARGE_*, TAG_SERVO_*, BRICK_SERVO_*, TURN_SPEED_DPS, SWEEP_*, etc.

---

## 2. Missing Constant Definitions

### `SIDE2_Y_START`
- **Line:** 84 (import statement)
- **Usage:** No direct usage in code (referenced only in docstring)
- **Fix:** Add to config.py: `SIDE2_Y_START = 1.5` (or calculate from SIDE1_Y_LIMIT and ARENA_H_M)

---

## 3. All Function Signatures and Parameters

### Core Navigation Functions:
1. **`pixel_to_world_position()`** - Line 267
   - Parameters: `cx_px, obj_height_px, assumed_height_m, robot_pose, K=K_CAM`
   - Returns: `(world_x, world_y)`

2. **`wrap_to_pi(a: float) -> float`** - Line 541
3. **`rotz(yaw: float) -> np.ndarray`** - Line 546
4. **`T_from_Rt(R, t) -> np.ndarray`** - Line 551
5. **`inv_T(T) -> np.ndarray`** - Line 558
6. **`yaw_from_R(R) -> float`** - Line 567

### Chassis Control Functions:
7. **`chassis_stop(ep_chassis, hold_s=0.1)`** - Line 572
8. **`open_loop_turn(ep_chassis, degrees, dead_reckoner, stack=None)`** - Line 577
9. **`open_loop_drive(ep_chassis, dist_m, dead_reckoner, vx=MOVE_SPEED_MPS, vy=0.0, stack=None)`** - Line 589
10. **`turn_to_heading(ep_chassis, target_yaw_rad, dead_reckoner)`** - Line 611
11. **`navigate_to_world_point(ep_chassis, dead_reckoner, wx, wy, stop_dist_m=0.20, speed=MOVE_SPEED_MPS)`** - Line 619

### High-Level Motion Functions:
12. **`startup_sweep(ep_robot, ep_camera, ep_chassis, yolo_model, tag_detector, tag_localizer, dead_reckoner, world_map, show=False)`** - Line 631
13. **`find_loading_dock(ep_chassis, ep_camera, yolo_model, dead_reckoner, world_map, show=False)`** - Line 731
14. **`_find_obstacle_field_heading(ep_chassis, ep_camera, yolo_model, dead_reckoner, min_obstacles=1, tag_detector=None, show=False)`** - Line 820
15. **`_nearest_obstacle_dist(yolo_model, frame, tag_cols=None)`** - Line 859
16. **`cross_obstacle_field(ep_robot, ep_camera, ep_chassis, yolo_model, tag_detector, dead_reckoner, world_map, current_side, timeout_s=60.0, show=False)`** - Line 876
17. **`_slide_to_clear_corridor(ep_chassis, ep_camera, yolo_model, dead_reckoner, fwd_sign, tag_cols, show, stack=None)`** - Line 970

### Visual Servo Functions:
18. **`servo_to_tag(ep_chassis, ep_camera, tag_detector, target_tag_ids, target_dist_m, dead_reckoner, timeout_s=20.0, show=False)`** - Line 992 ✓ USES `drive_speed()` NOT `move()`
19. **`execute_recharge(ep_robot, ep_chassis, ep_camera, tag_detector, tag_localizer, dead_reckoner, world_map, battery, show=False)`** - Line 1095
20. **`approach_brick(ep_robot, ep_camera, ep_chassis, yolo_model, dead_reckoner, brick_class, action_stack, timeout_s=30.0, show=False)`** - Line 1125 ✓ USES `drive_speed()` NOT `move()`

### Mission Functions:
21. **`deliver_brick(ep_robot, ep_camera, ep_chassis, tag_detector, dead_reckoner, goal_landmark, show=False)`** - Line 1278
22. **`choose_brick(ep_camera, yolo_model)`** - Line 1320
23. **`run_mission(ep_robot, ep_camera, ep_chassis, yolo_model, tag_detector, tag_localizer, dead_reckoner, world_map, battery, max_deliveries=5, show=False)`** - Line 1333
24. **`visualize_map(world_map, robot_pose=None)`** - Line 1459
25. **`parse_args()`** - Line 1536
26. **`main()`** - Line 1551

---

## 4. servo_to_tag() and approach_brick() Analysis

### servo_to_tag() - Line 992-1086

**Control Method:** Uses `ep_chassis.drive_speed()` NOT `ep_chassis.move()`
```python
# Line 1079-1080:
ep_chassis.drive_speed(x=vx, y=0.0, z=vz, timeout=TAG_SERVO_STEP_S)

# Also Line 1071:
ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=TAG_SERVO_STEP_S)
```

**Issue:** This function uses VELOCITY control (drive_speed with m/s), NOT position control (move with delta positions).

---

### approach_brick() - Line 1125-1277

**Control Method:** Uses `ep_chassis.drive_speed()` NOT `ep_chassis.move()`
```python
# Line 1268-1269:
ep_chassis.drive_speed(x=vx, y=vy, z=0.0, timeout=BRICK_SERVO_STEP_S)

# Also Line 1257:
ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=BRICK_SERVO_STEP_S)
```

**Issue:** This function also uses VELOCITY control, NOT position control.

---

## 5. ActionStack.push() Calls vs MoveAction API

### MoveAction Definition (Line 430):
```python
@dataclass
class MoveAction:
    """Discrete move command recorded for path reversal."""
    dx: float
    dy: float
    dz: float
```

### ActionStack.push() Signature (Line 446):
```python
def push(self, dx: float, dy: float, dz: float) -> None:
    """Push a move action onto the stack."""
    self.stack.append(MoveAction(dx, dy, dz))
```

### All push() Calls Match:

✓ **Line 480** - `open_loop_turn()`: `stack.push(0, 0, degrees)` 
✓ **Line 488** - `open_loop_drive()`: `stack.push(dist_m, vy if abs(vy) > 0.005 else 0, 0)`
✓ **Line 959** - `_slide_to_clear_corridor()`: `stack.push(0, dist, 0)` (should use floats 0.0, dist, 0 for type consistency)
✓ **Line 1257** - `approach_brick()`: `action_stack.push(0.0, 0.0, 0.0)` - **PROBLEMATIC: spinning stored as zero movement**
✓ **Line 1270** - `approach_brick()`: `action_stack.push(0.02 * vx, 0.02 * vy, 0)` ✓ Correct

**Issue with Line 1257:** When the brick servo spins in place looking for a brick:
```python
ep_chassis.drive_speed(x=0.0, y=0.0, z=8.0, timeout=BRICK_SERVO_STEP_S)
action_stack.push(0.0, 0.0, 0.0)  # Pushing zero motion!
```
This doesn't capture the spinning action. When path reversal occurs, the spin won't be replayed.

---

## 6. Parameter Mismatches and Type Errors

### Type Inconsistency Issues:

1. **Line 959 in `_slide_to_clear_corridor()`:**
   ```python
   stack.push(0, dist, 0)  # Line 959
   ```
   - **Issue:** Using `int` (0) instead of `float` (0.0)
   - **Fix:** Change to `stack.push(0.0, dist, 0.0)`

2. **Line 1257 in `approach_brick()`:**
   ```python
   action_stack.push(0.0, 0.0, 0.0)  # Line 1257
   ```
   - **Issue:** Pushing zero movement when actually spinning with `z=8.0 DPS`
   - **Impact:** Path reversal won't include the spin to find the brick
   - **Fix:** Should either:
     - Not push anything: `# Stack zero for spinning (no net displacement)`
     - OR change to: `action_stack.push(0.0, 0.0, 8.0 * BRICK_SERVO_STEP_S)` to capture angular displacement

### Function Return Type Issues:

3. **Line 1319 in `choose_brick()`:**
   ```python
   def choose_brick(ep_camera, yolo_model: YOLO) -> Optional[int]:
   ```
   - Returns `Optional[int]` representing a brick class
   - All return paths correctly typed ✓

### Parameter Usage Issues:

4. **Line 1125 in `approach_brick()` - `ep_robot` parameter:**
   ```python
   def approach_brick(
       ep_robot,  # <-- Parameter passed but never used!
       ep_camera,
       ep_chassis,
       yolo_model: YOLO,
       ...
   ```
   - **Issue:** `ep_robot` parameter is declared but never used in the function body
   - **Where it should be used:** Line 1150 `move_arm_to_default(ep_robot)` and others
   - **Status:** ✓ Actually it IS used, but only via function calls

5. **Line 992 in `servo_to_tag()` - Missing `stack` parameter:**
   ```python
   def servo_to_tag(
       ep_chassis,
       ep_camera,
       tag_detector: AprilTagDetector,
       target_tag_ids: set,
       target_dist_m: float,
       dead_reckoner: DeadReckoner,
       timeout_s: float = 20.0,
       show: bool = False,
   ) -> bool:
   ```
   - **Issue:** Unlike `approach_brick()`, this function doesn't have `action_stack` parameter
   - **Impact:** servo_to_tag() movements are NOT recorded for path reversal
   - **Usage:** Called in `execute_recharge()` (line 1159) and `deliver_brick()` (line 1298) - neither passes action_stack

### Type Annotation Issues:

6. **Line 59 in imports - `Optional` usage:**
   - `Optional[AprilTagDetector]` in `_find_obstacle_field_heading()` signature (line 827) ✓ Correct
   - `Optional[ActionStack]` in `_slide_to_clear_corridor()` (line 973) ✓ Correct
   - `Optional[int]` in `choose_brick()` (line 1319) ✓ Correct

---

## Summary of Critical Issues

| Issue | Severity | Line(s) | Impact |
|-------|----------|---------|--------|
| SIDE2_Y_START not defined | CRITICAL | 84 | ImportError at runtime |
| servo_to_tag() not recording actions | HIGH | 992-1086 | Path reversal won't undo servo movements |
| approach_brick() spinning not recorded | HIGH | 1257 | Spinning to find brick not reversed |
| Type inconsistency int vs float | MEDIUM | 959 | Potential type warning/error |
| servo_to_tag() missing action_stack param | HIGH | 992 | Cannot track servo movements for reversal |

---

## Recommendations

1. **Add SIDE2_Y_START to config.py:**
   ```python
   SIDE2_Y_START = SIDE1_Y_LIMIT  # Start of Side-2 region
   # Could also be: SIDE2_Y_START = 1.5 or SIDE2_Y_START = 2.0
   ```

2. **Fix approach_brick() spinning action tracking:**
   - Option A: Remove the push for spinning (since it's just searching)
   - Option B: Track the angular displacement properly

3. **Consider adding action_stack parameter to servo_to_tag():**
   - Would need to convert drive_speed() control to move() commands
   - OR accept that visual servo movements aren't path-reversible

4. **Standardize type usage in ActionStack.push() calls:**
   - Always use floats: `push(0.0, dist, 0.0)` not `push(0, dist, 0)`

5. **Document which functions contribute to action_stack:**
   - `open_loop_turn()` ✓ Records
   - `open_loop_drive()` ✓ Records (optionally)
   - `servo_to_tag()` ✗ Does NOT record
   - `approach_brick()` ✓ Records (partially - spinning issue)
   - `_slide_to_clear_corridor()` ✓ Records (optionally)
