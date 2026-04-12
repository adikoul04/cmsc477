# Swap-and-Place Notes

This note documents the complete swap-and-place pipeline and the math used in `swap_and_place.py` to track robot pose and navigate between targets.

## What The Script Does

The script executes the full two-tower swap task with RoboMaster:

1. Detect both towers in camera view.
2. Go to tower 1 (left side), align by yaw, drive straight in, and pick it up.
3. Turn left 90 degrees, move to temporary placement area, place tower 1.
4. Return to home pose and heading.
5. Go to tower 2 (right side), align and pick it up.
6. Move to tower 1's original pose and place tower 2 there.
7. Return home, search for moved tower 1 while excluding tower 2.
8. Reacquire and pick tower 1.
9. Move to tower 2's original pose and place tower 1.

## Why This Structure

The behavior is organized into two layers:

- Visual approach layer:
  - in `tower_utils.py`, `go_to_tower()` handles camera-based approach.
  - logic is rotate-first, then forward-only to top-y stopping target.
- Geometry layer:
  - in `swap_and_place.py`, the script tracks `(x, y, yaw)` from commanded motion.
  - this tracked pose is used to return home and move between saved slot poses.

This split keeps close-range control robust to bbox clipping while still enabling deterministic point-to-point motions.

## Core Data Structures

- `Pose2D(x_m, y_m, yaw_deg)`:
  - saved robot pose targets for tower slots and navigation goals.
- `PoseTracker`:
  - live estimate of robot home-relative pose integrated from commands.
- `Detection`:
  - YOLO detection fields used by approach and reacquisition.

## Full Pipeline Walkthrough

### 1) Initial detection phase

- The system waits for a stable pair of tower detections.
- Leftmost and rightmost towers define initial visual identities for tower 1 and tower 2.

### 2) Tower 1 pickup and slot recording

- `go_to_tower(..., selection_mode="leftmost")`:
  - rotate until centered,
  - drive forward to top-y target.
- Immediately after approach, current tracked pose is saved:
  - `tower1_slot = tracker.current_pose()`.
- This stores tower 1 original placement as full `(x, y, yaw)`.

### 3) Temporary placement and home return

- Turn left 90 degrees.
- Move forward to temporary drop location.
- Place tower 1.
- Return to home `(0, 0, 0)` via geometric solver.

### 4) Tower 2 pickup and slot recording

- `go_to_tower(..., selection_mode="rightmost")`.
- Save full original slot pose:
  - `tower2_slot = tracker.current_pose()`.
- Pick tower 2.

### 5) Place tower 2 in tower 1 original slot

- Solver moves to `tower1_slot`:
  - position match `(x, y)`,
  - orientation match `yaw`.
- Place tower 2.

### 6) Reacquire moved tower 1 from home

- Return to home pose.
- Sweep search across rows and lateral spans.
- During sweep, exclude tower 2 using predicted image x location from known world pose and current tracked pose.
- Select remaining valid tower as reacquired tower 1.

### 7) Final pickup and placement

- Approach reacquired tower 1 with same rotate-then-forward strategy.
- Pick tower 1.
- Move to `tower2_slot` with full pose matching and place.

## Pose Tracking Math

Pose tracking is command-integrated (dead reckoning) relative to home.

### Heading integration

If yaw rate command is `z` (deg/s) for duration `dt`:

$$
\Delta \psi = z \cdot dt
$$

$$
\psi_{k+1} = \mathrm{wrapTo180}(\psi_k + \Delta \psi)
$$

### Body-to-world translation integration

For commanded body motion `(u_f, u_l)` at heading `\psi`:

$$
\Delta x = u_f \cos \psi - u_l \sin \psi
$$

$$
\Delta y = u_f \sin \psi + u_l \cos \psi
$$

$$
x_{k+1} = x_k + \Delta x, \quad y_{k+1} = y_k + \Delta y
$$

During visual approach in this workflow, `u_l = 0` by design (no lateral drift during final alignment/approach).

## Visual Approach Math (Rotate First, Then Forward)

`go_to_tower()` runs a closed loop over camera frames.

### Centering step

Horizontal pixel error:

$$
e_x = c_x - c_{x,frame}
$$

If `|e_x|` is above tolerance:

$$
\omega_z = \mathrm{clamp}(-k_{yaw} e_x, -\omega_{max}, \omega_{max}), \quad v_x = 0
$$

The implementation also requires stable centering for multiple consecutive frames (`center_stable >= 2`) before forward motion is enabled. If centering is lost while moving in, forward motion is immediately set back to zero and the controller returns to yaw-only realignment.

### Forward step

Top-edge error:

$$
e_t = y_{top,target} - y_{top,current}
$$

When centered:

$$
v_x = \mathrm{clamp}(k_f e_t, -v_{max}, v_{max}), \quad \omega_z = 0
$$

So the robot first achieves angular alignment, then advances straight to the tuned top-y reach condition.

## Point-to-Point Solver Math

`move_to_pose()` moves from `(x, y, \psi)` to `(x_t, y_t, \psi_t)`:

1. Compute delta and distance:

$$
\Delta x = x_t - x, \quad \Delta y = y_t - y
$$

$$
d = \sqrt{\Delta x^2 + \Delta y^2}
$$

2. Compute travel heading:

$$
\psi_{travel} = \mathrm{atan2}(\Delta y, \Delta x)
$$

3. Execute motion sequence:

- turn to `\psi_{travel}`,
- drive forward distance `d`,
- turn to final slot yaw `\psi_t`.

This is why slot storage includes yaw: placement returns to both correct location and orientation.

## Tower 2 Exclusion During Reacquisition

To avoid mistaking tower 2 for tower 1, the script predicts tower 2 image x from current tracked pose.

1. Bearing to tower 2 world position:

$$
\theta_w = \mathrm{atan2}(y_{t2} - y, x_{t2} - x)
$$

2. Relative bearing in robot frame:

$$
\theta_r = \mathrm{wrapTo180}(\theta_w - \psi)
$$

3. Expected image x:

$$
c_{x,expected} = C_X + F_X \tan(\theta_r)
$$

Detections close to this predicted x (within configured tolerance) are filtered out.

## Important Assumptions

- Camera sees tower tops clearly enough for stable top-y stopping.
- Motion integration error is moderate (limited slip/drift).
- The robot starts each major phase from a known tracked pose (often home).
- Tower identities are initialized from left/right ordering and reacquisition exclusion is tuned well.

## Most Important Tuning Parameters

- `--target-top-y-ratio`: final reach distance proxy (manual tuning required).
- `--k-yaw`, `--max-yaw-dps`: centering responsiveness.
- `--k-forward`, `--max-v`: straight-in approach responsiveness.
- `--turn-speed-dps`: geometric turn execution.
- `--tower2-exclusion-cx-tol-px`: tower2 rejection aggressiveness in reacquire.
- `--scan-side-m`, `--scan-forward-m`, `--scan-rows`: search coverage.
- `--temp-forward-m`: temporary placement offset after 90-degree left turn.

## Practical Test Order

1. Verify detection stability with `--show`.
2. Tune center-then-forward behavior (`k_yaw`, `k_forward`, tolerances).
3. Tune `target-top-y-ratio` for reliable arm reach.
4. Validate pose tracking by running home -> tower -> home loops.
5. Validate temporary drop and return-home accuracy.
6. Validate tower2 exclusion in reacquire by moving tower1 during run.
7. Run full end-to-end swap.
