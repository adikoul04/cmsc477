# Swap and Place 2 Guide

This document explains how the action-stack swap workflow works, what math it uses, how arm posture is handled, and what to run for calibration/tuning versus full execution.

## 1) What the script does

The script [swap_and_place_2.py](swap_and_place_2.py) swaps two detected towers:

1. Detect left and right towers from home.
2. Go to left tower (Tower 1), pick it up.
3. Stash Tower 1 to a side location, then return to Tower 1 original slot.
4. Return home by reversing the recorded path.
5. Go to right tower (Tower 2), pick it up.
6. Return home, replay Tower 1 route, place Tower 2 at Tower 1 original slot.
7. Return home, rescan and reacquire stashed Tower 1 (excluding the new Tower 2 column).
8. Go to stashed Tower 1, pick it up.
9. Return home, replay Tower 2 route, place Tower 1 at Tower 2 original slot.
10. Return home.

## 2) Why action-stack homing works

Every visual-servo chassis command is recorded as:

- forward speed $v_x$ (m/s)
- lateral speed $v_y$ (m/s)
- yaw rate $v_z$ (deg/s)
- duration $\Delta t$ (s)

A route is a list of these actions in time order. To return, the script executes the list in reverse with negated velocities.

If forward command $u_i = (v_{x,i}, v_{y,i}, v_{z,i}, \Delta t_i)$, reverse command is:

$$
\tilde{u}_i = (-v_{x,i}, -v_{y,i}, -v_{z,i}, \Delta t_i)
$$

and commands are applied from last to first.

This avoids needing explicit geometric pose estimation for homing.

## 3) Visual-servo math used during approach

In [swap_and_place_2.py](swap_and_place_2.py), the target tower is selected and controlled using:

- Image center error: $e_x = c_x - c_{x,frame}$
- Top-of-box error: $e_f = y_{target} - y_{top}$
- $y_{target} = r_{top} \cdot H$ where:
  - $r_{top}$ is --target-top-y-ratio
  - $H$ is frame height in pixels

Control law:

- If not centered yet:
  - $v_x = 0$
  - $v_z = clamp(-k_{yaw} e_x, -v_{z,max}, v_{z,max})$
- If centered stably:
  - $v_x = clamp(k_{forward} e_f, -v_{x,max}, v_{x,max})$
  - $v_z = 0$

Stop condition requires several consecutive stable frames where both errors are within tolerance.

## 4) Arm posture guarantee during movement

Travel-related functions now explicitly raise the arm to top before moving the chassis:

- Action stack unwind
- Route replay
- Route reverse
- Stash travel
- Return from stash
- Visual approach already raised arm at the start

Calibration script [calibrate_top_y_ratio.py](calibrate_top_y_ratio.py) also now raises the arm to top before collecting calibration frames.

## 5) Scripts and their role

- [bounding_box_capture.py](bounding_box_capture.py)
  - Quick live detection sanity check with your trained YOLO weights and robot feed.
- [calibrate_top_y_ratio.py](calibrate_top_y_ratio.py)
  - Measures top-of-box ratio at your desired pickup distance/orientation.
- [swap_and_place_2.py](swap_and_place_2.py)
  - Full two-tower swap routine.

## 6) Parameter tuning order

Use this order for stable tuning:

1. Detection confidence and class filtering
2. Top-y stop ratio
3. Servo gains and limits
4. Tolerances and timing
5. Stash offsets

### 6.1 Detection sanity

Run [bounding_box_capture.py](bounding_box_capture.py) to verify:

- Correct model path
- Correct class labels
- Reliable two-tower detections under your lighting

### 6.2 Top-y ratio calibration

Run [calibrate_top_y_ratio.py](calibrate_top_y_ratio.py) with the robot at desired pickup distance.
Use median (or slightly smaller) output as --target-top-y-ratio.

### 6.3 Parameter reference and behavior-based tuning

In [swap_and_place_2.py](swap_and_place_2.py), these are the most important approach-control parameters.

#### k_forward

What it does:

- Gain from forward image error to forward speed.
- Larger value means faster in/out motion when approaching target top y.

Behavior-based tuning:

- If robot approaches too slowly and takes too long to stop at pickup distance: increase slightly.
- If robot surges forward, overshoots, or bounces near the tower: decrease.
- Tune in small increments (about 10% at a time).

#### k_yaw

What it does:

- Gain from horizontal pixel error to yaw rate.
- Larger value means faster turning to center the tower in the image.

Behavior-based tuning:

- If robot takes too long to center: increase slightly.
- If robot oscillates left-right or jitters around center: decrease.
- Keep this balanced with max_yaw_dps so turns are responsive but not abrupt.

#### max_v

What it does:

- Maximum forward speed clamp during visual servo.
- Prevents very large forward commands when error is high.

Behavior-based tuning:

- If robot feels too slow even with good k_forward: increase modestly.
- If robot is aggressive near the tower or slips: decrease.
- Lowering max_v is often safer than heavily reducing k_forward.

#### max_yaw_dps

What it does:

- Maximum yaw-rate clamp in deg/s.
- Limits turn aggressiveness and protects against sudden spins.

Behavior-based tuning:

- If centering is sluggish despite adequate k_yaw: increase modestly.
- If heading snaps too hard or overshoots center often: decrease.
- Indoors on smooth floors, slightly lower values can improve stability.

#### center_tol_px

What it does:

- Horizontal alignment tolerance in pixels before tower is considered centered.
- Also gates forward motion: robot typically centers first, then advances.

Behavior-based tuning:

- If robot stalls rotating and rarely starts forward motion: increase tolerance.
- If robot starts driving while still visibly misaligned: decrease tolerance.
- Keep this tight enough for clean pickup alignment, but not so tight that noise blocks progress.

#### top_y_tol_px

What it does:

- Tolerance around target top y used in stable stop condition.
- Smaller value means stricter distance stop; larger value means easier stop acceptance.

Behavior-based tuning:

- If robot never settles and keeps inching/oscillating: increase slightly.
- If stop distance is too inconsistent across trials: decrease slightly.
- Tune together with target_top_y_ratio.

#### target_top_y_ratio

What it does:

- Desired top-of-box image y position as a fraction of image height.
- Directly sets final standoff distance from tower.

Behavior-based tuning:

- If robot stops too far from tower: increase ratio.
- If robot stops too close or bumps tower: decrease ratio.
- Calibrate with [calibrate_top_y_ratio.py](calibrate_top_y_ratio.py), then use median as a baseline and adjust by about 0.02 to 0.05 as needed.

#### Practical tuning loop

1. Fix target_top_y_ratio first.
2. Tune k_yaw and max_yaw_dps for smooth centering.
3. Tune k_forward and max_v for controlled approach.
4. Tighten center_tol_px and top_y_tol_px only after motion is stable.

### 6.4 Stash offset tuning

- --stash-yaw-deg: angle to face stash direction.
- --stash-forward-m: distance from Tower 1 original slot to stash drop point.

## 7) Run commands (Windows examples)

From Project 2 directory:

### 7.1 Calibrate top-y ratio

python calibrate_top_y_ratio.py --conn-type sta --sn 3JKCH8800100RC --robot-ip 192.168.50.117 --model-path C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt --conf 0.45 --show

Notes:

- Press q to stop.
- Use reported median as initial --target-top-y-ratio.

### 7.2 Run swap and place

python swap_and_place_2.py --conn-type sta --sn 3JKCH8800100RC --robot-ip 192.168.50.117 --model-path C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt --detect-conf 0.45 --target-top-y-ratio 0.72 --align-center-tol-px 24 --align-top-tol-px 18 --k-forward 0.0028 --k-yaw 0.12 --max-v 0.16 --max-yaw-dps 45 --servo-step-s 0.12 --stash-yaw-deg 90 --stash-forward-m 0.35 --show

## 8) Practical notes

- Keep lighting and camera exposure stable between calibration and execution.
- Keep tower spacing sufficiently large so exclusion logic can separate them by image x-position.
- The action-stack method is robust and simple, but physical slip can still introduce some placement drift over long sequences.
