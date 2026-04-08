# Swap-and-Place Notes

This note summarizes what [swap_and_place.py](swap_and_place.py) does and what to tune next before using it on the RoboMaster.

## What the script does

The script implements a full state machine for the Project 2 swap task:

1. Detect two LEGO towers with YOLO from the RoboMaster camera.
2. Estimate each tower's approximate forward distance and lateral offset from the camera using the bounding-box height and horizontal position.
3. Drive the chassis to tower 1, align visually, and use the existing arm/gripper helpers from [tower_utils.py](tower_utils.py) to pick it up.
4. Move tower 1 to a temporary location and release it.
5. Return home, move to tower 2, align again, and place tower 2 at tower 1's original slot.
6. Treat tower 2's placed location as an exclusion zone, then search for tower 1 if it was moved by a human, reacquire it with a sweep motion, and pick it back up.
7. Place tower 1 at tower 2's original slot.

## Why the file is structured this way

The script keeps the robot behavior separated into small helpers so the logic is easier to tune:

- `Detection` stores raw YOLO bounding-box data.
- `RelativeTarget` stores the estimated forward and lateral position of a tower.
- `PoseTracker` keeps track of how far the chassis has moved from the home pose so the robot can return before the next tower.
- `detect_stable_two_towers()` waits for a consistent pair of detections before starting the task.
- `align_and_approach_target()` uses a simple visual servo loop to center the tower and match the desired apparent size before pickup.
- `reacquire_any_tower_with_sweep()` searches translationally for a tower again if it was moved during the swap, while rejecting detections that fall inside tower 2's exclusion radius.

## Existing robot methods reused

The script reuses the code you already had for the RoboMaster arm and gripper:

- `connect_robot()`
- `pick_up_tower()`
- `place_down_tower()`

It also follows the chassis motion style used in [Project 1/project1_nav.py](../Project%201/project1_nav.py), especially the use of `chassis.move(...).wait_for_completed()` for coarse moves and `chassis.drive_speed(...)` for short servo updates.

## Important assumptions

- The YOLO model has already been fine-tuned on your tower dataset.
- The tower class can be filtered with `--target-class` if needed, but the script can also run without that filter.
- Tower size is estimated from bounding-box height, so the camera view should stay reasonably similar to the training view.
- The two towers can be at different distances from the robot.
- Tower 2's placed position is treated as a fixed reference during the final reacquisition step so the robot does not accidentally confuse tower 2 with the moved tower 1.
- The robot may need left/right sign correction through `--lateral-sign` depending on your chassis behavior.

## Suggested next steps

1. Run the script with `--show` so you can watch the detections and alignment behavior live.
2. Verify the lateral correction direction first. If the tower moves the wrong way when it should slide left or right, flip `--lateral-sign` between `1` and `-1`.
3. Tune `--align-desired-h-px`, `--align-center-tol-px`, and `--align-height-tol-px` so the arm reaches a good pickup distance.
4. Tune `--temp-back-m` and `--temp-side-m` so the temporary placement is reachable and does not interfere with the other tower.
5. Tune `--scan-side-m`, `--scan-forward-m`, and `--scan-rows` so the reacquisition sweep covers the area where tower 1 might be moved.
6. Tune `--tower2-exclusion-radius-m` so tower 2 is ignored reliably without excluding a valid tower 1 that was moved nearby.
7. If the robot overshoots or stops too early, adjust `--k-forward`, `--k-lateral`, and `--max-v`.
8. Once the motion is stable, record a full demo video showing the human moving the towers while the robot is executing.

## Quick run example

```bash
python3 "Project 2/swap_and_place.py" \
  --model-path "Project 2/cmsc477_yolo/runs/detect/train/weights/best.pt" \
  --show
```
