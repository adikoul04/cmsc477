import robomaster
from robomaster import robot
import time

def sub_data_handler(sub_info):
    pos_x, pos_y = sub_info

    # Fix the integer overflow in pos_y
    if pos_y > 2**31 - 1:
        pos_y = pos_y - 2**32

    # You can use these values to confirm the robot arm is where it is supposed to be
    # It is also usable for determine the right setpoints to send to "moveto" commands
    print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta", sn="3JKCH880010129")
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper

    # Start printing the gripper position
    ep_arm.sub_position(freq=5, callback=sub_data_handler)

    # Open the gripper
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()

    # Move the arm to the "retracted" position
    ep_robot.robotic_arm.moveto(x=100, y=30).wait_for_completed()

    # Move the arm forward and down in order to pickup an object
    # (we do this in two moves to avoid the "keep out zone" where the robot may hit itself)
    ep_robot.robotic_arm.moveto(x=180, y=30).wait_for_completed()
    ep_robot.robotic_arm.moveto(x=180, y=-50).wait_for_completed()

    # Close the gripper on the object
    ep_gripper.close(power=50)
    time.sleep(1)
    ep_gripper.pause()

    # Lift the object in the gripper
    ep_robot.robotic_arm.moveto(x=180, y=100).wait_for_completed()

    # Relative move commands work inconsistently but if you want to use them they are:
    # ep_arm.move(x=50).wait_for_completed()
    # ep_arm.move(x=-50).wait_for_completed()
    # ep_arm.move(y=50).wait_for_completed()
    # ep_arm.move(y=-50).wait_for_completed()

    ep_robot.close()
