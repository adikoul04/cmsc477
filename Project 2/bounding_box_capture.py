from ultralytics import YOLO
import cv2
import time
from robomaster import robot
from robomaster import camera

DEFAULT_ROBOT_IP = "192.168.50.117"
DEFAULT_ROBOT_SN = "3JKCH8800100RC"

print('model')
model = YOLO(r"C:\Users\dutta\OneDrive\Documents\College\Senior (Spring)\CMSC477\cmsc477\runs\detect\train5\weights\best.pt")
# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap", sn=DEFAULT_ROBOT_SN, ip=DEFAULT_ROBOT_IP)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

while True:
    # ret, frame = vid.read()
    frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    if frame is not None:
        start = time.time()
        if model.predictor:
            model.predictor.args.verbose = False
            result = model.predict(source=frame, show=False)[0]
            # DIY visualization is much faster than show=True for some reason
            boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            cv2.rectangle(frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        color=(0, 0, 255), thickness=2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # print(results)
        end = time.time()
        print(1.0 / (end-start))
