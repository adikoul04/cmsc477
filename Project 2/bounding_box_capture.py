from ultralytics import YOLO
import cv2
import time
import queue
from robomaster import robot
from robomaster import camera

DEFAULT_ROBOT_IP = "192.168.50.117"
DEFAULT_ROBOT_SN = "3JKCH8800100RC"

print('model')
model = YOLO(r"C:\Users\dutta\Documents\cmsc477\runs\detect\train5\weights\best.pt")
# Use vid instead of ep_camera to use your laptop's webcam
# vid = cv2.VideoCapture(0)
ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta", sn=DEFAULT_ROBOT_SN)
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

while True:
    # ret, frame = vid.read()
    try:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
    except queue.Empty:
        # no frame available yet; retry
        continue
    except Exception as e:
        print(f"camera read error: {e}")
        continue
    if frame is not None:
        start = time.time()
        boxes = []
        print("predicting")
        # Do not access `model.predictor` before calling `model.predict()`; the predictor
        # is created internally on the first call. Pass `verbose=False` via kwargs instead.
        result = model.predict(source=frame, show=False, verbose=False)[0]
        # DIY visualization is much faster than show=True for some reason
        boxes = result.boxes
        if len(boxes) == 0:
            print("No detections")
        else:
            print(f"Detections: {len(boxes)}")
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            cv2.rectangle(frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        color=(0, 0, 255), thickness=2)
            cls = None
            conf = None
            try:
                if hasattr(box, 'cls'):
                    cls = int(box.cls.cpu().numpy())
            except Exception:
                cls = None
            try:
                if hasattr(box, 'conf'):
                    conf = float(box.conf.cpu().numpy())
            except Exception:
                conf = None
            try:
                name = model.names[cls]
            except Exception:
                name = str(cls)
            label = f"{name} {conf:.2f}" if conf is not None else f"{name}"
            cv2.putText(frame, label, (int(xyxy[0]), max(15, int(xyxy[1]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f" - class={name} id={cls} conf={conf}")
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # print(results)
        end = time.time()