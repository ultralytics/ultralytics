from collections import defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm  
os.environ["LD_LIBRARY_PATH"] = "/home/anting555/local_libs/usr/lib/x86_64-linux-gnu"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/anting555/yolo/yolo_env/lib/python3.8/site-packages/qt5_applications/Qt/plugins/platforms"

model = YOLO("yolo11n.pt")
video_path = "/home/anting555/yolo/ultralytics/multimodal_senator/Bob_Casey_C_eRKK1vFyz.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_10frame_tqdm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

track_history = defaultdict(lambda: [])

for frame_id in tqdm(range(total_frames), desc="processing", unit="frame"):
    success, frame = cap.read()
    if not success:
        break

    if frame_id % 10 == 0:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        track_ids = track_ids.int().cpu().tolist() if track_ids is not None else []

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    else:
        annotated_frame = frame

    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
