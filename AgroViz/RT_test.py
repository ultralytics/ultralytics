from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolov8m.pt")

# Real time detection on webcam
results = model.predict(source="0", show=True)  # save predictions as labels

print(results)
