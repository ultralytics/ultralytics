from ultralytics import YOLO
model = YOLO("/home/ralampay/workspace/ultralytics/custom_yolo12.yaml")
print(model.model)
