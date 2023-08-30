from ultralytics import YOLO

image_size = 608

expirment_name = 'yolov8n-baseline-608-SGD-1-2023-08-30_11-43-46'
model_path = f'/home/zh/pythonhub/military-target-detection/military/{expirment_name}/weights/best.pt'

model = YOLO(model_path)

metrics = model.val(imgsz=image_size)
