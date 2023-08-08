from ultralytics import YOLO

model = YOLO('./../models/yolov8n.pt')
metrics = model.val(data='./ultralytics/cfg/datasets/coco.yaml', save_json=True ,verbose=False, plots=False, device=[6,7])


x=0