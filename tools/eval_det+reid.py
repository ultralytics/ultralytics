from ultralytics import YOLOManitou_MultiCam

project = "runs/manitou_remap"
data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
# weights = "/root/workspace/ultralytics/tools/runs/manitou_remap/train/weights/best.pt"
weights = "/root/workspace/ultralytics/tools/runs/manitou_remap/train3/weights/best.pt"

device = [0]
batch_size = 8
imgsz = (1552, 1936)  # (height, width)
model = YOLOManitou_MultiCam(model="yolo11s.yaml").load(weights)
metrics = model.val(data=data_cfg, imgsz=imgsz, batch=batch_size, device=device)
