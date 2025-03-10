# 此文件由程序自动生成.
# 请勿手动修改.
from ultralytics import YOLO

w = 640
h = 640
model_path = "./export_results/train_001_猫狗模型训练.pt"
model = YOLO(model_path)
model.export(format="paddle", imgsz=[w, h], opset=12, device="0")
