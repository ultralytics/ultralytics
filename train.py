# Author: Zenghui Tang
# Date: 2024/3/26
# Time: 13:50

from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # 从YAML构建并转移权重

# 训练模型
results = model.train(task="detect", data="coco128.yaml", epochs=10, imgsz=640, batch=1, device=0, workers=0)


if __name__ == "__main__":
    pass
