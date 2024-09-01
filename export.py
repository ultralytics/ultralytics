# Author: Zenghui Tang
# Date: 2024/4/8
# Time: 10:36
from ultralytics import YOLO

# 加载模型
model = YOLO(r"C:\Users\i\Desktop\fsdownload/best1.pt")  # 加载自定义训练的模型

# 导出模型
model.export(format="onnx", imgsz=(544, 960), opset=12)


if __name__ == "__main__":
    pass
