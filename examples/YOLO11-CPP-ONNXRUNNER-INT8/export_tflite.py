import os.path

from ultralytics import YOLO

model = YOLO("best.pt")  # load a custom trained model
yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "YOLODataset", "dataset.yaml"))
model.export(
    data=yaml_path,  # path to dataset YAML
    format="tflite",  # 导出格式为 torchscript
    imgsz=(640, 640),  # 设置输入图像的尺寸
    half=False,  # 启用 FP16 量化
    int8=True,  # 不启用 INT8 量化
    batch=1,  # 指定批处理大小
    nms=False,
    fraction=1,
    device="cpu"        # 指定导出设备为CPU或GPU,对应参数为"cpu","0"
)
