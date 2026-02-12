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
)

# model = torch.jit.load("best.torchscript")
# input_names = 'images'
# output_names = 'output0'
# torch_input = torch.randn(1, 3, 640, 640)
# torch.onnx.export(model, torch_input, 'best.onnx')

# model = onnx.load("best.onnx")
# model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
# onnx.save(model_fp16, "king_game_fp16.onnx")
