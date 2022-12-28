import torch

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Segment


def export_onnx(model, file):
    # YOLOv5 ONNX export
    import onnx
    im = torch.zeros(1, 3, 640, 640)
    model.eval()
    model(im, profile=True)
    for k, m in model.named_modules():
        if isinstance(m, (Detect, Segment)):
            m.export = True

    torch.onnx.export(
        model,
        im,
        file,
        verbose=False,
        opset_version=12,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'])

    # Checks
    model_onnx = onnx.load(file)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, file)


if __name__ == "__main__":
    model = YOLO()
    print("yolov8n")
    model.new("yolov8n.yaml")
    print("yolov8n-seg")
    model.new("yolov8n-seg.yaml")
    print("yolov8s")
    model.new("yolov8s.yaml")
    # export_onnx(model.model, "yolov8s.onnx")
    print("yolov8s-seg")
    model.new("yolov8s-seg.yaml")
    # export_onnx(model.model, "yolov8s-seg.onnx")
    print("yolov8m")
    model.new("yolov8m.yaml")
    print("yolov8m-seg")
    model.new("yolov8m-seg.yaml")
    print("yolov8l")
    model.new("yolov8l.yaml")
    print("yolov8l-seg")
    model.new("yolov8l-seg.yaml")
    print("yolov8x")
    model.new("yolov8x.yaml")
    print("yolov8x-seg")
    model.new("yolov8x-seg.yaml")

    # n vs n-seg: 8.9GFLOPs vs 12.8GFLOPs, 3.16M vs 3.6M. ch[0] // 4 (11.9GFLOPs, 3.39M)
    # s vs s-seg: 28.8GFLOPs vs 44.4GFLOPs, 11.1M vs 12.9M. ch[0] // 4 (39.5GFLOPs, 11.7M)
    # m vs m-seg: 79.3GFLOPs vs 113.8GFLOPs, 25.9M vs 29.5M. ch[0] // 4 (103.GFLOPs, 27.1M)
    # l vs l-seg: 165.7GFLOPs vs 226.3GFLOPs, 43.7M vs 49.6M. ch[0] // 4 (207GFLOPs, 45.7M)
    # x vs x-seg: 258.5GFLOPs vs 353.0GFLOPs, 68.3M vs 77.5M. ch[0] // 4 (324GFLOPs, 71.4M)
