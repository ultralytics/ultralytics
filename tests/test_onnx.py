from ultralytics import YOLO


def test_onnx():
    model = YOLO("yolo11n.pt")
    model.export(imgsz=160, format="onnx", half=False, int8=False, device="cpu", verbose=False)
