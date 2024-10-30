from ultralytics import YOLO


def test_ncnn():
    # export ncnn and do inference
    model = YOLO("yolo11n.pt")
    filename = model.export(imgsz=160, format="ncnn", half=False, int8=False, device="cpu", verbose=False)
    exported_model = YOLO(filename, task=model.task)
    results = exported_model.val(
        data="coco8.yaml",
        batch=1,
        imgsz=160,
        plots=False,
        device="cpu",
        half=False,
        int8=False,
        verbose=False,
    )
    # then export onnx
    model = YOLO("yolo11n.pt")
    filename = model.export(imgsz=160, format="onnx", half=False, int8=False, device="cpu", verbose=False)
