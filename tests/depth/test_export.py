from ultralytics import YOLO


def test_depth_onnx_export():
    m = YOLO("ultralytics/cfg/models/26/yolo26-depth.yaml")
    f = m.export(format="onnx", imgsz=128, nms=True)  # nms=True must be forced off, not crash
    assert str(f).endswith(".onnx")
    import onnx

    model = onnx.load(f)
    out_names = [o.name for o in model.graph.output]
    assert "depth" in out_names


def test_depth_onnx_export_dynamic(tmp_path):
    from ultralytics import YOLO

    m = YOLO("ultralytics/cfg/models/26/yolo26-depth.yaml")
    f = m.export(format="onnx", imgsz=128, dynamic=True)
    import onnx

    model = onnx.load(f)
    out_names = [o.name for o in model.graph.output]
    assert out_names == ["depth"]
    # the depth output should have dynamic dims on batch/height/width
    out = model.graph.output[0]
    dims = out.type.tensor_type.shape.dim
    assert any(d.dim_param for d in dims)  # at least one symbolic (dynamic) dim


def test_depth_imx_blocked():
    import pytest

    from ultralytics import YOLO

    m = YOLO("ultralytics/cfg/models/26/yolo26-depth.yaml")
    with pytest.raises(Exception):  # ValueError: IMX not supported for depth
        m.export(format="imx", imgsz=128)
