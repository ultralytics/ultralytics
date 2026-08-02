"""Release-facing export and runtime checks for the public Detect3D interface."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect3d.val import Detection3DMetrics


def test_detect3d_onnx_round_trip_keeps_task_and_geometry(tmp_path, monkeypatch):
    """A static ONNX export must reload as Detect3D and retain its eight geometry outputs."""
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    monkeypatch.chdir(tmp_path)

    config = Path(__file__).parents[1] / "ultralytics/cfg/models/26/yolo26n-3d.yaml"
    model = YOLO(config)
    exported = Path(model.export(format="onnx", imgsz=320, simplify=False, device="cpu")).resolve()
    assert exported == tmp_path / "yolo26n-3d.onnx"
    assert YOLO(exported).task == "detect3d"

    session = onnxruntime.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: np.zeros((1, 3, 320, 320), dtype=np.float32)})[0]
    assert output.shape == (1, 300, 14)
    assert np.isfinite(output).all()


def test_detect3d_results_exports_geometry_fields():
    """Result summaries and JSON exports must not silently discard decoded 3D geometry."""
    result = Results(
        np.zeros((100, 200, 3), dtype=np.uint8),
        path="frame.png",
        names={0: "Car"},
        boxes=torch.tensor([[10.0, 20.0, 50.0, 80.0, 0.9, 0.0]]),
        d3_params=torch.tensor([[100.0, 50.0, 20.0, 0.0, 1.0, 1.5, 1.8, 4.0]]),
    )

    summary = result.summary(normalize=True)[0]
    assert summary["box3d"] == {
        "center_x": 0.5,
        "center_y": 0.5,
        "depth": 20.0,
        "sin_alpha": 0.0,
        "cos_alpha": 1.0,
        "height": 1.5,
        "width": 1.8,
        "length": 4.0,
    }
    assert json.loads(result.to_json())[0]["box3d"]["depth"] == 20.0
    assert "'depth': 20.0" in result.to_csv()


def test_detect3d_metric_summary_includes_3d_columns():
    """Metric DataFrame/CSV exports must expose task-primary 3D metrics alongside box metrics."""
    metrics = Detection3DMetrics({0: "Car"})
    metrics.box.p = np.array([0.8])
    metrics.box.r = np.array([0.7])
    metrics.box.f1 = np.array([0.7467])
    metrics.box.all_ap = np.full((1, 10), 0.75)
    metrics.box.ap_class_index = np.array([0])
    metrics.box.nc = 1
    metrics.d3.p = np.array([0.5])
    metrics.d3.r = np.array([0.4])
    metrics.d3.f1 = np.array([0.4444])
    metrics.d3.all_ap = np.array([[0.3, *([0.2] * 9)]])
    metrics.d3.ap_class_index = np.array([0])
    metrics.d3.nc = 1
    metrics.nt_per_class = np.array([10])
    metrics.nt_per_image = np.array([3])
    metrics.nt_per_class_d3 = np.array([8])

    summary = metrics.summary(decimals=3)[0]
    assert summary["3D-Instances"] == 8
    assert summary["3D-P"] == 0.5
    assert summary["3D-mAP50"] == 0.3
