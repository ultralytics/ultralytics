# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import copy
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from tests import SOURCE
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_requirements


@pytest.mark.slow
def test_tensorboard():
    """Test training with TensorBoard logging enabled."""
    SETTINGS["tensorboard"] = True
    YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    SETTINGS["tensorboard"] = False


@pytest.mark.skipif(not check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_ray_tune():
    """Tune YOLO model using Ray for hyperparameter optimization."""
    YOLO("yolo26n-cls.yaml").tune(
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow(tmp_path, monkeypatch):
    """Test training with MLflow tracking enabled."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_mlflow")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active(tmp_path, monkeypatch):
    """Ensure MLFLOW_KEEP_RUN_ACTIVE controls whether new MLflow runs remain active."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "keep_run_active")
    monkeypatch.setenv("MLFLOW_RUN", "Test Run")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "True")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        active = mlflow.active_run()
        assert active is not None and active.info.status == "RUNNING", (
            "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"
        )
        mlflow.end_run()

        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "False")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"

        monkeypatch.delenv("MLFLOW_KEEP_RUN_ACTIVE", raising=False)
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is unset"
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] not installed")
def test_triton(tmp_path, isolated_model):
    """Test NVIDIA Triton Server functionalities with YOLO model."""
    check_requirements("tritonclient[all]")
    from tritonclient.http import InferenceServerClient

    # Create variables
    model_name = "yolo"
    triton_repo = tmp_path / "triton_repo"  # Triton repo path
    triton_model = triton_repo / model_name  # Triton model path

    # Export model to ONNX
    f = YOLO(isolated_model).export(format="onnx", dynamic=True)

    # Prepare Triton repo
    (triton_model / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model / "1" / "model.onnx")
    (triton_model / "config.pbtxt").touch()

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB

    # Pull the image
    subprocess.call(f"docker pull {tag}", shell=True)

    # Run the Triton server and capture the container ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_repo}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    # Wait for the Triton server to start
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    # Wait until model is ready
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)

    # Check Triton inference
    YOLO(f"http://localhost:8000/{model_name}", "detect")(SOURCE)  # exported model inference

    # Kill and remove the container at the end of the test
    subprocess.call(f"docker kill {container_id}", shell=True)


@pytest.mark.skipif(
    not check_requirements("ultrafast-pycocotools", install=False), reason="ultrafast-pycocotools not installed"
)
@pytest.mark.parametrize(
    "task,suffix,iou_type", [("detect", "", "bbox"), ("segment", "-seg", "segm"), ("pose", "-pose", "keypoints")]
)
def test_ultrafast_pycocotools(task, suffix, iou_type, caplog, tmp_path):
    """Exercise real YOLO prediction serialization and COCO evaluation against synthetic ground truth."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    cls = {"detect": DetectionValidator, "segment": SegmentationValidator, "pose": PoseValidator}[task]
    validator = cls(
        args={"model": f"yolo26n{suffix}.pt", "data": f"coco8{suffix}.yaml", "save_json": True, "imgsz": 64},
        save_dir=tmp_path,
    )
    validator()
    assert validator.jdict
    annotations = []
    for prediction in validator.jdict:
        annotation = dict(prediction, id=len(annotations) + 1, iscrowd=0)
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        if iou_type == "keypoints":
            annotation["num_keypoints"] = len(annotation["keypoints"]) // 3
        annotations.append(annotation)
    images = {p["image_id"]: {"id": p["image_id"]} for p in validator.jdict}
    if task == "segment":
        for prediction in validator.jdict:
            height, width = prediction["segmentation"]["size"]
            images[prediction["image_id"]].update(height=height, width=width)
    data = {
        "images": list(images.values()),
        "categories": [{"id": i} for i in sorted({p["category_id"] for p in validator.jdict})],
        "annotations": annotations,
    }
    validator.gdict, validator.is_coco, validator._coco_api = data, True, None
    types = ["bbox"] if task == "detect" else ["bbox", iou_type]
    suffixes = ["Box"] if task == "detect" else ["Box", "Mask" if task == "segment" else "Pose"]
    stats = validator.coco_evaluate({}, validator.jdict, data, types, suffix=suffixes)
    for name in suffixes:
        assert 0 <= stats[f"metrics/mAP50-95({name[0]})"] <= 1
    assert "unable to run" not in caplog.text


@pytest.mark.parametrize(
    "iou_type,lvis", [("bbox", False), ("segm", False), ("keypoints", False), ("bbox", True), ("segm", True)]
)
def test_coco_evaluator_parity(iou_type, lvis, tmp_path):
    """Compare complete arrays and validator metrics against the previous COCO backend."""
    faster = pytest.importorskip("faster_coco_eval")  # Reference only; not a runtime dependency.
    from ultrafast_pycocotools import COCO, COCOeval

    from ultralytics.models.yolo.detect import DetectionValidator

    data = {
        "images": [
            {"id": 1, "height": 128, "width": 128, "neg_category_ids": [1, 2, 3], "not_exhaustive_category_ids": [2]}
        ],
        "categories": [{"id": i, "name": str(i), "frequency": f} for i, f in enumerate("rcf", 1)],
        "annotations": [
            {
                "id": i,
                "image_id": 1,
                "category_id": i,
                "bbox": [10, 10, 40, 40],
                "area": 1600,
                "iscrowd": int(i == 3),
                "ignore": int(i == 2),
                "num_keypoints": 17,
                "keypoints": [20, 20, 2] * 17,
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            }
            for i in range(1, 4)
        ],
    }
    predictions = [dict(a, score=0.8) for a in data["annotations"]]
    predictions = [
        dict(
            predictions[0],
            bbox=[80, 80, 20, 20],
            keypoints=[90, 90, 2] * 17,
            segmentation=[[80, 80, 100, 80, 100, 100, 80, 100]],
            score=0.9,
        )
    ] * (301 if lvis else 2) + predictions
    evaluators = []
    for coco, evaluator in [(faster.COCO, faster.COCOeval_faster), (COCO, COCOeval)]:
        gt = coco(copy.deepcopy(data))
        kwargs = {"lvis_protocol": "coco"} if evaluator is COCOeval else {}
        ev = evaluator(gt, gt.loadRes(copy.deepcopy(predictions)), iou_type, lvis_style=lvis, **kwargs)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        evaluators.append(ev)
    reference, actual = evaluators
    for key in ("precision", "recall", "scores"):
        np.testing.assert_allclose(actual.eval[key], reference.eval[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.stats, reference.stats, rtol=0, atol=1e-12)

    validator = DetectionValidator(args={"save_json": True}, save_dir=tmp_path)
    validator.is_lvis, validator.is_coco = lvis, not lvis
    validator.gdict, validator.jdict, validator.training = data, predictions, False
    stats = validator.coco_evaluate({}, predictions, data, iou_type)
    for name, key in [("mAP50", "AP_50"), ("mAP50-95", "AP_all")]:
        assert stats[f"metrics/{name}(B)"] == pytest.approx(reference.stats_as_dict[key], rel=0, abs=1e-12)
    if lvis:
        for key in ("APr", "APc", "APf"):
            assert stats[f"metrics/{key}(B)"] == pytest.approx(reference.stats_as_dict[key], rel=0, abs=1e-12)
    expected = reference.stats_as_dict["AP_all"]
    if not lvis:
        expected = 0.9 * expected + 0.1 * reference.stats_as_dict["AP_50"]
    assert stats["fitness"] == pytest.approx(expected, rel=0, abs=1e-12)
