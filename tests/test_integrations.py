# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import subprocess
import time
from pathlib import Path

import pytest

from tests import SOURCE
from ultralytics import YOLO, download
from ultralytics.utils import ASSETS_URL, DATASETS_DIR, SETTINGS
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


@pytest.mark.skipif(not check_requirements("faster-coco-eval", install=False), reason="faster-coco-eval not installed")
def test_faster_coco_eval():
    """Validate YOLO model predictions on COCO dataset using faster-coco-eval."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    args = {"model": "yolo26n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    _ = validator.eval_json(validator.stats)

    args = {"model": "yolo26n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    _ = validator.eval_json(validator.stats)

    args = {"model": "yolo26n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    _ = validator.eval_json(validator.stats)


@pytest.mark.skipif(not check_requirements("faster-coco-eval", install=False), reason="faster-coco-eval not installed")
@pytest.mark.parametrize(
    "iou_types, suffix, expected",
    [
        (["bbox", "segm"], ["Box", "Mask"], {"metrics/mAP_small(M)": 0.0, "metrics/mAP_large(M)": 1.0}),
        (["bbox", "keypoints"], ["Box", "Pose"], {"metrics/mAP_large(P)": 1.0}),
    ],
)
def test_coco_evaluate_size_metrics_per_iou_type(iou_types, suffix, expected):
    """Keep the box size mAP when a second IoU type is evaluated and report that type under its own suffix."""
    from ultralytics.models.yolo.detect import DetectionValidator

    def square(x0, y0, x1, y1):
        return [x0, y0, x1, y0, x1, y1, x0, y1]

    def keypoints(x0, y0, step):
        return [v for i in range(17) for v in (x0 + step * i, y0 + step * i, 2)]

    # One small object (20x20 < 32**2) and one large object (120x120 > 96**2) in a 200x200 image
    objects = [
        {
            "bbox": [10, 10, 20, 20],
            "area": 400.0,
            "segmentation": [square(10, 10, 30, 30)],
            "keypoints": keypoints(11, 11, 1),
        },
        {
            "bbox": [50, 50, 120, 120],
            "area": 14400.0,
            "segmentation": [square(50, 50, 170, 170)],
            "keypoints": keypoints(60, 60, 6),
        },
    ]
    gt = {
        "images": [{"id": 1, "width": 200, "height": 200}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": 1, "iscrowd": 0, "num_keypoints": 17, **obj}
            for i, obj in enumerate(objects, 1)
        ],
    }
    # Exact boxes and keypoints, but the small mask covers 40% of its object and misses every IoU threshold
    preds = [{"image_id": 1, "category_id": 1, "score": 0.9 - 0.1 * i, **obj} for i, obj in enumerate(objects)]
    preds[0]["segmentation"] = [square(10, 10, 18, 30)]

    validator = DetectionValidator(args={"save_json": True})
    validator.jdict, validator.gdict = preds, gt
    stats = validator.coco_evaluate({}, preds, gt, iou_types, suffix)

    assert stats["metrics/mAP_small(B)"] == 1.0
    assert stats["metrics/mAP_large(B)"] == 1.0
    assert "metrics/mAP_small(P)" not in stats  # COCO keypoint evaluation has no small area range
    for k, v in expected.items():
        assert stats[k] == v
