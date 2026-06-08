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

    # Pin an explicit sqlite backend: the default file store is in maintenance mode on mlflow>=3 and raises on setup
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_mlflow")
    SETTINGS["mlflow"] = True
    try:
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()
        SETTINGS["mlflow"] = False


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active(tmp_path, monkeypatch):
    """Ensure MLflow run status honors the MLFLOW_KEEP_RUN_ACTIVE environment variable across all states."""
    import mlflow

    # Pin an explicit sqlite backend: the default file store is in maintenance mode on mlflow>=3 and raises on setup
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "keep_run_active")
    monkeypatch.setenv("MLFLOW_RUN", "Test Run")
    SETTINGS["mlflow"] = True
    try:
        # MLFLOW_KEEP_RUN_ACTIVE=True keeps the run alive after training
        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "True")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        active = mlflow.active_run()
        assert active is not None and active.info.status == "RUNNING", (
            "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"
        )
        run_id = active.info.run_id

        # MLFLOW_KEEP_RUN_ACTIVE=False ends the still-active run after training
        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "False")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.get_run(run_id).info.status == "FINISHED", (
            "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"
        )

        # MLFLOW_KEEP_RUN_ACTIVE unset ends the run by default
        monkeypatch.delenv("MLFLOW_KEEP_RUN_ACTIVE", raising=False)
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, (
            "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is unset"
        )
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()
        SETTINGS["mlflow"] = False


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
