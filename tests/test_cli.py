# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest
from PIL import Image

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODELS, TASK_MODEL_DATA
from ultralytics.utils import ARM64, ASSETS, DATASETS_DIR, IS_RASPBERRYPI, LINUX, WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_VERSION


def run(cmd: str) -> None:
    """Execute a shell command using subprocess."""
    subprocess.run(cmd.split(), check=True)


def test_special_modes() -> None:
    """Test various special command-line modes for YOLO functionality."""
    run("yolo help")
    run("yolo checks")
    run("yolo version")
    run("yolo settings reset")
    run(f"yolo settings weights_dir={WEIGHTS_DIR} datasets_dir={DATASETS_DIR}")
    run("yolo cfg")


@pytest.mark.parametrize("api_key", ["legacy_api_key", "ul_" + "a" * 40])
def test_settings_migration(tmp_path: Path, api_key: str) -> None:
    """Verify schema migration preserves user settings and only retains Platform API keys."""
    from ultralytics.utils import SettingsManager

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "settings_version": "0.0.7",
                "runs_dir": "/custom/runs",
                "api_key": api_key,
                "hub": True,
                "neptune": True,
            }
        )
    )
    settings = SettingsManager(settings_file)

    assert settings["runs_dir"] == "/custom/runs"
    assert settings["api_key"] == (api_key if api_key.startswith("ul_") else "")
    assert settings["settings_version"] == "0.0.8"
    assert "hub" not in settings
    assert "neptune" not in settings


@pytest.mark.parametrize(
    "status,body",
    [
        (200, '{"username":"tester"}'),
        (200, "{}"),
        (201, '{"username":"tester"}'),
        (302, '{"username":"tester"}'),
        (401, "ul_secret"),
        (500, "ul_secret"),
    ],
)
def test_platform_login(tmp_path: Path, status: int, body: str) -> None:
    """Verify real login clients persist only validated keys and never retry through another endpoint."""
    from ultralytics.utils import SettingsManager

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, self.headers.get("Authorization")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    settings_file = tmp_path / "Ultralytics" / "settings.json"
    SettingsManager(settings_file)["api_key"] = "ul_previous"
    command = [sys.executable, "-c", "from ultralytics.cfg import entrypoint; entrypoint()"]
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        env = dict(
            os.environ,
            YOLO_CONFIG_DIR=str(tmp_path),
            ULTRALYTICS_PLATFORM_URL=f"http://127.0.0.1:{server.server_port}",
            ULTRALYTICS_API_KEY="ul_environment",
        )
        try:
            result = subprocess.run(
                [*command, "login", "ul_supplied"], env=env, capture_output=True, text=True, check=True
            )
            subprocess.run([*command, "login", ""], env=env, check=True)
        finally:
            server.shutdown()
            thread.join()
    expected = "ul_supplied" if status == 200 and (sys.version_info < (3, 11) or "username" in body) else "ul_previous"
    assert json.loads(settings_file.read_text())["api_key"] == expected
    assert received == [
        ("/api/account/summary" if sys.version_info >= (3, 11) else "/api/settings", "Bearer ul_supplied")
    ]
    assert "ul_secret" not in result.stdout + result.stderr
    subprocess.run([*command, "logout"], env=env, check=True)
    assert json.loads(settings_file.read_text())["api_key"] == ""


def test_cli_imports_defer_torchvision() -> None:
    """Verify startup imports do not load torchvision or SAM3 geometry."""
    code = (
        "import sys; "
        "from ultralytics import YOLO; "
        "from ultralytics.models.sam import Predictor; "
        "assert 'torchvision' not in sys.modules; "
        "assert 'ultralytics.models.sam.sam3.geometry_encoders' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train(task: str, model: str, data: str) -> None:
    """Test YOLO training for different tasks, models, and datasets."""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task: str, model: str, data: str) -> None:
    """Test YOLO validation process for specified task, model, and data using a shell command."""
    for end2end in (False, True):
        run(f"yolo val {task} model={model} data={data} imgsz=32 end2end={end2end} max_det=100 agnostic_nms")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task: str, model: str, data: str) -> None:
    """Test YOLO prediction on provided sample assets for specified task and model."""
    for end2end in (False, True):
        run(f"yolo {task} predict model={model} source={ASSETS} imgsz=32 save end2end={end2end} max_det=100")


@pytest.mark.parametrize("model", MODELS)
def test_export(model: str, tmp_path: Path) -> None:
    """Test exporting a YOLO model to TorchScript format."""
    from ultralytics.utils.downloads import attempt_download_asset

    isolated = tmp_path / model
    shutil.copy(Path(attempt_download_asset(model)), isolated)
    for end2end in (False, True):
        run(f"yolo export model={isolated} format=torchscript imgsz=32 end2end={end2end} max_det=100")


@pytest.mark.parametrize(
    "task,data,student,teacher",
    [
        ("detect", "coco8.yaml", "yolo26n.yaml", WEIGHTS_DIR / "yolo26s.pt"),
        ("segment", "coco8-seg.yaml", "yolo26n-seg.yaml", WEIGHTS_DIR / "yolo26s-seg.pt"),
        ("pose", "coco8-pose.yaml", "yolo26n-pose.yaml", WEIGHTS_DIR / "yolo26s-pose.pt"),
        ("obb", "dota8.yaml", "yolo26n-obb.yaml", WEIGHTS_DIR / "yolo26s-obb.pt"),
    ],
)
def test_distill(task: str, data: str, student: str, teacher: Path) -> None:
    """Test YOLO knowledge distillation training via CLI for supported tasks."""
    run(f"yolo train {task} model={student} distill_model={teacher} data={data} imgsz=32 epochs=1")


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
@pytest.mark.skipif(
    LINUX and ARM64 and checks.IS_PYTHON_3_8 and "2.1.0a0" in TORCH_VERSION,
    reason="RTDETR CPU training produces NaN losses with JetPack 5 torch 2.1.0a0",
)
def test_rtdetr(task: str = "detect", model: Path = WEIGHTS_DIR / "rtdetr-l.pt", data: str = "coco8.yaml") -> None:
    """Test the RTDETR functionality within Ultralytics for detection tasks using specified model and data."""
    # Add comma and spaces to test CLI arg cleanup.
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save")
    run(f"yolo train {task} model={model} data={data} --imgsz= 160 epochs =1, cache = disk")


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for heavy FastSAM tests")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="MobileSAM with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="MobileSAM with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_fastsam(
    task: str = "segment", model: str = WEIGHTS_DIR / "FastSAM-s.pt", data: str = "coco8-seg.yaml"
) -> None:
    """Test FastSAM model for segmenting objects in images using various prompts within Ultralytics."""
    source = ASSETS / "bus.jpg"

    run(f"yolo segment val {task} model={model} data={data} imgsz=32")
    run(f"yolo segment predict model={model} source={source} imgsz=32 save")

    from ultralytics import FastSAM
    from ultralytics.models.sam import Predictor

    # Create a FastSAM model
    sam_model = FastSAM(model)  # or FastSAM-x.pt

    # Run inference on an image
    for s in (source, Image.open(source)):
        everything_results = sam_model(s, device="cpu", retina_masks=True, imgsz=160, conf=0.4, iou=0.9)

        # Remove small regions
        _new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

        # Run inference with bboxes and points and texts prompt at the same time
        sam_model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")


def test_mobilesam() -> None:
    """Test MobileSAM segmentation with point and box prompts using Ultralytics."""
    from ultralytics import SAM

    # Load the model
    model = SAM(WEIGHTS_DIR / "mobile_sam.pt")

    # Source
    source = ASSETS / "zidane.jpg"

    # Predict a segment based on a 1D point prompt and 1D labels.
    model.predict(source, points=[900, 370], labels=[1])

    # Predict a segment based on 3D points and 2D labels (multiple points per object).
    model.predict(source, points=[[[900, 370], [1000, 100]]], labels=[[1, 1]])

    # Predict a segment based on a box prompt
    model.predict(source, bboxes=[439, 437, 524, 709], save=True)

    # Predict all
    # model(source)


# Slow Tests -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task: str, model: str, data: str) -> None:
    """Test YOLO training on GPU(s) for various tasks and models."""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # single GPU
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # multi GPU


@pytest.mark.parametrize(
    "solution",
    ["count", "blur", "workout", "heatmap", "isegment", "visioneye", "speed", "queue", "analytics", "trackzone"],
)
def test_solutions(solution: str) -> None:
    """Test yolo solutions command-line modes."""
    run(f"yolo solutions {solution} verbose=False")
