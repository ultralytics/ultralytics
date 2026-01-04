# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
import time
from pathlib import Path

import pytest

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
from ultralytics.utils.downloads import safe_download

# Lock TTL for video asset downloads (in seconds)
LOCK_TTL = 600


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    """Modify the list of test items to exclude tests marked as slow if the --slow option is not specified.

    Args:
        config: The pytest configuration object that provides access to command-line options.
        items (list): The list of collected pytest item objects to be modified based on the presence of --slow option.
    """
    if not config.getoption("--slow"):
        # Remove the item entirely from the list of test items if it's marked as 'slow'
        items[:] = [item for item in items if "slow" not in item.keywords]


def pytest_sessionstart(session):
    """Initialize session configurations for pytest.

    This function is automatically called by pytest after the 'Session' object has been created but before performing
    test collection. It sets the initial seeds for the test session.

    Args:
        session: The pytest session object.
    """
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Cleanup operations after pytest session.

    This function is automatically called by pytest at the end of the entire test session. It removes certain files and
    directories used during testing.

    Args:
        terminalreporter: The terminal reporter object used for terminal output.
        exitstatus (int): The exit status of the test run.
        config: The pytest config object.
    """
    from ultralytics.utils import WEIGHTS_DIR

    # Remove files
    models = [path for x in {"*.onnx", "*.torchscript"} for path in WEIGHTS_DIR.rglob(x)]
    for file in ["decelera_portrait_min.mov", "bus.jpg", "yolo11n.onnx", "yolo11n.torchscript", *models]:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    models = [path for x in {"*.mlpackage", "*_openvino_model"} for path in WEIGHTS_DIR.rglob(x)]
    for directory in [WEIGHTS_DIR / "path with spaces", *models]:
        shutil.rmtree(directory, ignore_errors=True)


# Mapping of task names to their standard weight files
MODEL_WEIGHTS = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}


@pytest.fixture(scope="session")
def model_factory():
    """Session-scoped factory for YOLO models.

    Notes:
    With pytest-xdist this is instantiated once per worker,
    which is thread-safe and still significantly faster than
    loading models per-test.
    """
    loaded_models = {}

    def _get_model(task: str):
        if task not in loaded_models:
            weight_file = MODEL_WEIGHTS.get(task)
            if not weight_file:
                raise ValueError(f"Unknown task: {task}")
            loaded_models[task] = YOLO(WEIGHTS_DIR / weight_file)
        return loaded_models[task]

    return _get_model


# Backwards-compatible task-specific fixtures
@pytest.fixture(scope="session")
def yolo_model(model_factory):
    return model_factory("detect")


@pytest.fixture(scope="session")
def yolo_seg_model(model_factory):
    return model_factory("segment")


@pytest.fixture(scope="session")
def yolo_cls_model(model_factory):
    return model_factory("classify")


@pytest.fixture(scope="session")
def yolo_pose_model(model_factory):
    return model_factory("pose")


@pytest.fixture(scope="session")
def yolo_obb_model(model_factory):
    return model_factory("obb")


@pytest.fixture(scope="session")
def solutions_videos(tmp_path_factory):
    """Pre-download solution videos once per session.

    Uses a simple lock directory to prevent race conditions when running with pytest-xdist.
    """
    # Shared directory across all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    video_dir = root_tmp_dir / "shared_video_assets"
    video_dir.mkdir(exist_ok=True)

    videos = [
        "solutions_ci_demo.mp4",
        "decelera_landscape_min.mov",
        "workout_small.mp4",
    ]

    lock_dir = video_dir / "download_lock"
    lock_acquired = False

    try:
        # Simple spin-lock (max ~30s)
        for _ in range(60):
            try:
                lock_dir.mkdir(exist_ok=False)
                lock_acquired = True
                break
            except FileExistsError:
                time.sleep(0.5)

        if not lock_acquired:
            raise TimeoutError("Could not acquire video asset download lock after 30s. Aborting test setup.")
        else:
            # Download missing assets
            for video in videos:
                if not (video_dir / video).exists():
                    safe_download(url=f"{ASSETS_URL}/{video}", dir=video_dir)
    finally:
        if lock_acquired and lock_dir.exists():
            lock_dir.rmdir()

    return video_dir
