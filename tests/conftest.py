# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

import numpy.testing  # noqa: F401  # Pre-import before any test can corrupt numpy via in-place upgrade
import pytest


@pytest.fixture(scope="session")
def solution_assets():
    """Return cached solution asset paths by name."""
    from tests import SOLUTION_ASSETS
    from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
    from ultralytics.utils.downloads import safe_download

    cache_dir = WEIGHTS_DIR / "solution_assets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_asset(name):
        asset_path = cache_dir / SOLUTION_ASSETS[name]
        if not asset_path.exists():
            safe_download(url=f"{ASSETS_URL}/{asset_path.name}", dir=cache_dir)
        return asset_path

    return get_asset


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


@pytest.fixture
def isolated_model(tmp_path):
    """Provide an isolated copy of the test model to prevent export file races under pytest-xdist.

    When multiple xdist workers run export tests simultaneously, they derive output filenames from the model path (e.g.,
    model.onnx, model.torchscript). Using the same MODEL path causes workers to overwrite each other's
    intermediate/export files. This fixture copies the shared model to a per-test temporary directory so each test
    exports to a unique path.
    """
    from tests import MODEL

    if not Path(MODEL).exists():
        from ultralytics.utils.downloads import attempt_download_asset

        MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(attempt_download_asset("yolo26n.pt"), MODEL)

    dst = tmp_path / "model.pt"
    shutil.copy(MODEL, dst)
    return str(dst)


def pytest_sessionfinish(session, exitstatus):
    """Cleanup operations after pytest session.

    Runs only on the pytest controller (or serial run), skipping xdist workers to avoid race conditions where one worker
    deletes shared assets while another is still reading them.
    """
    # Skip on xdist workers - only the controller should clean up shared resources
    if hasattr(session.config, "workerinput"):
        return

    from ultralytics.utils import WEIGHTS_DIR

    # Remove files
    models = [path for x in {"*.onnx", "*.torchscript"} for path in WEIGHTS_DIR.rglob(x)]
    for file in ["decelera_portrait_min.mov", "bus.jpg", "yolo26n.onnx", "yolo26n.torchscript", *models]:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    for directory in [path for x in {"*.mlpackage", "*_openvino_model"} for path in WEIGHTS_DIR.rglob(x)]:
        shutil.rmtree(directory, ignore_errors=True)
