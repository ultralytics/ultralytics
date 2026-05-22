# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path


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
    models = [path for x in {"*.mlpackage", "*_openvino_model"} for path in WEIGHTS_DIR.rglob(x)]
    for directory in [WEIGHTS_DIR / "solution_assets", WEIGHTS_DIR / "path with spaces", *models]:
        shutil.rmtree(directory, ignore_errors=True)
