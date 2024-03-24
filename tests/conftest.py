# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
from pathlib import Path

TMP = Path(__file__).resolve().parent / "tmp"  # temp directory for test files


def pytest_addoption(parser):
    """
    Add custom command-line options to pytest.

    Args:
        parser (pytest.config.Parser): The pytest parser object.
    """
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    """
    Modify the list of test items to remove tests marked as slow if the --slow option is not provided.

    Args:
        config (pytest.config.Config): The pytest config object.
        items (list): List of test items to be executed.
    """
    if not config.getoption("--slow"):
        # Remove the item entirely from the list of test items if it's marked as 'slow'
        items[:] = [item for item in items if "slow" not in item.keywords]


def pytest_sessionstart(session):
    """
    Initialize session configurations for pytest.

    This function is automatically called by pytest after the 'Session' object has been created but before performing
    test collection. It sets the initial seeds and prepares the temporary directory for the test session.

    Args:
        session (pytest.Session): The pytest session object.
    """
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds()
    shutil.rmtree(TMP, ignore_errors=True)  # delete any existing tests/tmp directory
    TMP.mkdir(parents=True, exist_ok=True)  # create a new empty directory


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Cleanup operations after pytest session.

    This function is automatically called by pytest at the end of the entire test session. It removes certain files
    and directories used during testing.

    Args:
        terminalreporter (pytest.terminal.TerminalReporter): The terminal reporter object.
        exitstatus (int): The exit status of the test run.
        config (pytest.config.Config): The pytest config object.
    """
    from ultralytics.utils import WEIGHTS_DIR

    # Remove files
    models = [path for x in ["*.onnx", "*.torchscript"] for path in WEIGHTS_DIR.rglob(x)]
    for file in ["bus.jpg", "yolov8n.onnx", "yolov8n.torchscript"] + models:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    models = [path for x in ["*.mlpackage", "*_openvino_model"] for path in WEIGHTS_DIR.rglob(x)]
    for directory in [TMP.parents[1] / ".pytest_cache", TMP] + models:
        shutil.rmtree(directory, ignore_errors=True)
