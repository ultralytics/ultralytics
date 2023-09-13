# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
from pathlib import Path

import pytest

from ultralytics.utils import ROOT
from ultralytics.utils.torch_utils import init_seeds

TMP = (ROOT / '../tests/tmp').resolve()  # temp directory for test files


def pytest_addoption(parser):
    """Add custom command-line options to pytest.

    Args:
        parser (pytest.config.Parser): The pytest parser object.
    """
    parser.addoption('--slow', action='store_true', default=False, help='Run slow tests')


def pytest_configure(config):
    """Register custom markers to avoid pytest warnings.

    Args:
        config (pytest.config.Config): The pytest config object.
    """
    config.addinivalue_line('markers', 'slow: mark test as slow to run')


def pytest_runtest_setup(item):
    """Setup hook to skip tests marked as slow if the --slow option is not provided.

    Args:
        item (pytest.Item): The test item object.
    """
    if 'slow' in item.keywords and not item.config.getoption('--slow'):
        pytest.skip('skip slow tests unless --slow is set')


def pytest_sessionstart(session):
    """
    Initialize session configurations for pytest.

    This function is automatically called by pytest after the 'Session' object has been created but before performing
    test collection. It sets the initial seeds and prepares the temporary directory for the test session.

    Args:
        session (pytest.Session): The pytest session object.
    """
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
    # Remove files
    for file in ['bus.jpg', 'decelera_landscape_min.mov']:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    for directory in [ROOT / '../.pytest_cache', TMP]:
        shutil.rmtree(directory, ignore_errors=True)
