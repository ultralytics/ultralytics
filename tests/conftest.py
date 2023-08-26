# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
from pathlib import Path

import pytest

from ultralytics.utils import ROOT
from ultralytics.utils.torch_utils import init_seeds

TMP = (ROOT / '../tests/tmp').resolve()  # temp directory for test files


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='run slow tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: mark test as slow to run')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


def pytest_sessionstart(session):
    """
    Called after the 'Session' object has been created and before performing test collection.
    """
    init_seeds()
    shutil.rmtree(TMP, ignore_errors=True)  # delete any existing tests/tmp directory
    TMP.mkdir(parents=True, exist_ok=True)  # create a new empty directory


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Remove files
    for file in ['bus.jpg', 'decelera_landscape_min.mov']:
        Path(file).unlink(missing_ok=True)

    # Remove directories
    for directory in [ROOT / '../.pytest_cache', TMP]:
        shutil.rmtree(directory, ignore_errors=True)
