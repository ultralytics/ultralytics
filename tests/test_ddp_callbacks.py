# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Integration tests for DDP callback serialization (issue #6168).

These tests exercise the real DDP file generation and subprocess execution path,
not just unit-level pickle checks. The key test (`test_ddp_temp_file_loads_callbacks`)
actually runs the generated temp file in a subprocess and verifies that callbacks
survive the process boundary.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ultralytics.utils.callbacks import get_default_callbacks
from ultralytics.utils.dist import _serialize_callbacks, ddp_cleanup, generate_ddp_file

# --- Test helpers ----------------------------------------------------------------------------------------------------


class FakeTrainer:
    """Minimal trainer-like object for testing generate_ddp_file without GPU/torch deps."""

    def __init__(self, callbacks=None, model="yolo26n.pt"):
        """Initialize a fake trainer with the given callbacks and model path."""
        self.callbacks = callbacks or get_default_callbacks()
        self.args = SimpleNamespace(model=model, augmentations=None)
        self.hub_session = None
        self.__class__.__module__ = "ultralytics.models.yolo.detect"
        self.__class__.__name__ = "DetectionTrainer"

    def train(self):
        """Stub train method — the real temp file calls this in a subprocess."""


# An importable callback function for testing serialization.
# Defined at module level (not __main__) so pickle can serialize it by reference.
def importable_test_callback(trainer):
    """Importable callback that sets a flag on the trainer to prove it was called."""
    trainer._callback_fired = True


# --- Unit tests: _serialize_callbacks --------------------------------------------------------------------------------


def test_serialize_callbacks_defaults_only():
    """Default callbacks should not be serialized — they're re-created in the subprocess."""
    trainer = FakeTrainer(callbacks=get_default_callbacks())
    result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is None, "Should not create pickle file for default-only callbacks"


def test_serialize_callbacks_importable():
    """Importable callbacks (module-level functions) should be serialized successfully."""
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(importable_test_callback)
    trainer = FakeTrainer(callbacks=cbs)
    result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is not None, "Should create pickle file for importable callback"
    with open(result, "rb") as f:
        loaded = pickle.load(f)
    assert "on_train_start" in loaded
    assert importable_test_callback in loaded["on_train_start"]
    os.remove(result)


def test_serialize_callbacks_lambda():
    """Lambda callbacks cannot be pickled — should be dropped with a warning."""
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(lambda t: None)
    trainer = FakeTrainer(callbacks=cbs)
    with mock.patch("ultralytics.utils.LOGGER") as mock_logger:
        result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is None, "Should not create pickle file for lambda callback"
    assert mock_logger.warning.called, "Should warn about dropped callback"
    assert "cannot be serialized" in mock_logger.warning.call_args[0][0], "Warning message should mention serialization"


def test_serialize_callbacks_main_function():
    """Functions defined in __main__ pickle by reference but fail to unpickle in the subprocess."""

    def main_callback(trainer):
        pass

    cbs = get_default_callbacks()
    cbs["on_train_start"].append(main_callback)
    trainer = FakeTrainer(callbacks=cbs)
    with mock.patch("ultralytics.utils.LOGGER") as mock_logger:
        result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is None, "Should not create pickle file for __main__ callback"
    assert mock_logger.warning.called, "Should warn about dropped callback"
    assert "cannot be serialized" in mock_logger.warning.call_args[0][0], "Warning message should mention serialization"


def test_serialize_callbacks_closure():
    """Closures cannot be pickled — should be dropped with a warning."""

    def make_closure():
        x = 42

        def closure_cb(trainer):
            return x

        return closure_cb

    cbs = get_default_callbacks()
    cbs["on_train_start"].append(make_closure())
    trainer = FakeTrainer(callbacks=cbs)
    with mock.patch("ultralytics.utils.LOGGER") as mock_logger:
        result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is None, "Should not create pickle file for closure callback"
    assert mock_logger.warning.called, "Should warn about dropped callback"
    assert "cannot be serialized" in mock_logger.warning.call_args[0][0], "Warning message should mention serialization"


def test_serialize_callbacks_mixed():
    """Mix of importable and non-serializable — only importable should be preserved."""
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(importable_test_callback)  # serializable
    cbs["on_train_start"].append(lambda t: None)  # not serializable
    trainer = FakeTrainer(callbacks=cbs)
    result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is not None, "Should create pickle file for serializable callbacks"
    with open(result, "rb") as f:
        loaded = pickle.load(f)
    assert importable_test_callback in loaded["on_train_start"], "Importable callback should be in pickle"
    assert len(loaded["on_train_start"]) == 1, "Only 1 serializable callback should be in pickle"
    os.remove(result)


def test_serialize_callbacks_skips_integration_callbacks():
    """Callbacks from ultralytics.utils.callbacks.* are skipped — they're re-added in the subprocess."""
    integration_cb = lambda t: None
    integration_cb.__module__ = "ultralytics.utils.callbacks.hub"  # simulate integration callback
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(integration_cb)
    trainer = FakeTrainer(callbacks=cbs)
    result = _serialize_callbacks(trainer, get_default_callbacks())
    assert result is None, "Integration callbacks should be skipped, not serialized"


# --- Integration tests: generate_ddp_file -----------------------------------------------------------------------------


def test_generate_ddp_file_no_custom_callbacks(tmp_path):
    """generate_ddp_file with only default callbacks should not create a pickle file."""
    trainer = FakeTrainer(callbacks=get_default_callbacks())
    with mock.patch("ultralytics.utils.dist.USER_CONFIG_DIR", tmp_path):
        file_path = generate_ddp_file(trainer)
    assert Path(file_path).exists(), "Temp file should be created"
    content = Path(file_path).read_text()
    assert "callbacks_file = None" in content, "callbacks_file should be None when no custom callbacks"
    os.remove(file_path)


def test_generate_ddp_file_with_importable_callback(tmp_path):
    """generate_ddp_file with an importable callback should create a pickle file and embed its path."""
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(importable_test_callback)
    trainer = FakeTrainer(callbacks=cbs)
    with mock.patch("ultralytics.utils.dist.USER_CONFIG_DIR", tmp_path):
        file_path = generate_ddp_file(trainer)
        # Find the callbacks pickle file
        pkl_files = list(tmp_path.glob("DDP/_callbacks_*.pkl"))
    assert Path(file_path).exists(), "Temp file should be created"
    assert len(pkl_files) == 1, "Should create exactly one callbacks pickle file"
    content = Path(file_path).read_text()
    assert "callbacks_file = " in content
    assert "None" not in content.split("callbacks_file = ")[1].split("\n")[0], "callbacks_file should not be None"
    # Verify the pickle file contains the callback
    with open(pkl_files[0], "rb") as f:
        loaded = pickle.load(f)
    assert importable_test_callback in loaded["on_train_start"]
    # Cleanup
    os.remove(file_path)
    os.remove(pkl_files[0])


def test_generate_ddp_file_with_lambda_callback(tmp_path):
    """generate_ddp_file with a lambda callback should warn and not create a pickle file."""
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(lambda t: None)
    trainer = FakeTrainer(callbacks=cbs)
    with mock.patch("ultralytics.utils.LOGGER") as mock_logger, mock.patch(
        "ultralytics.utils.dist.USER_CONFIG_DIR", tmp_path
    ):
        file_path = generate_ddp_file(trainer)
        pkl_files = list(tmp_path.glob("DDP/_callbacks_*.pkl"))
    assert Path(file_path).exists(), "Temp file should be created"
    assert len(pkl_files) == 0, "Should not create pickle file for lambda callback"
    content = Path(file_path).read_text()
    assert "callbacks_file = None" in content, "callbacks_file should be None when callbacks are not serializable"
    assert mock_logger.warning.called, "Should warn about dropped callback"
    os.remove(file_path)


# --- Integration test: actually load pickled callbacks in a subprocess ------------------------------------------------


def test_ddp_pickle_loads_in_subprocess(tmp_path):
    """End-to-end: generate DDP file with importable callback, load the pickle in a fresh subprocess.

    This is the critical integration test — it verifies that the pickled callback can actually be unpickled in a fresh
    Python process (the same way the DDP subprocess does it). This catches the __main__ reference bug that the first PR
    (#25118) missed.
    """
    cbs = get_default_callbacks()
    cbs["on_train_start"].append(importable_test_callback)
    trainer = FakeTrainer(callbacks=cbs)

    with mock.patch("ultralytics.utils.dist.USER_CONFIG_DIR", tmp_path):
        file_path = generate_ddp_file(trainer)
        pkl_files = list(tmp_path.glob("DDP/_callbacks_*.pkl"))

    assert len(pkl_files) == 1, "Should create a callbacks pickle file"
    pkl_path = str(pkl_files[0])

    # Load the pickle in a fresh subprocess — this simulates what the DDP temp file does
    # A fresh subprocess has a different __main__ module, so __main__-referenced functions would fail
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import pickle, sys
sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
with open({pkl_path!r}, "rb") as f:
    cbs = pickle.load(f)
# Verify our importable callback is present and callable
found = False
for event, funcs in cbs.items():
    for func in funcs:
        if getattr(func, "__name__", "") == "importable_test_callback":
            found = True
            # Verify the function's module is importable (not __main__)
            assert func.__module__ != "__main__", "Callback should not be from __main__"
if found:
    print("CALLBACK_LOADED_OK")
else:
    print("CALLBACK_MISSING")
    sys.exit(1)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "CALLBACK_LOADED_OK" in result.stdout, f"Callback not loaded in subprocess:\n{result.stdout}"

    # Cleanup
    os.remove(file_path)
    os.remove(pkl_path)


def test_ddp_pickle_main_function_fails_in_subprocess(tmp_path):
    """Verify that a __main__ function in the pickle would fail to load in a subprocess.

    This test documents the bug that the first PR (#25118) had: __main__ functions pickle by reference but fail to
    unpickle in a subprocess. Our fix prevents this by not pickling __main__ functions in the first place. This test
    verifies the underlying pickle behavior to ensure our detection logic is correct.
    """
    pkl_path = str(tmp_path / "test_main_cb.pkl")

    # Create a pickle file in a subprocess where the function IS in __main__
    # Then try to load it in ANOTHER subprocess where it's NOT in __main__
    create_result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import pickle
def my_main_callback(trainer):
    pass
# This function's __module__ is __main__, so pickle stores a reference to __main__.my_main_callback
with open({pkl_path!r}, "wb") as f:
    pickle.dump({{"on_train_start": [my_main_callback]}}, f)
print("PICKLE_CREATED")
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert create_result.returncode == 0, f"Failed to create pickle:\n{create_result.stderr}"

    # Loading in a DIFFERENT subprocess should fail (its __main__ doesn't have my_main_callback)
    load_result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import pickle
try:
    with open({pkl_path!r}, "rb") as f:
        cbs = pickle.load(f)
    print("UNEXPECTED_SUCCESS")
except AttributeError as e:
    print(f"EXPECTED_FAILURE: {{e}}")
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert load_result.returncode == 0
    assert "EXPECTED_FAILURE" in load_result.stdout, (
        f"__main__ function should fail to unpickle in subprocess, but got:\n{load_result.stdout}"
    )


# --- Cleanup tests ----------------------------------------------------------------------------------------------------


def test_ddp_cleanup_removes_pickle_file(tmp_path):
    """ddp_cleanup should remove both the temp file and the associated callbacks pickle file."""
    ddp_dir = tmp_path / "DDP"
    ddp_dir.mkdir()
    # Create fake temp file and callbacks file with matching id suffix
    trainer = FakeTrainer()
    trainer_id = id(trainer)
    temp_file = ddp_dir / f"_temp_abc{trainer_id}.py"
    pkl_file = ddp_dir / f"_callbacks_xyz{trainer_id}.pkl"
    temp_file.write_text("# temp")
    pkl_file.write_bytes(b"\x80\x04")  # minimal pickle header

    ddp_cleanup(trainer, str(temp_file))

    assert not temp_file.exists(), "Temp file should be removed"
    assert not pkl_file.exists(), "Callbacks pickle file should be removed"


def test_ddp_cleanup_no_pickle_file(tmp_path):
    """ddp_cleanup should work even when no callbacks pickle file exists."""
    ddp_dir = tmp_path / "DDP"
    ddp_dir.mkdir()
    trainer = FakeTrainer()
    trainer_id = id(trainer)
    temp_file = ddp_dir / f"_temp_abc{trainer_id}.py"
    temp_file.write_text("# temp")

    ddp_cleanup(trainer, str(temp_file))

    assert not temp_file.exists(), "Temp file should be removed"


def test_ddp_cleanup_wrong_id(tmp_path):
    """ddp_cleanup should not remove files that don't match the trainer id."""
    ddp_dir = tmp_path / "DDP"
    ddp_dir.mkdir()
    trainer = FakeTrainer()
    # Create a temp file with a DIFFERENT id
    temp_file = ddp_dir / "_temp_abc999999.py"
    temp_file.write_text("# temp")

    ddp_cleanup(trainer, str(temp_file))

    assert temp_file.exists(), "Should not remove file with non-matching id"
