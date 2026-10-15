# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ultralytics.utils import dist


class FailingTrainer:
    """Trainer stand-in whose training raises, as a DDP rank would."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Accept the arguments the DDP temp file passes."""
        self.model = None

    def train(self):
        """Raise the error a rank hits mid-training."""
        raise KeyError("keypoints")


def _raise(exc):
    raise exc


def test_report_failure_writes_headline_and_trace(tmp_path, monkeypatch):
    path = tmp_path / "out" / "failure"
    monkeypatch.setenv(dist.FAILURE_FILE_ENV, str(path))
    try:
        _raise(ValueError("bad label"))
    except ValueError as exc:
        dist.report_failure(exc)
    report = path.read_text()
    assert report.startswith("ValueError: bad label")
    assert "in _raise" in report


def test_report_failure_keeps_first_report(tmp_path, monkeypatch):
    path = tmp_path / "failure"
    path.write_text("first")
    monkeypatch.setenv(dist.FAILURE_FILE_ENV, str(path))
    dist.report_failure(RuntimeError("second"))
    assert path.read_text() == "first"


def test_report_failure_noop_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(dist.FAILURE_FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    dist.report_failure(RuntimeError("x"))
    assert not any(tmp_path.iterdir())


def test_report_failure_fits_budget(tmp_path, monkeypatch):
    path = tmp_path / "failure"
    monkeypatch.setenv(dist.FAILURE_FILE_ENV, str(path))
    try:
        _raise(RuntimeError("é" * 5000))
    except RuntimeError as exc:
        dist.report_failure(exc)
    report = path.read_bytes()
    assert len(report) <= dist.FAILURE_REPORT_MAX_BYTES
    assert report.decode("utf-8").rstrip().endswith("in _raise")


def test_ddp_file_reports_rank_error(tmp_path):
    trainer = FailingTrainer()
    trainer.args = SimpleNamespace(model="m.pt")
    trainer.callbacks = {}
    file = Path(dist.generate_ddp_file(trainer))
    report = tmp_path / "failure"
    try:
        result = subprocess.run(
            [sys.executable, str(file)],
            env={**os.environ, dist.FAILURE_FILE_ENV: str(report)},
            capture_output=True,
            text=True,
        )
    finally:
        file.unlink(missing_ok=True)
        file.with_suffix(".pt").unlink(missing_ok=True)
    assert result.returncode != 0, result.stderr
    assert report.read_text().startswith("KeyError: 'keypoints'"), result.stderr
