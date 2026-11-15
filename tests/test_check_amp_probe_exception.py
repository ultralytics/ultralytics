# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal regression coverage: check_amp must degrade to FP32 when its accelerator probe raises.

PR #26967 accidentally dropped the probe's historical failure contract (8.4.182 disabled AMP with a
warning; 8.4.197+ propagated backend errors into trainer setup and crashed `train()`). These two tests
keep that contract from regressing again on ordinary CPU-only CI runners: a fake accelerator model routes
check_amp into its probe path and a fault is injected at the probe boundary, because the real failure mode
(backend kernel-load errors, OOM) cannot be reproduced safely on healthy hardware. The healthy path is
covered by the existing `tests/test_cuda.py::test_amp` on real accelerator CI targets.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_amp


class _FakeAcceleratorModel:
    """Stand-in exposing an accelerator device so check_amp takes its probe path without real hardware."""

    def parameters(self):
        """Yield one parameter-like object whose device claims accelerator placement."""
        p = SimpleNamespace()
        p.device = torch.device("cuda")  # ROCm/HIP reports device.type == "cuda" too
        return iter([p])


@pytest.fixture
def capture_amp_logs():
    """Capture ultralytics LOGGER records for the duration of a test."""
    records = []

    class _Capture(logging.Handler):
        """Logging handler appending formatted messages to a list."""

        def emit(self, record):
            """Append the record's message to the shared list."""
            records.append(record.getMessage())

    handler = _Capture()
    LOGGER.addHandler(handler)  # ultralytics LOGGER does not propagate to root
    try:
        yield records
    finally:
        LOGGER.removeHandler(handler)


@pytest.fixture
def accelerator_probe_env(monkeypatch):
    """Make check_amp deterministic on GPU-less hosts (avoid real CUDA enumeration)."""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")


def test_check_amp_probe_exception_degrades_to_fp32(monkeypatch, accelerator_probe_env, capture_amp_logs):
    """A backend error anywhere in the probe must disable AMP with a logged error, never crash training."""

    def _broken_init(self, *args, **kwargs):
        raise RuntimeError("simulated backend failure while building the probe model")

    monkeypatch.setattr(DetectionModel, "__init__", _broken_init)
    assert check_amp(_FakeAcceleratorModel()) is False
    assert any("checks failed with an exception" in m for m in capture_amp_logs), capture_amp_logs


def test_check_amp_user_interruption_propagates(monkeypatch, accelerator_probe_env):
    """KeyboardInterrupt is not an Exception: it must propagate so a user's interruption is never swallowed."""

    def _interrupted_init(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(DetectionModel, "__init__", _interrupted_init)
    with pytest.raises(KeyboardInterrupt):
        check_amp(_FakeAcceleratorModel())
