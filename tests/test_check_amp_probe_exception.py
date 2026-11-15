# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests: check_amp must degrade gracefully when the AMP probe raises.

A backend error during any part of the AMP probe — probe-model construction, device
transfer, or the forward pass itself (e.g. HIP `hipErrorInvalidKernelFile` on a corrupt
MIOpen install, CUDA kernel-load failures) — must disable AMP with a logged error and
return False, never propagate into trainer setup and crash `train()`.
`KeyboardInterrupt`/`SystemExit` must still propagate (they are not `Exception` subclasses).

All tests are CPU-runnable: they inject faults at dependency boundaries (probe-model
construction/transfer/forward), so no accelerator hardware is required. The healthy-accelerator
positive control is covered by the existing `tests/test_cuda.py::test_amp`, which runs on any
CUDA-capable device (NVIDIA CI and AMD GPU runners).
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checks import check_amp

PROBE_SHAPE = (8, 3, 256, 256)  # check_amp probe batch; DetectionModel's own stride pass uses (1, 3, 256, 256)

# torch.cuda.OutOfMemoryError requires torch>=1.13; the guard treats it like any other Exception subclass
_TRANSFER_EXC_TYPES = [RuntimeError]
if hasattr(torch.cuda, "OutOfMemoryError"):
    _TRANSFER_EXC_TYPES.append(torch.cuda.OutOfMemoryError)


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
    from ultralytics.utils import LOGGER

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
    """Make check_amp's probe path deterministic on any host.

    Patches torch.cuda.get_device_name so check_amp never performs real CUDA enumeration and never hits the GTX 16xx
    exclusion list, regardless of the host GPU.
    """
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")


def test_check_amp_construction_failure_returns_false(monkeypatch, accelerator_probe_env, capture_amp_logs):
    """A failure while building the probe model must disable AMP, not crash training."""

    def _broken_init(self, *args, **kwargs):
        raise RuntimeError("simulated backend failure while building the probe model")

    monkeypatch.setattr(DetectionModel, "__init__", _broken_init)
    verdict = check_amp(_FakeAcceleratorModel())
    assert verdict is False, "check_amp must return False (not raise) when probe-model construction fails"
    assert any("checks failed with an exception" in m for m in capture_amp_logs), capture_amp_logs


@pytest.mark.parametrize("exc_type", _TRANSFER_EXC_TYPES)
def test_check_amp_transfer_failure_returns_false(monkeypatch, accelerator_probe_env, capture_amp_logs, exc_type):
    """A failure while moving the probe model to the device (e.g. tight memory) must disable AMP, not crash."""

    def _broken_to(self, *args, **kwargs):
        raise exc_type("simulated backend failure while transferring the probe model to the device")

    monkeypatch.setattr(DetectionModel, "to", _broken_to)
    verdict = check_amp(_FakeAcceleratorModel())
    assert verdict is False, "check_amp must return False (not raise) when probe-model device transfer fails"
    assert any("checks failed with an exception" in m for m in capture_amp_logs), capture_amp_logs


def test_check_amp_forward_failure_returns_false(monkeypatch, accelerator_probe_env, capture_amp_logs):
    """A backend error during the probe forward pass must disable AMP, not crash training."""
    original_forward = DetectionModel.forward

    def _forward_boom_on_probe_batch(self, x):
        # The DetectionModel constructor legitimately runs one (1, 3, 256, 256) stride pass;
        # only the probe's (8, 3, 256, 256) batch must fail.
        if tuple(x.shape) == PROBE_SHAPE:
            raise RuntimeError("simulated backend kernel failure during AMP probe")
        return original_forward(self, x)

    def _deviceless_rand(*args, **kwargs):
        kwargs.pop("device", None)  # keep input allocation on-CPU so this test also runs without a GPU
        return torch.zeros(*args, **kwargs)

    monkeypatch.setattr(DetectionModel, "to", lambda self, *args, **kwargs: self)  # keep transfer off-hardware
    monkeypatch.setattr(torch, "rand", _deviceless_rand)
    monkeypatch.setattr(DetectionModel, "forward", _forward_boom_on_probe_batch)
    verdict = check_amp(_FakeAcceleratorModel())
    assert verdict is False, "check_amp must return False (not raise) when the probe forward pass fails"
    assert any("checks failed with an exception" in m for m in capture_amp_logs), capture_amp_logs


def test_check_amp_repeated_invocation_after_failure(monkeypatch, accelerator_probe_env, capture_amp_logs):
    """check_amp must stay usable after a failed probe: no stale state, no exception on retry."""

    def _broken_to(self, *args, **kwargs):
        raise RuntimeError("simulated backend failure while transferring the probe model to the device")

    monkeypatch.setattr(DetectionModel, "to", _broken_to)
    for attempt in (1, 2):
        assert check_amp(_FakeAcceleratorModel()) is False, f"attempt {attempt} must return False"
    failures = [m for m in capture_amp_logs if "checks failed with an exception" in m]
    assert len(failures) == 2, capture_amp_logs


def test_check_amp_user_interruption_propagates(monkeypatch, accelerator_probe_env):
    """KeyboardInterrupt is not an Exception: it must propagate so a user's interruption is never swallowed."""

    def _interrupted_init(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(DetectionModel, "__init__", _interrupted_init)
    with pytest.raises(KeyboardInterrupt):
        check_amp(_FakeAcceleratorModel())
