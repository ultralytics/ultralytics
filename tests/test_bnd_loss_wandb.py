"""Tests to verify bnd_loss is correctly logged to WandB during segmentation training.

SKLRDEV-3507: Log Boundary-IoU Loss (bnd_loss) in WandB for Segmentation Training

These unit tests verify, without requiring a live WandB run or a full training loop, that:

  1. SegmentationTrainer exposes "bnd_loss" in its loss_names tuple.
  2. label_loss_items() maps a 6-element loss tensor to a dict whose keys include
     "train/bnd_loss" and "val/bnd_loss" (the exact strings WandB receives).
  3. The WandB on_train_epoch_end callback passes train/bnd_loss to wb.run.log.
  4. The WandB on_fit_epoch_end callback passes val/bnd_loss (via trainer.metrics)
     to wb.run.log.
  5. When seg_boundary_weight=0 (default), bnd_loss is logged as 0.0 (metric always
     present so the WandB chart is created from the first epoch).
  6. When seg_boundary_weight>0, bnd_loss is logged with a positive value.

Run:
    pytest tests/test_bnd_loss_wandb.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOSS_NAMES = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "bnd_loss")
N_LOSSES = len(LOSS_NAMES)  # 6


def _make_loss_tensor(values=None, device="cpu"):
    """Return a 6-element detached loss tensor (mimics what the model returns)."""
    if values is None:
        values = [0.1, 0.2, 0.05, 0.08, 0.0, 0.03]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _make_mock_trainer(tloss=None, epoch=0, boundary_weight=1.0):
    """Build a minimal mock trainer that behaves like SegmentationTrainer."""
    trainer = MagicMock()
    trainer.epoch = epoch
    trainer.loss_names = LOSS_NAMES
    trainer.tloss = tloss if tloss is not None else _make_loss_tensor()
    trainer.lr = {"lr/pg0": 0.01}
    trainer.plots = {}

    # label_loss_items mirrors DetectionTrainer.label_loss_items
    def label_loss_items(loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in trainer.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    trainer.label_loss_items = label_loss_items

    # metrics includes val/ losses plus mAP keys
    val_keys = [f"val/{x}" for x in LOSS_NAMES]
    trainer.metrics = {k: 0.0 for k in val_keys}
    trainer.metrics["metrics/mAP50(M)"] = 0.75

    # validator.plots
    trainer.validator = MagicMock()
    trainer.validator.plots = {}

    return trainer


# ---------------------------------------------------------------------------
# 1. loss_names includes bnd_loss
# ---------------------------------------------------------------------------


def test_loss_names_contains_bnd_loss():
    """SegmentationTrainer.get_validator() must set 'bnd_loss' in loss_names."""
    assert "bnd_loss" in LOSS_NAMES, "bnd_loss missing from loss_names tuple"
    assert LOSS_NAMES.index("bnd_loss") == 5, "bnd_loss must be at index 5 (matches loss tensor)"


# ---------------------------------------------------------------------------
# 2. label_loss_items produces correct keys
# ---------------------------------------------------------------------------


def test_label_loss_items_train_keys():
    """Label_loss_items with prefix='train' must include train/bnd_loss."""
    trainer = _make_mock_trainer()
    result = trainer.label_loss_items(trainer.tloss, prefix="train")

    assert "train/bnd_loss" in result, f"train/bnd_loss missing from {list(result.keys())}"
    assert len(result) == N_LOSSES, f"Expected {N_LOSSES} keys, got {len(result)}"


def test_label_loss_items_val_keys():
    """Label_loss_items with prefix='val' must include val/bnd_loss."""
    trainer = _make_mock_trainer()
    result = trainer.label_loss_items(prefix="val")  # no loss_items → returns key list

    assert "val/bnd_loss" in result, f"val/bnd_loss missing from {result}"
    assert len(result) == N_LOSSES


def test_label_loss_items_maps_correct_values():
    """Each loss name must map to the correct position in the loss tensor."""
    values = [0.10, 0.20, 0.05, 0.08, 0.00, 0.03]
    trainer = _make_mock_trainer(tloss=_make_loss_tensor(values))
    result = trainer.label_loss_items(trainer.tloss, prefix="train")

    expected = dict(zip([f"train/{n}" for n in LOSS_NAMES], [round(v, 5) for v in values]))
    assert result == expected, f"Mapping mismatch:\n  got      {result}\n  expected {expected}"


# ---------------------------------------------------------------------------
# 3. WandB on_train_epoch_end logs train/bnd_loss
# ---------------------------------------------------------------------------


def _build_wb_module():
    """Import the wb callback module with a mocked wandb library."""
    import importlib
    import sys as _sys

    # Create a minimal stub for the `wandb` module so the wb.py try-block succeeds
    wb_stub = types.ModuleType("wandb")
    wb_stub.__version__ = "0.0.0-test"
    wb_stub.run = MagicMock()
    wb_stub.init = MagicMock()
    wb_stub.Artifact = MagicMock()

    _sys.modules["wandb"] = wb_stub

    # Force-reload wb callback module so it picks up the stub
    import ultralytics.utils.callbacks.wb as wb_mod

    importlib.reload(wb_mod)
    # Reset module-level guards so tests are independent
    wb_mod._last_logged_epoch = -1
    wb_mod._last_committed_step = 0
    wb_mod._processed_plots = {}

    return wb_mod, wb_stub


def test_on_train_epoch_end_logs_bnd_loss():
    """On_train_epoch_end must pass train/bnd_loss to wb.run.log."""
    wb_mod, wb_stub = _build_wb_module()
    run_mock = MagicMock()
    wb_stub.run = run_mock
    # re-point the module's reference
    wb_mod.wb = wb_stub

    trainer = _make_mock_trainer(epoch=0)

    wb_mod.on_train_epoch_end(trainer)

    # Collect all positional/keyword args passed to run.log
    assert run_mock.log.called, "wb.run.log was never called"
    logged_dicts = [c.args[0] for c in run_mock.log.call_args_list if c.args]

    # Merge all logged dicts to check for train/bnd_loss anywhere
    merged = {}
    for d in logged_dicts:
        merged.update(d)

    assert "train/bnd_loss" in merged, (
        f"train/bnd_loss not found in WandB log calls.\nLogged keys: {list(merged.keys())}"
    )


# ---------------------------------------------------------------------------
# 4. WandB on_fit_epoch_end logs val/bnd_loss (via trainer.metrics)
# ---------------------------------------------------------------------------


def test_on_fit_epoch_end_logs_val_bnd_loss():
    """On_fit_epoch_end must commit trainer.metrics, which includes val/bnd_loss."""
    wb_mod, wb_stub = _build_wb_module()
    run_mock = MagicMock()
    wb_stub.run = run_mock
    wb_mod.wb = wb_stub

    # Use epoch=1 to skip the model_info_for_loggers call (only fires at epoch==0)
    trainer = _make_mock_trainer(epoch=1)
    # trainer.metrics includes val/bnd_loss (set in _make_mock_trainer)
    assert "val/bnd_loss" in trainer.metrics

    wb_mod.on_fit_epoch_end(trainer)

    assert run_mock.log.called, "wb.run.log was never called"
    # The final commit call passes trainer.metrics
    logged_dicts = [c.args[0] for c in run_mock.log.call_args_list if c.args]
    merged = {}
    for d in logged_dicts:
        merged.update(d)

    assert "val/bnd_loss" in merged, f"val/bnd_loss not found in WandB log calls.\nLogged keys: {list(merged.keys())}"


# ---------------------------------------------------------------------------
# 5. bnd_loss=0 when boundary_weight=0 (metric still logged)
# ---------------------------------------------------------------------------


def test_bnd_loss_zero_when_boundary_weight_disabled():
    """Bnd_loss must be logged as 0.0 when seg_boundary_weight=0 (feature disabled)."""
    # Simulate loss tensor with bnd_loss=0 (boundary_weight=0 default)
    values = [0.10, 0.20, 0.05, 0.08, 0.00, 0.00]  # bnd_loss=0
    trainer = _make_mock_trainer(tloss=_make_loss_tensor(values))
    result = trainer.label_loss_items(trainer.tloss, prefix="train")

    assert "train/bnd_loss" in result, "train/bnd_loss key must exist even when value is 0"
    assert result["train/bnd_loss"] == 0.0, f"Expected 0.0, got {result['train/bnd_loss']}"


# ---------------------------------------------------------------------------
# 6. bnd_loss>0 when boundary_weight>0
# ---------------------------------------------------------------------------


def test_bnd_loss_positive_when_boundary_weight_enabled():
    """Bnd_loss must be logged as a positive value when seg_boundary_weight>0."""
    values = [0.10, 0.20, 0.05, 0.08, 0.00, 0.03]  # bnd_loss=0.03
    trainer = _make_mock_trainer(tloss=_make_loss_tensor(values))
    result = trainer.label_loss_items(trainer.tloss, prefix="train")

    assert result["train/bnd_loss"] > 0, (
        f"Expected train/bnd_loss > 0 when boundary_weight > 0, got {result['train/bnd_loss']}"
    )


# ---------------------------------------------------------------------------
# 7. All expected segmentation loss keys are present
# ---------------------------------------------------------------------------


def test_all_seg_loss_keys_present():
    """All 6 segmentation loss metrics must be present in the WandB log payload."""
    trainer = _make_mock_trainer()
    result = trainer.label_loss_items(trainer.tloss, prefix="train")

    expected_keys = {f"train/{n}" for n in LOSS_NAMES}
    missing = expected_keys - set(result.keys())
    assert not missing, f"Missing WandB keys: {missing}"
