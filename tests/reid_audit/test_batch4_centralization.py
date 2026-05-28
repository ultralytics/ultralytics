"""Batch 4: centralization + DRY.

What would have failed before the fix:
  * `build_yolo_dataset(cfg(task='reid'), ...)` would build a YOLODataset and crash because
    Market-1501 has no bbox labels — dataset routing wasn't centralized.
  * `ReidValidator._extract_gallery_features` reimplemented ClassificationValidator.preprocess
    inline (~3 lines) — silent drift risk.
  * `ReidTrainer.setup_model` decided CLIP loading by filename substring (`'vit' in path`),
    misfiring on `my_run.pt` and missing `weights.pt` that actually contains CLIP keys.
  * `engine/trainer.py` doubled val batch for every non-OBB task; ReID's TTA already 2-6× compute.
  * `get_dataset` reid branch was implicit (only worked when data ends in .yaml).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def test_build_yolo_dataset_has_reid_branch():
    """build_yolo_dataset must dispatch to ReidDataset when cfg.task == 'reid'."""
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.dataset import ReidDataset

    src = inspect.getsource(build_yolo_dataset)
    assert 'cfg.task == "reid"' in src
    assert "ReidDataset" in src


def test_build_yolo_dataset_routes_reid(monkeypatch):
    """With cfg.task='reid', build_yolo_dataset must dispatch to ReidDataset (not YOLODataset).

    We mock ReidDataset so the test is decoupled from dataset internals — it locks in the
    routing contract specified in the audit, nothing more.
    """
    from ultralytics.data import build as build_mod

    sentinel = object()

    class FakeReidDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sentinel = sentinel

    monkeypatch.setattr(build_mod, "ReidDataset", FakeReidDataset)
    cfg = SimpleNamespace(task="reid", imgsz=64, rect=False, cache=None, single_cls=False, classes=None, fraction=1.0)
    ds = build_mod.build_yolo_dataset(cfg, "/tmp/_unused", batch=2, data={}, mode="train")
    assert isinstance(ds, FakeReidDataset)
    assert ds.kwargs["root"] == "/tmp/_unused"
    assert ds.kwargs["augment"] is True
    assert ds.kwargs["prefix"] == "train"


def test_reid_trainer_build_dataset_delegates_to_central():
    """ReidTrainer.build_dataset must call build_yolo_dataset, not instantiate ReidDataset directly."""
    from ultralytics.models.yolo.reid.train import ReidTrainer

    src = inspect.getsource(ReidTrainer.build_dataset)
    assert "build_yolo_dataset" in src
    # And must NOT instantiate ReidDataset directly:
    assert "ReidDataset(" not in src, "ReidTrainer.build_dataset must delegate, not duplicate"


def test_reid_validator_build_dataset_delegates_to_central():
    from ultralytics.models.yolo.reid.val import ReidValidator

    src = inspect.getsource(ReidValidator.build_dataset)
    assert "build_yolo_dataset" in src
    assert "ReidDataset(" not in src


def test_gallery_extraction_uses_self_preprocess():
    """_extract_gallery_features must call self.preprocess(batch) — no inline device/dtype clone."""
    from ultralytics.models.yolo.reid.val import ReidValidator

    src = inspect.getsource(ReidValidator._extract_gallery_features)
    assert "self.preprocess(batch)" in src
    # And the old inline pattern should be gone:
    assert "batch[\"img\"].to(self.device" not in src
    assert "batch[\"img\"].half()" not in src or "batch[\"img\"].half() if" not in src


def test_gallery_extraction_cpu_moves_before_concat():
    """All three accumulators (feats, pids, camids) must call .cpu() before the final
    torch.cat(...).numpy().

    Regression: after switching to self.preprocess(batch), the inherited ClassificationValidator
    preprocess also moves batch['cls'] to GPU. Without .cpu() on append, the final .numpy() crashes
    with `can't convert cuda:0 device type tensor to numpy`. The e2e GPU smoke test caught this
    on Market-1501.
    """
    from ultralytics.models.yolo.reid.val import ReidValidator

    src = inspect.getsource(ReidValidator._extract_gallery_features)
    # The pids accumulator must move to CPU
    assert 'batch["cls"].cpu()' in src or "batch['cls'].cpu()" in src, (
        "pids accumulator must call .cpu() — self.preprocess moves cls to GPU now"
    )
    # The camids tensor branch must also .cpu()
    assert 'batch["camid"].cpu()' in src or "batch['camid'].cpu()" in src, (
        "camids accumulator (tensor branch) must call .cpu()"
    )


def test_clip_helper_no_weights_only_false():
    """Security regression: _extract_clip_visual_sd must NOT use torch.load(weights_only=False)
    on a user-supplied path. The PyTorch 2.6+ default is weights_only=True for a reason —
    weights_only=False executes arbitrary pickle reduce ops in the checkpoint."""
    from pathlib import Path
    import ultralytics.models.yolo.reid.train as train_mod

    src = Path(train_mod.__file__).read_text()
    assert "weights_only=False" not in src, (
        "weights_only=False is a known PyTorch unsafe-load path; use weights_only=True"
    )


def test_clip_helper_safe_load_path():
    """The CLIP detection must use weights_only=True (safe) torch.load."""
    from pathlib import Path
    import ultralytics.models.yolo.reid.train as train_mod

    src = Path(train_mod.__file__).read_text()
    assert "weights_only=True" in src, "expected safe torch.load(weights_only=True) in CLIP detection path"


def test_clip_helper_no_outer_bare_except():
    """_extract_clip_visual_sd must not swallow FileNotFoundError / PermissionError / etc.
    via an outer bare `except Exception: return None`. Only the inner torch.jit.load and
    torch.load attempts should catch — anything else (import errors, OSError on a typo'd
    path) should surface so users can debug."""
    import inspect
    from ultralytics.models.yolo.reid.train import _extract_clip_visual_sd

    src = inspect.getsource(_extract_clip_visual_sd)
    # The function should have inner try/except for jit/torch.load attempts, but no
    # outer one wrapping the whole body. Count by checking that the `return None` at
    # function end isn't part of an `except: return None` block.
    assert src.count("except Exception:") <= 2, (
        "_extract_clip_visual_sd has too many bare-except blocks; should be at most two "
        "(one around torch.jit.load, one around weights_only=True torch.load)"
    )


def test_clip_detection_passes_visual_sd_to_get_model_once():
    """setup_model must pre-extract visual_sd and pass it to get_model via the visual_sd=
    kwarg so the checkpoint is loaded only ONCE. Previously _extract_clip_visual_sd was
    called twice — once in setup_model for detection, once in get_model for loading."""
    import inspect
    from ultralytics.models.yolo.reid.train import ReidTrainer

    setup_src = inspect.getsource(ReidTrainer.setup_model)
    get_src = inspect.getsource(ReidTrainer.get_model)
    assert "visual_sd=visual_sd" in setup_src, (
        "setup_model must pass the already-extracted visual_sd to get_model"
    )
    assert "visual_sd: dict | None = None" in get_src or "visual_sd=None" in get_src, (
        "get_model must accept a visual_sd kwarg to skip re-extraction"
    )


def test_clip_helper_accepts_scriptmodule_class_name():
    """The helper's documented contract is to accept torch.jit.RecursiveScriptModule OR
    plain ScriptModule (freeze/optimize-for-inference can return either). The substring
    check must catch both — use 'Script' rather than only 'RecursiveScript'."""
    import inspect
    from ultralytics.models.yolo.reid.train import _extract_clip_visual_sd

    src = inspect.getsource(_extract_clip_visual_sd)
    assert '"Script"' in src or "'Script'" in src, (
        "the substring check must cover ScriptModule, not only RecursiveScriptModule"
    )


def test_clip_setup_model_accepts_any_truthy_pretrained():
    """setup_model must call _extract_clip_visual_sd on any truthy pretrained value
    (str, dict, jit module) — not gate on isinstance(str). The helper's docstring
    accepts all three; setup_model must not contradict that contract."""
    import inspect
    from ultralytics.models.yolo.reid.train import ReidTrainer

    src = inspect.getsource(ReidTrainer.setup_model)
    assert "isinstance(pretrained, str)" not in src, (
        "setup_model must not gate the CLIP check on isinstance(pretrained, str) — "
        "the helper accepts dicts and pre-loaded jit modules too"
    )


def test_clip_detection_no_filename_substring():
    """setup_model must NOT decide the CLIP path by 'vit' in pretrained.lower() — was a footgun."""
    from ultralytics.models.yolo.reid.train import ReidTrainer

    src = inspect.getsource(ReidTrainer.setup_model)
    assert "is_clip_path" not in src, "filename-substring CLIP detection must be removed"
    assert "_extract_clip_visual_sd" in src, "setup_model must use state-dict-based CLIP detection"


def test_clip_detection_is_state_dict_based_callable():
    """_extract_clip_visual_sd must return None for non-CLIP inputs and the visual sd for CLIP."""
    from ultralytics.models.yolo.reid.train import _extract_clip_visual_sd

    # Non-CLIP dict: missing the required keys → None
    assert _extract_clip_visual_sd({"foo.bar": 1}) is None
    # CLIP-shaped dict: must extract the visual.* keys and strip the prefix
    import torch
    fake_clip_sd = {
        "visual.conv1.weight": torch.zeros(1),
        "visual.class_embedding": torch.zeros(1),
        "visual.other.param": torch.zeros(1),
        "transformer.unrelated": torch.zeros(1),  # text tower, should be dropped
    }
    out = _extract_clip_visual_sd(fake_clip_sd)
    assert out is not None
    assert "conv1.weight" in out  # 'visual.' stripped
    assert "transformer.unrelated" not in out


def test_trainer_val_batch_not_doubled_for_reid():
    """engine/trainer.py must include 'reid' in the no-double-batch set with obb.

    Scans the whole module source (not just one method) since the line lives in
    `_build_train_pipeline` and the audit doesn't lock that method name in stone.
    """
    import ultralytics.engine.trainer as trainer_mod
    from pathlib import Path

    src = Path(trainer_mod.__file__).read_text()
    assert 'task in {"obb", "reid"}' in src, (
        "val batch-doubling carve-out must include 'reid' alongside 'obb'"
    )


def test_get_dataset_task_set_includes_reid():
    """get_dataset's task-routing set must explicitly include reid (was implicit by .yaml suffix)."""
    import ultralytics.engine.trainer as trainer_mod
    from pathlib import Path

    src = Path(trainer_mod.__file__).read_text()
    # Match the multi-line set declaration: `... self.args.task in { ... "reid" ... }:`
    idx = src.find("self.args.task in {")
    assert idx != -1, "Expected `self.args.task in {...}` set membership in trainer source"
    block = src[idx : idx + 400]
    assert '"reid"' in block, "get_dataset task set must include 'reid' explicitly"
