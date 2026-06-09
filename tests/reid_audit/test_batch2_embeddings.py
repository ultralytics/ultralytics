"""Batch 2: typed Embeddings field on Results.

What would have failed before the fix:
  * `Results(embeddings=t)` would raise TypeError (no such kwarg).
  * `result.embeddings` would be `None` (predictor stuffed embeddings into `.probs`).
  * `Results.verbose()` would return classification top-5 nonsense or empty string.
  * `Results.save_crop()` would warn "Classify task does not support save_crop" — misleading.
  * `Results.summary()` would return [] for a 0-D 'classification' input.

All assertions below would have failed in that state.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.engine.results import BaseTensor, Embeddings, Results


# ---------- Embeddings class -----------------------------------------------------------------


def test_embeddings_is_basetensor_subclass():
    """Embeddings must inherit BaseTensor so it composes with .cpu()/.numpy()/.cuda()/.to()."""
    e = Embeddings(torch.randn(512))
    assert isinstance(e, BaseTensor)


def test_embeddings_dim_1d_and_2d():
    """dim returns the LAST axis (the feature dim) for both unbatched and batched inputs."""
    assert Embeddings(torch.randn(512)).dim == 512
    assert Embeddings(torch.randn(8, 256)).dim == 256


def test_embeddings_roundtrip_cpu_numpy_to():
    """Movement methods preserve the Embeddings class and dim."""
    e = Embeddings(torch.randn(4, 128))
    assert isinstance(e.cpu(), Embeddings)
    assert e.cpu().dim == 128
    assert isinstance(e.numpy(), Embeddings)
    assert isinstance(e.numpy().data, np.ndarray)
    e2 = e.to(dtype=torch.float64)
    assert isinstance(e2, Embeddings)
    assert e2.data.dtype == torch.float64


# ---------- Results integration --------------------------------------------------------------


@pytest.fixture
def fake_image() -> np.ndarray:
    return np.zeros((100, 80, 3), dtype=np.uint8)


def test_results_accepts_embeddings_kwarg(fake_image):
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    assert r.embeddings is not None
    assert isinstance(r.embeddings, Embeddings)
    assert r.embeddings.dim == 512
    # critical: must NOT have hijacked the probs slot
    assert r.probs is None


def test_results_embeddings_in_keys(fake_image):
    """_keys must include 'embeddings' so .cpu()/.numpy()/_apply walks it."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(64))
    assert "embeddings" in r._keys


def test_results_cpu_numpy_preserves_embeddings(fake_image):
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(128))
    r_cpu = r.cpu()
    assert r_cpu.embeddings is not None
    assert isinstance(r_cpu.embeddings, Embeddings)
    r_np = r.numpy()
    assert r_np.embeddings is not None
    assert isinstance(r_np.embeddings.data, np.ndarray)


def test_results_update_embeddings(fake_image):
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"})
    assert r.embeddings is None
    r.update(embeddings=torch.randn(256))
    assert r.embeddings is not None
    assert r.embeddings.dim == 256


def test_results_verbose_emits_embedding_line(fake_image):
    """verbose() must produce a ReID-shaped log line, not classify top-5."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    msg = r.verbose()
    assert "embedding" in msg
    assert "512" in msg


def test_results_save_crop_warns_reid_not_classify(fake_image, monkeypatch):
    """save_crop must say 'ReID', not the misleading 'Classify task' warning."""
    from ultralytics.engine import results as results_mod

    messages: list[str] = []
    monkeypatch.setattr(results_mod.LOGGER, "warning", lambda msg: messages.append(str(msg)))
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(8))
    with tempfile.TemporaryDirectory() as tmp:
        r.save_crop(tmp)
        # Confirm the function returned early without writing any crop subdirs
        assert list(Path(tmp).iterdir()) == []
    combined = " ".join(messages)
    assert "ReID" in combined, f"save_crop must warn with 'ReID' (got: {messages!r})"
    assert "Classify" not in combined, "save_crop must NOT use the misleading 'Classify' wording"


def test_results_summary_returns_embedding_dict(fake_image):
    """summary() for a ReID result must return [{'embedding': [...]}], NOT classify top-5."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(16))
    out = r.summary()
    assert len(out) == 1
    assert "embedding" in out[0]
    assert len(out[0]["embedding"]) == 16


def test_results_save_txt_writes_embedding_vector(fake_image):
    """save_txt for ReID, when opt-in via save_conf=True, persists the embedding row(s) as
    space-separated floats. (Embedding dumps are large — see test_results_save_txt_opt_in_for_embeddings
    for the gating contract.)"""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(32))
    with tempfile.TemporaryDirectory() as tmp:
        out = r.save_txt(Path(tmp) / "out.txt", save_conf=True)
        line = Path(out).read_text().splitlines()[0]
        assert len(line.split()) == 32


# ---------- Predictor wiring (no model required) --------------------------------------------


def test_reid_predictor_postprocess_uses_embeddings_slot():
    """ReidPredictor.postprocess builds Results with embeddings= (not probs=).

    We mock the predictor minimally: instantiate it, monkeypatch self.batch + self.model.names,
    feed a fake preds tensor, and confirm the resulting Results has embeddings (not probs).
    """
    from ultralytics.models.yolo.reid.predict import ReidPredictor
    from types import SimpleNamespace

    p = ReidPredictor.__new__(ReidPredictor)  # bypass __init__ (it tries to load a model)
    p.batch = [["/tmp/a.jpg", "/tmp/b.jpg"]]
    p.model = SimpleNamespace(names={0: "id0"})
    preds = torch.randn(2, 512)
    orig_imgs = [np.zeros((50, 30, 3), dtype=np.uint8) for _ in range(2)]
    results = p.postprocess(preds, img=torch.zeros(2, 3, 32, 32), orig_imgs=orig_imgs)
    assert len(results) == 2
    for r in results:
        assert r.embeddings is not None, "ReID predictor must populate the typed embeddings slot"
        assert r.probs is None, "ReID predictor must NOT reuse the classification probs slot"
        assert r.embeddings.dim == 512


# ---------- Trainer no-op plot ---------------------------------------------------------------


def test_results_len_returns_one_for_single_embedding(fake_image):
    """Results.__len__ for a ReID-only result must return 1 (one image), NOT the embedding
    dimensionality D. Previously len(result) on a (D,) embedding returned D=512 because
    BaseTensor.__len__ inherits from torch.Tensor's first-axis size."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    assert len(r) == 1, f"expected 1 (one image), got {len(r)}"


def test_embeddings_len_for_1d_and_2d():
    """Embeddings.__len__ returns 1 for a (D,) input (single image) and B for (B, D)."""
    assert len(Embeddings(torch.randn(512))) == 1
    assert len(Embeddings(torch.randn(8, 256))) == 8


def test_embeddings_getitem_1d_returns_self():
    """Slicing a (D,) embedding must NOT shrink to a 0-D scalar — slicing the feature axis
    is meaningless. Return self unchanged for 1-D inputs."""
    e = Embeddings(torch.randn(512))
    sliced = e[0]
    assert sliced is e or sliced.data.shape == e.data.shape
    # critical: the dim property must still work after the slice (was IndexErroring)
    assert sliced.dim == 512


def test_embeddings_getitem_2d_slices_batch():
    """For (B, D) input, slicing should pick one row (→ (D,))."""
    e = Embeddings(torch.randn(4, 256))
    row = e[1]
    assert row.data.shape == (256,)
    assert row.dim == 256


def test_results_getitem_does_not_break_embeddings(fake_image):
    """Regression: Results[i] propagates to Embeddings.__getitem__ which used to slice the
    feature axis on a 1-D embedding, producing a 0-D scalar whose .dim raised IndexError."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    r0 = r[0]  # must not crash
    assert r0.embeddings is not None
    assert r0.embeddings.dim == 512  # dim still 512 after slicing
    # verbose() must not crash either
    assert "embedding" in r0.verbose()


def test_results_verbose_keeps_box_log_when_both_set(fake_image):
    """If a tracker attaches embeddings to a detection Results, verbose() must still show
    the per-class detection counts — embedding line is for embedding-only Results."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]])
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "person"}, boxes=boxes, embeddings=torch.randn(128))
    msg = r.verbose()
    assert "person" in msg, f"verbose() must include the box count line; got {msg!r}"


def test_results_save_txt_opt_in_for_embeddings(fake_image):
    """save_txt for a ReID Results must require save_conf=True to write the embedding
    vector — otherwise the labels/ dir grows by ~5 KB per frame silently."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(128))
    # Without save_conf: no file written
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "no_conf.txt"
        r.save_txt(out_path, save_conf=False)
        assert not out_path.exists(), "save_txt with save_conf=False must not write embedding files"
    # With save_conf: file is written
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "with_conf.txt"
        r.save_txt(out_path, save_conf=True)
        assert out_path.exists()
        assert len(out_path.read_text().split()) == 128


def test_embeddings_unwraps_double_wrapped_input():
    """Embeddings(Embeddings(t)) must unwrap to a single-level Embeddings wrapping t —
    not produce a wrapper whose .data is another Embeddings. Same for Results.update
    and Results.__init__ paths that re-wrap an already-Embeddings input."""
    e1 = Embeddings(torch.randn(512))
    e2 = Embeddings(e1)
    assert isinstance(e2.data, torch.Tensor), f"expected Tensor, got {type(e2.data).__name__}"
    assert e2.data is e1.data  # no copy


def test_results_update_unwraps_embeddings_instance(fake_image):
    """Results.update(embeddings=existing_embeddings_obj) must not double-wrap; .data
    stays a tensor so subsequent .cpu()/.numpy() continue to work."""
    r1 = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    r2 = Results(fake_image, path="/tmp/y.jpg", names={0: "a"})
    r2.update(embeddings=r1.embeddings)  # pass the Embeddings instance, not the tensor
    assert isinstance(r2.embeddings.data, torch.Tensor)
    assert r2.embeddings.dim == 512


def test_reid_trainer_plot_training_samples_is_noop():
    """plot_training_samples must NOT render pid integers as class names."""
    from ultralytics.models.yolo.reid.train import ReidTrainer

    t = ReidTrainer.__new__(ReidTrainer)
    # Should run without touching plot_images (i.e. not raise even with a malformed batch).
    t.plot_training_samples(batch={"img": torch.zeros(2, 3, 4, 4)}, ni=0)
