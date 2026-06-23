# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for the ReID task: Embeddings results, dataset routing, validator caching, and retrieval engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.engine.results import BaseTensor, Embeddings, Results

# ---------- Embeddings class ------------------------------------------------------------------


def test_embeddings_is_basetensor_subclass():
    """Embeddings must inherit BaseTensor so it composes with .cpu()/.numpy()/.cuda()/.to()."""
    assert isinstance(Embeddings(torch.randn(512)), BaseTensor)


def test_embeddings_dim_1d_and_2d():
    """Dim returns the LAST axis (the feature dim) for both unbatched and batched inputs."""
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


def test_embeddings_len_and_getitem():
    """__len__ returns 1 for (D,) and B for (B, D); slicing a 1-D embedding must not shrink to 0-D."""
    assert len(Embeddings(torch.randn(512))) == 1
    assert len(Embeddings(torch.randn(8, 256))) == 8
    e = Embeddings(torch.randn(512))
    sliced = e[0]
    assert sliced is e or sliced.data.shape == e.data.shape
    assert sliced.dim == 512
    row = Embeddings(torch.randn(4, 256))[1]
    assert row.data.shape == (256,)


def test_embeddings_unwraps_double_wrapped_input():
    """Embeddings(Embeddings(t)) must unwrap to a single-level wrapper (no copy)."""
    e1 = Embeddings(torch.randn(512))
    e2 = Embeddings(e1)
    assert isinstance(e2.data, torch.Tensor)
    assert e2.data is e1.data


# ---------- Results integration ---------------------------------------------------------------


@pytest.fixture
def fake_image() -> np.ndarray:
    """Return a blank uint8 image for constructing Results objects."""
    return np.zeros((100, 80, 3), dtype=np.uint8)


def test_results_accepts_embeddings_kwarg(fake_image):
    """Results must carry embeddings in the typed slot without hijacking probs."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    assert isinstance(r.embeddings, Embeddings)
    assert r.embeddings.dim == 512
    assert r.probs is None
    assert "embeddings" in r._keys  # so .cpu()/.numpy()/_apply walk it
    assert len(r) == 1  # one image, not the embedding dimensionality


def test_results_cpu_numpy_preserves_embeddings(fake_image):
    """Device-movement methods must preserve the embeddings attribute."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(128))
    assert isinstance(r.cpu().embeddings, Embeddings)
    assert isinstance(r.numpy().embeddings.data, np.ndarray)


def test_results_update_embeddings(fake_image):
    """Results.update(embeddings=...) must wrap tensors and unwrap existing Embeddings instances."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"})
    r.update(embeddings=torch.randn(256))
    assert r.embeddings.dim == 256
    r2 = Results(fake_image, path="/tmp/y.jpg", names={0: "a"})
    r2.update(embeddings=r.embeddings)  # pass the Embeddings instance, not the tensor
    assert isinstance(r2.embeddings.data, torch.Tensor)


def test_results_update_probs_wraps(fake_image):
    """Results.update(probs=...) must wrap raw tensors in Probs like __init__ does."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a", 1: "b"})
    r.update(probs=torch.tensor([0.3, 0.7]))
    assert r.probs.top1 == 1  # Probs API must work after update


def test_results_verbose_emits_embedding_line(fake_image):
    """verbose() must produce a ReID-shaped log line, not classify top-5."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(512))
    msg = r.verbose()
    assert "embedding" in msg and "512" in msg
    r0 = r[0]  # Results.__getitem__ must not break the 1-D embedding
    assert r0.embeddings.dim == 512
    assert "embedding" in r0.verbose()


def test_results_verbose_keeps_box_log_when_both_set(fake_image):
    """If a tracker attaches embeddings to a detection Results, verbose() must still show boxes."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]])
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "person"}, boxes=boxes, embeddings=torch.randn(128))
    assert "person" in r.verbose()


def test_results_save_crop_warns_reid_not_classify(fake_image, monkeypatch):
    """save_crop must warn with 'ReID', not the misleading 'Classify task' wording."""
    from ultralytics.engine import results as results_mod

    messages: list[str] = []
    monkeypatch.setattr(results_mod.LOGGER, "warning", lambda msg: messages.append(str(msg)))
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(8))
    with tempfile.TemporaryDirectory() as tmp:
        r.save_crop(tmp)
        assert list(Path(tmp).iterdir()) == []  # returned early, no crops written
    combined = " ".join(messages)
    assert "ReID" in combined and "Classify" not in combined


def test_results_summary_returns_embedding_dict(fake_image):
    """summary() for a ReID result must return [{'embedding': [...]}], not classify top-5."""
    out = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(16)).summary()
    assert len(out) == 1 and len(out[0]["embedding"]) == 16


def test_results_save_txt_opt_in_for_embeddings(fake_image):
    """save_txt persists the embedding only with save_conf=True (dumps are ~5 KB per frame)."""
    r = Results(fake_image, path="/tmp/x.jpg", names={0: "a"}, embeddings=torch.randn(128))
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "no_conf.txt"
        r.save_txt(out_path, save_conf=False)
        assert not out_path.exists()
        out_path = Path(tmp) / "with_conf.txt"
        r.save_txt(out_path, save_conf=True)
        assert len(out_path.read_text().split()) == 128


# ---------- Predictor / trainer wiring (no model required) ------------------------------------


def test_reid_predictor_postprocess_uses_embeddings_slot():
    """ReidPredictor.postprocess builds Results with embeddings= (not probs=)."""
    from ultralytics.models.yolo.reid.predict import ReidPredictor

    p = ReidPredictor.__new__(ReidPredictor)  # bypass __init__ (it tries to load a model)
    p.batch = [["/tmp/a.jpg", "/tmp/b.jpg"]]
    p.model = SimpleNamespace(names={0: "id0"})
    preds = torch.randn(2, 512)
    orig_imgs = [np.zeros((50, 30, 3), dtype=np.uint8) for _ in range(2)]
    results = p.postprocess(preds, img=torch.zeros(2, 3, 32, 32), orig_imgs=orig_imgs)
    assert len(results) == 2
    for r in results:
        assert r.embeddings is not None and r.embeddings.dim == 512
        assert r.probs is None


def test_reid_trainer_plot_training_samples_is_noop():
    """plot_training_samples must not render pid integers as class names."""
    from ultralytics.models.yolo.reid.train import ReidTrainer

    t = ReidTrainer.__new__(ReidTrainer)
    t.plot_training_samples(batch={"img": torch.zeros(2, 3, 4, 4)}, ni=0)  # must not raise


def test_build_yolo_dataset_routes_reid(monkeypatch):
    """With cfg.task='reid', build_yolo_dataset must dispatch to ReidDataset (not YOLODataset)."""
    from ultralytics.data import build as build_mod

    class FakeReidDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(build_mod, "ReidDataset", FakeReidDataset)
    cfg = SimpleNamespace(task="reid", imgsz=64, rect=False, cache=None, single_cls=False, classes=None, fraction=1.0)
    ds = build_mod.build_yolo_dataset(cfg, "/tmp/_unused", batch=2, data={}, mode="train")
    assert isinstance(ds, FakeReidDataset)
    assert ds.kwargs["root"] == "/tmp/_unused"
    assert ds.kwargs["augment"] is True
    assert ds.kwargs["prefix"] == "train"


# ---------- Validator gallery cache ------------------------------------------------------------


def _stub_validator(calls):
    """Return a ReidValidator with a counting _extract_gallery_features stub."""
    from ultralytics.models.yolo.reid.val import ReidValidator

    v = ReidValidator.__new__(ReidValidator)
    v.args = SimpleNamespace(task="reid", batch=4, workers=0, half=False, imgsz=64, reid_scales=None, reid_tta=False)
    v._feats, v._pids, v._camids = [], [], []
    v.metrics = SimpleNamespace(update_gallery=lambda *a, **kw: None)
    v.data = {"path": "/tmp", "gallery": "g"}

    def stub(self, path):
        calls[0] += 1
        return (
            np.zeros((4, 8), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            ["a", "b", "c", "d"],
        )

    v._extract_gallery_features = stub.__get__(v, ReidValidator)
    return v


def test_gallery_cache_hits_on_same_model():
    """Repeated standalone init_metrics on the same model must reuse cached gallery features."""
    calls = [0]
    v = _stub_validator(calls)
    v.training = False
    model_a = torch.nn.Linear(8, 8)
    model_a.names = {0: "id0"}
    model_b = torch.nn.Linear(8, 8)
    model_b.names = {0: "id0"}
    v.init_metrics(model_a)
    v.init_metrics(model_a)  # hit
    assert calls[0] == 1
    v.init_metrics(model_b)  # miss (different model)
    assert calls[0] == 2


def test_gallery_cache_bypassed_during_training_val():
    """In-train val passes the SAME EMA module with mutated weights — the cache must be bypassed.

    id(model) can never detect in-place weight updates, so serving cached gallery features would
    score epoch-N queries against epoch-1 gallery embeddings, corrupting in-train mAP/fitness.
    """
    calls = [0]
    v = _stub_validator(calls)
    model = torch.nn.Linear(8, 8)
    model.names = {0: "id0"}
    v.training = True  # set by BaseValidator.__call__ when trainer is not None
    v.init_metrics(model)
    v.init_metrics(model)
    assert calls[0] == 2, "in-train val must re-extract gallery features every epoch"
    v.training = False
    v.init_metrics(model)  # standalone re-val of unchanged object may reuse the cache
    assert calls[0] == 2


def test_update_metrics_tta_handles_half_batch_fp32_model():
    """In-train AMP: fp16 batches + fp32 EMA model — the TTA re-embed must carry its own autocast guard."""
    from ultralytics.models.yolo.reid.val import ReidValidator

    v = ReidValidator.__new__(ReidValidator)
    v._model = (
        torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        .float()
        .eval()
    )
    v.training = True
    v.device = torch.device("cpu")
    v.args = SimpleNamespace(half=True, reid_scales=None, reid_tta=True)  # TTA active -> _embed path
    v._feats, v._pids, v._camids, v._paths = [], [], [], []
    batch = {
        "img": torch.randn(2, 3, 16, 16).half(),
        "cls": torch.zeros(2, dtype=torch.long),
        "camid": [0, 1],
        "im_file": ["a.jpg", "b.jpg"],
    }
    v.update_metrics(preds=None, batch=batch)  # must not raise
    assert v._feats[0].dtype == torch.float32  # guarded path upcasts before accumulation


# ---------- Retrieval engine -------------------------------------------------------------------


def test_l2_normalize_unit_norm_and_zero_safe():
    """l2_normalize produces unit rows and never divides by zero."""
    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.isfinite(l2_normalize(np.zeros((1, 4), dtype=np.float32))).all()


def test_cosine_topk_orders_and_clamps():
    """cosine_topk ranks by similarity and clamps topk to the gallery size."""
    from ultralytics.models.yolo.reid.retrieval import cosine_topk, l2_normalize

    gallery = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    query = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    idx, scores = cosine_topk(query, gallery, topk=2)
    assert idx[0, 0] == 0 and idx[0, 1] == 1
    assert scores[0, 0] >= scores[0, 1]
    idx, scores = cosine_topk(query, gallery, topk=5)  # topk > N
    assert idx.shape[1] == 3 and scores.shape[1] == 3


def test_scan_gallery_recursive_and_empty(tmp_path):
    """scan_gallery finds images recursively, excludes non-images, raises on an empty dir."""
    from ultralytics.models.yolo.reid.retrieval import scan_gallery

    (tmp_path / "sub").mkdir()
    (tmp_path / "empty").mkdir()
    for name in ["a.jpg", "sub/b.png", "c.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert sorted(p.name for p in scan_gallery(tmp_path)) == ["a.jpg", "b.png"]
    with pytest.raises((FileNotFoundError, RuntimeError)):
        scan_gallery(tmp_path / "empty")


def _const_embedder(dim=4):
    """Return an embed_fn mapping each path to a deterministic vector by position."""

    def embed(paths):
        return np.stack([np.full(dim, float(i), dtype=np.float32) for i, _ in enumerate(paths)], axis=0)

    return embed


def test_build_gallery_cache_roundtrip(tmp_path):
    """build_gallery writes a cache, reuses it, and rebuilds when imgsz changes."""
    from ultralytics.models.yolo.reid import retrieval

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        (gdir / name).write_bytes(b"x")
    cache = tmp_path / "cache.pt"
    calls = {"n": 0}

    def counting_embed(paths):
        calls["n"] += 1
        return _const_embedder()(paths)

    p1, e1 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert cache.exists() and calls["n"] == 1
    assert np.allclose(np.linalg.norm(e1[1:], axis=1), 1.0, atol=1e-5)  # rows L2-normalized
    p2, e2 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert calls["n"] == 1  # cache hit, no re-embed
    assert [str(p) for p in p1] == [str(p) for p in p2] and np.allclose(e1, e2)
    retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=128)
    assert calls["n"] == 2  # stale cache (different imgsz) -> rebuild


def test_visualizer_rank_uses_engine(monkeypatch, tmp_path):
    """ReIDVisualizer.rank ranks generically (no Market PID logic) via the shared engine."""
    from ultralytics.solutions import reid_visualizer as rv

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["m1.jpg", "m2.jpg", "m3.jpg"]:
        (gdir / name).write_bytes(b"x")
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")

    viz = rv.ReIDVisualizer.__new__(rv.ReIDVisualizer)  # avoid constructing a real YOLO model
    viz.imgsz = 64
    viz.device = None
    viz.model = None
    vecs = {
        str(query): np.array([1.0, 0.0], dtype=np.float32),
        str(gdir / "m1.jpg"): np.array([1.0, 0.0], dtype=np.float32),  # identical -> top-1
        str(gdir / "m2.jpg"): np.array([0.2, 0.9], dtype=np.float32),
        str(gdir / "m3.jpg"): np.array([0.0, 1.0], dtype=np.float32),
    }
    monkeypatch.setattr(viz, "_embed_paths", lambda paths: np.stack([vecs[str(p)] for p in paths], axis=0))

    items = viz.rank(query, gdir, k=2)
    assert len(items) == 2
    assert Path(items[0].path).name == "m1.jpg"
    assert items[0].score >= items[1].score
    assert not hasattr(rv.ReIDVisualizer, "_pid_from_name")  # Market-specific helpers removed
