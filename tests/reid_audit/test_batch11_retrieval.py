"""Batch 11: gallery retrieval engine + Results.matches + predictor ranking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ---------- engine: cosine_topk / scan_gallery / l2_normalize ----------------


def test_l2_normalize_rows_unit_norm():
    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_zero_row_safe():
    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    x = np.zeros((1, 4), dtype=np.float32)
    out = l2_normalize(x)  # must not divide-by-zero -> NaN
    assert np.isfinite(out).all()


def test_cosine_topk_orders_by_similarity():
    from ultralytics.models.yolo.reid.retrieval import cosine_topk, l2_normalize

    gallery = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    query = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
    idx, scores = cosine_topk(query, gallery, topk=2)
    assert idx.shape == (1, 2) and scores.shape == (1, 2)
    assert idx[0, 0] == 0  # identical vector ranks first
    assert idx[0, 1] == 1  # near vector second
    assert scores[0, 0] >= scores[0, 1]


def test_cosine_topk_clamps_to_gallery_size():
    from ultralytics.models.yolo.reid.retrieval import cosine_topk, l2_normalize

    gallery = l2_normalize(np.random.rand(2, 8).astype(np.float32))
    query = l2_normalize(np.random.rand(1, 8).astype(np.float32))
    idx, scores = cosine_topk(query, gallery, topk=5)  # topk > N
    assert idx.shape[1] == 2 and scores.shape[1] == 2


def test_scan_gallery_finds_images_recursively(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import scan_gallery

    (tmp_path / "sub").mkdir()
    for name in ["a.jpg", "sub/b.png", "c.txt"]:
        (tmp_path / name).write_bytes(b"x")
    found = scan_gallery(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.jpg", "b.png"]  # .txt excluded


def test_scan_gallery_empty_raises(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import scan_gallery

    with pytest.raises((FileNotFoundError, RuntimeError)):
        scan_gallery(tmp_path)  # no images


# ---------- engine: cache + build_gallery -----------------------------------


def _const_embedder(dim=4):
    """Return an embed_fn that maps each path to a deterministic vector by filename order."""

    def embed(paths):
        return np.stack([np.full(dim, float(i), dtype=np.float32) for i, _ in enumerate(paths)], axis=0)

    return embed


def test_build_gallery_no_cache(tmp_path):
    from ultralytics.models.yolo.reid.retrieval import build_gallery

    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (tmp_path / name).write_bytes(b"x")
    paths, embs = build_gallery(_const_embedder(), tmp_path, cache=None, model_id="m", imgsz=64)
    assert len(paths) == 3
    assert embs.shape == (3, 4)
    # rows are L2-normalized
    assert np.allclose(np.linalg.norm(embs[1:], axis=1), 1.0, atol=1e-5)


def test_build_gallery_writes_and_reuses_cache(tmp_path):
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

    # First call builds + writes cache
    p1, e1 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert cache.exists()
    assert calls["n"] == 1
    # Second call loads from cache, does NOT re-embed
    p2, e2 = retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert calls["n"] == 1
    assert [str(p) for p in p1] == [str(p) for p in p2]
    assert np.allclose(e1, e2)


def test_build_gallery_rebuilds_on_stale_cache(tmp_path):
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

    retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=64)
    assert calls["n"] == 1
    # Different imgsz -> stale -> rebuild
    retrieval.build_gallery(counting_embed, gdir, cache=cache, model_id="m", imgsz=128)
    assert calls["n"] == 2


# ---------- Results.matches --------------------------------------------------


@pytest.fixture
def fake_image() -> np.ndarray:
    return np.zeros((40, 30, 3), dtype=np.uint8)


def test_results_accepts_matches_kwarg(fake_image):
    from ultralytics.engine.results import Results

    matches = [("/g/a.jpg", 0.93), ("/g/b.jpg", 0.81)]
    r = Results(fake_image, path="/q.jpg", names={0: "id"}, embeddings=torch.randn(8), matches=matches)
    assert r.matches == matches


def test_results_matches_defaults_none(fake_image):
    from ultralytics.engine.results import Results

    r = Results(fake_image, path="/q.jpg", names={0: "id"}, embeddings=torch.randn(8))
    assert r.matches is None


def test_results_matches_not_in_keys(fake_image):
    """matches is a plain list (paths+scores), must NOT be walked by _apply/.cpu()/.numpy()."""
    from ultralytics.engine.results import Results

    r = Results(fake_image, path="/q.jpg", names={0: "id"}, matches=[("/g/a.jpg", 0.5)])
    assert "matches" not in r._keys


def test_results_verbose_emits_matches_line(fake_image):
    from ultralytics.engine.results import Results

    matches = [("/g/a.jpg", 0.93), ("/g/b.jpg", 0.81)]
    r = Results(fake_image, path="/q.jpg", names={0: "id"}, embeddings=torch.randn(8), matches=matches)
    msg = r.verbose()
    assert "a.jpg" in msg and "0.93" in msg  # ranked match line, not the embedding line


def test_results_new_preserves_matches(fake_image):
    from ultralytics.engine.results import Results

    r = Results(fake_image, path="/q.jpg", names={0: "id"}, matches=[("/g/a.jpg", 0.5)])
    assert r.new().matches == [("/g/a.jpg", 0.5)]


# ---------- CLI custom keys --------------------------------------------------


def test_reid_custom_keys_registered():
    from ultralytics.cfg import TASK_CUSTOM_KEYS

    keys = TASK_CUSTOM_KEYS["reid"]
    assert {"gallery", "topk", "reid_cache"} <= keys


def test_cfg_accepts_gallery_args():
    """get_cfg must not reject gallery/topk/reid_cache for the reid task."""
    from ultralytics.cfg import get_cfg

    cfg = get_cfg(overrides={"task": "reid", "gallery": "g/", "topk": 5, "reid_cache": "c.pt"})
    assert cfg.gallery == "g/"
    assert cfg.topk == 5
    assert cfg.reid_cache == "c.pt"


# ---------- ReidPredictor ranking -------------------------------------------


def _make_predictor_with_gallery(monkeypatch, gallery_embs, gallery_paths, topk=2):
    """Build a ReidPredictor without loading a model, with a stubbed gallery index."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.reid.predict import ReidPredictor

    p = ReidPredictor.__new__(ReidPredictor)
    p.args = SimpleNamespace(gallery="g/", topk=topk, reid_cache=None, save=False, model="m.pt")
    p.batch = [["/q/q0.jpg"]]
    p.model = SimpleNamespace(names={0: "id"})
    p.gallery_paths = gallery_paths
    p.gallery_embs = gallery_embs
    p.save_dir = None
    return p


def test_predictor_attaches_matches(monkeypatch):
    import numpy as np

    from ultralytics.models.yolo.reid.retrieval import l2_normalize

    gallery_embs = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    gallery_paths = [Path("/g/match.jpg"), Path("/g/other.jpg")]
    p = _make_predictor_with_gallery(monkeypatch, gallery_embs, gallery_paths, topk=2)

    # query identical to gallery row 0
    preds = torch.tensor([[1.0, 0.0]])
    orig_imgs = [np.zeros((10, 10, 3), dtype=np.uint8)]
    results = p.postprocess(preds, img=torch.zeros(1, 3, 8, 8), orig_imgs=orig_imgs)

    assert len(results) == 1
    matches = results[0].matches
    assert matches is not None and len(matches) == 2
    assert Path(matches[0][0]).name == "match.jpg"  # top-1 is the identical vector
    assert matches[0][1] >= matches[1][1]  # descending score
    assert results[0].embeddings is not None  # embeddings still present


def test_predictor_no_gallery_is_embeddings_only(monkeypatch):
    from types import SimpleNamespace

    from ultralytics.models.yolo.reid.predict import ReidPredictor

    p = ReidPredictor.__new__(ReidPredictor)
    p.args = SimpleNamespace(gallery=None)
    p.batch = [["/q/q0.jpg"]]
    p.model = SimpleNamespace(names={0: "id"})
    p.gallery_paths = None
    p.gallery_embs = None
    preds = torch.randn(1, 8)
    results = p.postprocess(preds, img=torch.zeros(1, 3, 8, 8), orig_imgs=[np.zeros((10, 10, 3), dtype=np.uint8)])
    assert results[0].matches is None
    assert results[0].embeddings is not None


# ---------- montage save -----------------------------------------------------


def test_write_results_saves_montage(monkeypatch, tmp_path):
    """With a gallery + save=True, write_results writes one montage per query and returns a log line."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.reid import predict as predict_mod
    from ultralytics.engine.results import Results

    saved = {}

    def fake_plot(rows, save_path, **kw):
        Path(save_path).write_bytes(b"montage")
        saved["rows"] = rows
        saved["path"] = Path(save_path)
        return Path(save_path)

    monkeypatch.setattr(predict_mod, "plot_reid_retrieval", fake_plot)

    p = predict_mod.ReidPredictor.__new__(predict_mod.ReidPredictor)
    p.args = SimpleNamespace(gallery="g/", save=True)
    p.save_dir = tmp_path
    p.source_type = SimpleNamespace(stream=False, from_img=False, tensor=False)
    p.dataset = SimpleNamespace(count=0, mode="image")
    img = np.zeros((40, 30, 3), dtype=np.uint8)
    res = Results(img, path="/q/q0.jpg", names={0: "id"}, embeddings=torch.randn(8),
                  matches=[("/g/a.jpg", 0.93), ("/g/b.jpg", 0.81)])
    res.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0}
    p.results = [res]

    s = [""]
    out = p.write_results(0, Path("/q/q0.jpg"), torch.zeros(1, 3, 8, 8), s)
    assert saved["path"].exists()  # montage written
    assert "q0" in saved["path"].name
    assert "a.jpg" in out and "0.93" in out  # ranked log line
    # query tile + 2 match tiles in the single row
    assert len(saved["rows"]) == 1 and len(saved["rows"][0]) == 3


# ---------- ReIDVisualizer reuses the engine --------------------------------


def test_visualizer_rank_uses_engine(monkeypatch, tmp_path):
    """ReIDVisualizer.rank ranks generically (no Market PID logic) via the shared engine."""
    from ultralytics.solutions import reid_visualizer as rv

    gdir = tmp_path / "g"
    gdir.mkdir()
    for name in ["m1.jpg", "m2.jpg", "m3.jpg"]:
        (gdir / name).write_bytes(b"x")
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")

    # Avoid constructing a real YOLO model
    viz = rv.ReIDVisualizer.__new__(rv.ReIDVisualizer)
    viz.imgsz = 64
    viz.device = None
    viz.model = None

    vecs = {
        str(query): np.array([1.0, 0.0], dtype=np.float32),
        str(gdir / "m1.jpg"): np.array([1.0, 0.0], dtype=np.float32),  # identical → top-1
        str(gdir / "m2.jpg"): np.array([0.2, 0.9], dtype=np.float32),
        str(gdir / "m3.jpg"): np.array([0.0, 1.0], dtype=np.float32),
    }
    monkeypatch.setattr(viz, "_embed_paths", lambda paths: np.stack([vecs[str(p)] for p in paths], axis=0))

    items = viz.rank(query, gdir, k=2)
    assert len(items) == 2
    assert Path(items[0].path).name == "m1.jpg"
    assert items[0].score >= items[1].score


def test_visualizer_has_no_market_pid_helpers():
    """The Market-1501 filename parsing helpers are removed from the CLI-facing visualizer."""
    from ultralytics.solutions.reid_visualizer import ReIDVisualizer

    assert not hasattr(ReIDVisualizer, "_pid_from_name")
    assert not hasattr(ReIDVisualizer, "_cam_from_name")


# ---------- end-to-end CLI (downloads a tiny published weight) ---------------


@pytest.mark.slow
def test_e2e_predict_gallery_retrieval(tmp_path):
    """Full path: YOLO('yolo26n-reid.pt').predict(source, gallery, topk) -> matches + montage.

    Skips if the published reid weight cannot be fetched (offline CI).
    """
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    try:
        model = YOLO("yolo26n-reid.pt", task="reid")
    except Exception as e:  # offline / asset missing
        pytest.skip(f"reid weight unavailable: {e}")

    # Build a tiny gallery from bundled assets
    gdir = tmp_path / "gallery"
    gdir.mkdir()
    import shutil

    imgs = sorted(Path(ASSETS).glob("*.jpg"))[:3]
    if not imgs:
        pytest.skip("no bundled assets to build a gallery")
    for k, src in enumerate(imgs):
        shutil.copy(src, gdir / f"g{k}.jpg")

    results = model.predict(
        source=str(imgs[0]), gallery=str(gdir), topk=2, imgsz=64,
        project=str(tmp_path), name="run", save=True, verbose=False,
    )
    r = results[0]
    assert r.matches is not None and len(r.matches) == 2
    assert all(isinstance(p, str) and isinstance(s, float) for p, s in r.matches)
    # montage written under the run dir
    montages = list(Path(tmp_path / "run").glob("*_top2.jpg"))
    assert montages, "expected a query->top-2 montage to be saved"
