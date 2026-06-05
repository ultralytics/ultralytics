# ReID Predict Gallery Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `yolo reid predict source=query gallery=folder topk=5` rank a gallery by similarity and return the top-N matches as a saved montage, a console ranked list, and structured `Results.matches`.

**Architecture:** A new model-agnostic retrieval engine (`reid/retrieval.py`) owns gallery scanning, embedding-cache I/O, and exact-cosine top-k. `ReidPredictor` builds the gallery embedding index once (lazily, with optional cache), then ranks each streamed query against it in `postprocess`, attaches matches to `Results`, and saves a montage in `write_results`. A new plain `Results.matches` attribute + a `verbose()` branch surface the ranked list. The existing `ReIDVisualizer` solution is refactored to call the shared engine (batched, generic — Market PID parsing dropped). New CLI args (`gallery`, `topk`, `reid_cache`) are registered as reid custom keys.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, pytest. Ultralytics predictor/Results framework.

---

## Spec reference

Design: `docs/superpowers/specs/2026-06-05-reid-predict-gallery-retrieval-design.md`

## File Structure

- **Create** `ultralytics/models/yolo/reid/retrieval.py` — model-agnostic engine: `scan_gallery`, `l2_normalize`, `cosine_topk`, gallery cache load/save/validate, `build_gallery`.
- **Modify** `ultralytics/engine/results.py` — add `matches` attribute to `Results.__init__`/`new()`, add a `matches` branch to `verbose()`.
- **Modify** `ultralytics/cfg/__init__.py` — add `gallery`, `topk`, `reid_cache` to `TASK_CUSTOM_KEYS["reid"]`.
- **Modify** `ultralytics/models/yolo/reid/predict.py` — gallery index build, `_embed_paths`, ranking in `postprocess`, montage save via `write_results` override.
- **Modify** `ultralytics/solutions/reid_visualizer.py` — refactor onto the shared engine, batched embedding + cache, drop Market PID parsing from the CLI-facing path.
- **Modify** `tests/reid_audit/test_batch2_embeddings.py` — update the `write_results` guard test to allow the gallery-aware override.
- **Create** `tests/reid_audit/test_batch11_retrieval.py` — unit tests for the engine, `Results.matches`, predictor ranking, visualizer.

## Conventions to follow

- ReID custom CLI knobs are **not** in `default.yaml`; they live only in `TASK_CUSTOM_KEYS["reid"]` and are read via `getattr(self.args, "<key>", <default>)` (mirror `reid_tta`/`reid_reranking` in `ultralytics/models/yolo/reid/val.py`).
- `Results` typed tensor fields go in `self._keys`; `matches` is a plain list of `(path:str, score:float)` and is **deliberately not** in `_keys` (it is not a tensor and must not be walked by `_apply`/`.cpu()`/`.numpy()`).
- Run the full reid-audit suite with: `pytest tests/reid_audit/ -q` (use the venv interpreter `/home/rick/ultralytics/.venv/bin/python -m pytest ...` — system `python` lacks torch).
- `tests/reid_audit/` is matched by `.gitignore`; commit test files with `git add -f tests/reid_audit/<file>` (the plain `git add` in the steps below will otherwise silently skip them).

---

### Task 1: Retrieval engine — `scan_gallery`, `l2_normalize`, `cosine_topk`

**Files:**
- Create: `ultralytics/models/yolo/reid/retrieval.py`
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -q`
Expected: FAIL with `ModuleNotFoundError: ultralytics.models.yolo.reid.retrieval`.

- [ ] **Step 3: Write the engine functions**

Create `ultralytics/models/yolo/reid/retrieval.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model-agnostic gallery retrieval helpers for ReID predict.

Pure NumPy ranking + cache I/O, decoupled from any model. Callers supply an
``embed_fn(list[Path]) -> np.ndarray (N, D)`` so the same engine serves both the
``ReidPredictor`` CLI path and the ``ReIDVisualizer`` solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER


def scan_gallery(root: str | Path) -> list[Path]:
    """Recursively collect image paths under a directory (sorted)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"gallery '{root}' does not exist")
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS]
    if not paths:
        raise RuntimeError(f"no image files found under gallery '{root}'")
    return paths


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, safe for zero rows."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cosine_topk(query_embs: np.ndarray, gallery_embs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the top-k gallery rows per query by cosine similarity.

    Inputs are assumed L2-normalized, so cosine == dot product. ``topk`` is clamped to
    the gallery size. Shapes: query (Q, D), gallery (N, D) -> indices (Q, k), scores (Q, k).
    """
    k = min(topk, gallery_embs.shape[0])
    sims = query_embs @ gallery_embs.T  # (Q, N)
    idx = np.argsort(-sims, axis=1)[:, :k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return idx, scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/models/yolo/reid/retrieval.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "feat(reid): retrieval engine scan/normalize/cosine-topk"
```

---

### Task 2: Retrieval engine — embedding cache + `build_gallery`

**Files:**
- Modify: `ultralytics/models/yolo/reid/retrieval.py`
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k build_gallery -q`
Expected: FAIL with `AttributeError: ... has no attribute 'build_gallery'`.

- [ ] **Step 3: Implement cache + build_gallery**

Append to `ultralytics/models/yolo/reid/retrieval.py`:

```python
def _signature(paths: list[Path], model_id: str, imgsz) -> dict:
    """Cache key: gallery file list (as strings) + model id + imgsz."""
    return {"paths": [str(p) for p in paths], "model_id": str(model_id), "imgsz": list(np.atleast_1d(imgsz))}


def build_gallery(
    embed_fn: Callable[[list[Path]], np.ndarray],
    gallery: str | Path,
    cache: str | Path | None,
    model_id: str,
    imgsz,
) -> tuple[list[Path], np.ndarray]:
    """Scan the gallery, embed (or load from cache), and return (paths, L2-normalized embeddings).

    The cache (a ``.pt`` file) is reused only when its recorded gallery file list, model id, and
    imgsz match the current request; otherwise it is rebuilt and rewritten.
    """
    import torch

    paths = scan_gallery(gallery)
    sig = _signature(paths, model_id, imgsz)

    if cache is not None and Path(cache).exists():
        blob = torch.load(str(cache), weights_only=False)
        if blob.get("signature") == sig:
            return paths, np.asarray(blob["embs"], dtype=np.float32)
        LOGGER.warning(f"reid_cache '{cache}' is stale (model/imgsz/gallery changed); rebuilding.")

    embs = l2_normalize(np.asarray(embed_fn(paths), dtype=np.float32))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"signature": sig, "embs": embs}, str(cache))
    return paths, embs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -q`
Expected: all passed (9 total so far).

- [ ] **Step 5: Commit**

```bash
git add ultralytics/models/yolo/reid/retrieval.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "feat(reid): gallery embedding cache + build_gallery"
```

---

### Task 3: `Results.matches` field + `verbose()` line

**Files:**
- Modify: `ultralytics/engine/results.py:224-270` (`__init__`), `:437-447` (`new`), `:637-668` (`verbose`)
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k matches -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'matches'`.

- [ ] **Step 3: Implement the Results changes**

In `ultralytics/engine/results.py`, add the `matches` parameter to `__init__` (after `embeddings`):

Change the signature line:
```python
        embeddings: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
```
to:
```python
        embeddings: torch.Tensor | None = None,
        matches: list | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
```

Add the attribute assignment right after the `self.embeddings = ...` line (line ~265). Note it is set **after** `_keys` is acceptable, but place it next to embeddings for clarity and do NOT add it to `_keys`:
```python
        self.embeddings = Embeddings(embeddings) if embeddings is not None else None
        self.matches = matches  # ReID retrieval: list[(gallery_path, score)] or None; not a tensor, not in _keys
```

Update `new()` (line ~447) to preserve matches:
```python
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed, matches=self.matches)
```

Add a `matches` branch at the TOP of `verbose()` (before the embeddings short-circuit at line ~663), so a retrieval result logs its ranked list:
```python
        if self.matches is not None:
            from pathlib import Path

            return "".join(f"#{r + 1} {Path(p).name} {s:.4f}, " for r, (p, s) in enumerate(self.matches))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k "matches or verbose" -q`
Expected: all passed.

- [ ] **Step 5: Run the existing Results/embeddings tests to confirm no regression**

Run: `pytest tests/reid_audit/test_batch2_embeddings.py -q`
Expected: all passed (the embedding `verbose()` line still works because `matches` defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add ultralytics/engine/results.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "feat(reid): add Results.matches field + verbose ranked line"
```

---

### Task 4: Register `gallery`, `topk`, `reid_cache` CLI custom keys

**Files:**
- Modify: `ultralytics/cfg/__init__.py:92-101` (`TASK_CUSTOM_KEYS`)
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k "custom_keys or cfg_accepts" -q`
Expected: FAIL — `gallery`/`topk`/`reid_cache` rejected by `check_dict_alignment`.

- [ ] **Step 3: Add the keys**

In `ultralytics/cfg/__init__.py`, edit the `TASK_CUSTOM_KEYS["reid"]` set (after the `dg_*` line) to add the retrieval keys:
```python
        "dg_mixstyle", "dg_dann", "dann_gamma", "dg_mixstyle_layers",
        "gallery", "topk", "reid_cache",  # predict-time gallery retrieval knobs
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k "custom_keys or cfg_accepts" -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/cfg/__init__.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "feat(reid): register gallery/topk/reid_cache CLI keys"
```

---

### Task 5: `ReidPredictor` gallery embedding + ranking in `postprocess`

**Files:**
- Modify: `ultralytics/models/yolo/reid/predict.py`
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k predictor -q`
Expected: FAIL — `postprocess` does not attach matches / `gallery_embs` attribute unknown.

- [ ] **Step 3: Implement predictor `__init__`, `_embed_paths`, and gallery-aware `postprocess`**

Rewrite `ultralytics/models/yolo/reid/predict.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.reid import retrieval
from ultralytics.utils import DEFAULT_CFG, ops


class ReidPredictor(ClassificationPredictor):
    """Predictor for person re-identification models.

    Default behavior wraps each image's L2-normalized embedding in ``Results.embeddings``.
    When a ``gallery`` argument is supplied, the predictor instead performs retrieval: it embeds
    the gallery once (optionally cached via ``reid_cache``), ranks each streamed query against it
    by cosine similarity, and attaches the top-``topk`` ``(path, score)`` matches to
    ``Results.matches`` (a montage is saved in ``write_results``).

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="query.jpg", gallery="gallery/", topk=5)
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor, re-set task to 'reid', and clear the lazy gallery index."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"
        self.gallery_paths = None
        self.gallery_embs = None

    def _embed_paths(self, paths: list) -> np.ndarray:
        """Embed a list of image paths in batches using this predictor's model and transforms.

        Returns a (N, D) float32 array (un-normalized; ``build_gallery`` L2-normalizes).
        """
        bs = max(int(getattr(self.args, "batch", 16) or 16), 1)
        embs: list[np.ndarray] = []
        for start in range(0, len(paths), bs):
            chunk = paths[start : start + bs]
            ims = [cv2.imread(str(p)) for p in chunk]
            im = self.preprocess(ims)
            with torch.no_grad():
                preds = self.model(im)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            embs.append(preds.detach().cpu().float().numpy())
        return np.concatenate(embs, axis=0)

    def _ensure_gallery(self) -> None:
        """Build the gallery embedding index once (lazy), honoring the optional cache."""
        if self.gallery_embs is None:
            self.gallery_paths, self.gallery_embs = retrieval.build_gallery(
                self._embed_paths,
                getattr(self.args, "gallery"),
                cache=getattr(self.args, "reid_cache", None),
                model_id=str(getattr(self.args, "model", "")),
                imgsz=self.imgsz,
            )

    def postprocess(self, preds, img, orig_imgs):
        """Wrap embeddings in Results; when a gallery is set, also attach ranked matches.

        Args:
            preds (torch.Tensor | tuple): (B, D) embeddings or an ``(embedding, feat_bn)`` tuple.
            img (torch.Tensor): Preprocessed input batch.
            orig_imgs (list[np.ndarray] | torch.Tensor): Original images.

        Returns:
            (list[Results]): One Results per query, each with ``embeddings`` and (when a gallery
                is supplied) ``matches`` populated.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        matches_per_query = [None] * len(preds)
        if getattr(self.args, "gallery", None):
            self._ensure_gallery()
            query = retrieval.l2_normalize(preds.detach().cpu().float().numpy())
            topk = int(getattr(self.args, "topk", 5) or 5)
            idx, scores = retrieval.cosine_topk(query, self.gallery_embs, topk)
            matches_per_query = [
                [(str(self.gallery_paths[j]), float(s)) for j, s in zip(idx[q], scores[q])]
                for q in range(len(preds))
            ]

        return [
            Results(orig_img, path=img_path, names=self.model.names, embeddings=pred, matches=matches)
            for pred, orig_img, img_path, matches in zip(preds, orig_imgs, self.batch[0], matches_per_query)
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k predictor -q`
Expected: 2 passed.

- [ ] **Step 5: Confirm no regression in the existing predictor embedding test**

Run: `pytest tests/reid_audit/test_batch2_embeddings.py::test_reid_predictor_postprocess_uses_embeddings_slot -q`
Expected: passed (no-gallery path uses `getattr(self.args, "gallery", None)`; the test's `SimpleNamespace` has no `gallery`, so it returns `None` → embeddings-only).

- [ ] **Step 6: Commit**

```bash
git add ultralytics/models/yolo/reid/predict.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "feat(reid): gallery retrieval ranking in ReidPredictor.postprocess"
```

---

### Task 6: Montage save via gallery-aware `write_results` override

**Files:**
- Modify: `ultralytics/models/yolo/reid/predict.py` (add `write_results`)
- Modify: `tests/reid_audit/test_batch2_embeddings.py:163-170` (update the guard test)
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Update the existing guard test**

The old test asserted `ReidPredictor.write_results is BasePredictor.write_results` (it forbade a *clone*). We now add a thin, gallery-aware override that delegates to `super()` for the non-gallery path. Replace the test body in `tests/reid_audit/test_batch2_embeddings.py`:

```python
def test_reid_predictor_write_results_delegates_when_no_gallery():
    """ReidPredictor.write_results must NOT be a clone: for the non-gallery path it delegates
    to BasePredictor.write_results, only adding montage behavior for gallery retrieval."""
    import inspect

    from ultralytics.models.yolo.reid.predict import ReidPredictor

    src = inspect.getsource(ReidPredictor.write_results)
    assert "super().write_results" in src, "non-gallery path must delegate to BasePredictor (no clone)"
```

- [ ] **Step 2: Write the failing montage test**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k montage tests/reid_audit/test_batch2_embeddings.py -k write_results -q`
Expected: montage test FAILS (no `write_results` override / `plot_reid_retrieval` not imported in module); the updated guard test FAILS (`super().write_results` not yet in source).

- [ ] **Step 4: Implement the `write_results` override**

In `ultralytics/models/yolo/reid/predict.py`, add the import near the top:
```python
from ultralytics.utils.plotting import plot_reid_retrieval
```

Add this method to `ReidPredictor` (after `postprocess`):
```python
    def write_results(self, i: int, p, im, s: list) -> str:
        """Log the ranked list and save a query→top-k montage for gallery retrieval.

        For the non-gallery (embeddings-only) path, delegates to ``BasePredictor.write_results``
        so the default save/log behavior is unchanged.
        """
        if not getattr(self.args, "gallery", None):
            return super().write_results(i, p, im, s)

        from pathlib import Path

        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        prefix = f"{i}: " if (self.source_type.stream or self.source_type.from_img or self.source_type.tensor) else ""
        string = f"{prefix}{result.verbose()}{result.speed['inference']:.1f}ms"

        if self.args.save and result.matches:
            p = Path(p)
            row = [(p, "QUERY", (80, 170, 255))]
            for rank, (gp, score) in enumerate(result.matches, start=1):
                row.append((gp, f"#{rank}  sim={score:.4f}", (200, 200, 200)))
            out = self.save_dir / f"{p.stem}_top{len(result.matches)}.jpg"
            plot_reid_retrieval([row], out)
        return string
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k montage -q`
Run: `pytest tests/reid_audit/test_batch2_embeddings.py -q`
Expected: montage test passes; all batch2 tests pass (guard test now matches the override).

- [ ] **Step 6: Commit**

```bash
git add ultralytics/models/yolo/reid/predict.py tests/reid_audit/test_batch11_retrieval.py tests/reid_audit/test_batch2_embeddings.py
git commit -m "feat(reid): save query->top-k montage in write_results"
```

---

### Task 7: Refactor `ReIDVisualizer` onto the shared engine

**Files:**
- Modify: `ultralytics/solutions/reid_visualizer.py`
- Test: `tests/reid_audit/test_batch11_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k visualizer -q`
Expected: FAIL — `_embed_paths` missing / `_pid_from_name` still present.

- [ ] **Step 3: Rewrite `ReIDVisualizer`**

Replace `ultralytics/solutions/reid_visualizer.py`:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.reid import retrieval
from ultralytics.utils.plotting import plot_reid_retrieval


@dataclass(slots=True)
class RetrievalItem:
    """A single ranked gallery match for a query image."""

    path: Path
    score: float


class ReIDVisualizer:
    """Rank gallery images for a query person and render the top matches.

    Uses any Ultralytics ReID model (PyTorch or exported ONNX) to extract embeddings, ranks gallery
    images by cosine similarity via the shared retrieval engine, and writes a comparison montage.
    Generic: it does not parse Market-1501 filenames — tiles are labeled by rank and similarity.

    Args:
        model: Path or name of a ReID model, e.g. ``best.pt`` or ``best.onnx``.
        imgsz: Inference image size.
        device: Optional inference device.

    Examples:
        >>> from ultralytics.solutions import ReIDVisualizer
        >>> viz = ReIDVisualizer("best.onnx", imgsz=448)
        >>> viz.visualize("query.jpg", "gallery/", k=5)
    """

    def __init__(self, model: str | Path, imgsz: int = 448, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model, task="reid")
        self.imgsz = imgsz
        self.device = device

    def _embed_paths(self, paths: list) -> np.ndarray:
        """Embed image paths via the model's predict() in one batched call (N, D)."""
        results = self.model.predict(
            [str(p) for p in paths], imgsz=self.imgsz, task="reid", device=self.device, verbose=False
        )
        embs = []
        for r in results:
            if r.embeddings is None:
                raise RuntimeError("model produced no embedding")
            data = r.embeddings.data
            data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            embs.append(data.reshape(-1).astype(np.float32))
        return np.stack(embs, axis=0)

    def rank(self, query: str | Path, gallery: str | Path, k: int = 5) -> list[RetrievalItem]:
        """Rank gallery images for a query image by cosine similarity (top-k)."""
        query = Path(query)
        gallery_paths, gallery_embs = retrieval.build_gallery(
            self._embed_paths, gallery, cache=None, model_id=str(self.model.model_name), imgsz=self.imgsz
        )
        query_emb = retrieval.l2_normalize(self._embed_paths([query]))
        idx, scores = retrieval.cosine_topk(query_emb, gallery_embs, k)
        return [RetrievalItem(path=gallery_paths[j], score=float(s)) for j, s in zip(idx[0], scores[0])]

    def visualize(
        self, query: str | Path, gallery: str | Path, k: int = 5, out_path: str | Path | None = None
    ) -> Path:
        """Rank the gallery and save a comparison strip with the top-k matches."""
        query = Path(query)
        matches = self.rank(query, gallery, k=k)
        q_tile = (query, "QUERY", (80, 170, 255))
        match_tiles = [
            (item.path, f"#{rank}  sim={item.score:.4f}", (200, 200, 200))
            for rank, item in enumerate(matches, start=1)
        ]
        out_path = Path(out_path) if out_path is not None else query.with_name(f"{query.stem}_reid_top{k}.jpg")
        return plot_reid_retrieval([[q_tile, *match_tiles]], out_path)

    def __call__(self, query: str | Path, gallery: str | Path, k: int = 5, out_path: str | Path | None = None) -> Path:
        """Shortcut for ``visualize()``."""
        return self.visualize(query, gallery, k=k, out_path=out_path)
```

> Note: `self.model.model_name` is the loaded weights name used as the cache `model_id`. If unavailable on a given build, substitute `str(self.model.ckpt_path)` — verify with a quick `python -c "from ultralytics import YOLO; print(YOLO('yolo26n-reid.pt').model_name)"` during implementation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k visualizer -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/solutions/reid_visualizer.py tests/reid_audit/test_batch11_retrieval.py
git commit -m "refactor(reid): ReIDVisualizer reuses shared retrieval engine (generic, batched)"
```

---

### Task 8: End-to-end CLI integration test + docstring example

**Files:**
- Test: `tests/reid_audit/test_batch11_retrieval.py`
- Modify: `ultralytics/models/yolo/reid/predict.py` (module docstring example only — already added in Task 5)

- [ ] **Step 1: Write the integration test (gated on weight availability)**

Append to `tests/reid_audit/test_batch11_retrieval.py`:

```python
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
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/reid_audit/test_batch11_retrieval.py -k e2e -q`
Expected: PASS if `yolo26n-reid.pt` downloads; otherwise SKIP (acceptable). If it FAILS with a real error, debug before proceeding.

- [ ] **Step 3: Run the full retrieval + reid-audit suite**

Run: `pytest tests/reid_audit/ -q`
Expected: all pass (or skip for GPU/dataset/weights-gated tests). No failures.

- [ ] **Step 4: Manual CLI smoke (optional, if weights available)**

Run:
```bash
yolo reid predict model=yolo26n-reid.pt source=ultralytics/assets/bus.jpg gallery=ultralytics/assets/ topk=3 imgsz=64
```
Expected: console prints a `#1 ... 0.xxxx, #2 ...` ranked line per query; a `bus_top3.jpg` montage appears under `runs/reid/predict*/`.

- [ ] **Step 5: Commit**

```bash
git add tests/reid_audit/test_batch11_retrieval.py
git commit -m "test(reid): e2e CLI gallery retrieval integration test"
```

---

## Self-Review

**Spec coverage:**
- CLI `gallery`/`topk`/`reid_cache` → Task 4 (registration) + Task 5 (read & use). ✓
- No-gallery backward compat → Task 5 (`test_predictor_no_gallery_is_embeddings_only`) + Task 6 (delegates to super). ✓
- Saved montage → Task 6. ✓
- Console ranked list → Task 3 (`verbose()` line, surfaced via inherited/override write_results). ✓
- Structured Results (`Results.matches`) → Task 3 + Task 5. ✓
- Gallery embedded once / lazy index → Task 5 (`_ensure_gallery`). ✓
- Optional cache (load/validate/rebuild) → Task 2. ✓
- Generic, PID parsing dropped → Task 7. ✓
- Batched embedding → Task 5 (`_embed_paths` batches) + Task 7 (single predict call). ✓
- Error handling (missing/empty gallery, topk clamp, stale cache) → Task 1 (`scan_gallery` raises), Task 1 (`cosine_topk` clamp), Task 2 (stale rebuild). ✓
- Shared engine reused by both predictor and visualizer → Task 1/2 + Task 5 + Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `build_gallery(embed_fn, gallery, cache, model_id, imgsz)` signature identical across Tasks 2, 5, 7. `cosine_topk(query, gallery, topk) -> (idx, scores)` consistent in Tasks 1, 5, 7. `Results.matches` is `list[(str, float)] | None` everywhere. `_embed_paths(paths) -> np.ndarray (N, D)` consistent in Tasks 5 and 7. ✓

**Known follow-up flagged for implementer:** confirm `self.model.model_name` exists for the visualizer cache id (Task 7 note); `matches` is intentionally excluded from `Results._keys` and from `summary()`/`save_txt()` (out of scope — montage/console/Results.matches only).
