# ReID Predict: Gallery Retrieval — Design

**Date:** 2026-06-05
**Status:** Approved design, pending implementation plan

## Problem

The current ReID predict command produces output that is meaningless to a human:

```bash
yolo reid predict model=yolo26l-reid.pt source=ultralytics/assets/bus.jpg imgsz=448
# -> prints a per-image L2-normalized embedding vector
```

An embedding vector is only useful as an intermediate. The natural, useful task for a
re-identification model is **retrieval**: given a *query* image of a person, rank a *gallery*
of images by similarity and return the top-N closest matches.

A `ReIDVisualizer` solution (`ultralytics/solutions/reid_visualizer.py`) already implements
query→gallery ranking + montage rendering, but it is a Python-only helper, embeds images one
at a time, and carries Market-1501-specific filename parsing. It is not wired into the CLI.

## Goal

Make `yolo reid predict` perform gallery retrieval when a gallery is supplied, producing:

1. a saved **montage** image per query (query tile + top-k match tiles with similarity scores),
2. a **console** ranked list (paths + scores), and
3. structured **`Results`** objects carrying the ranked matches (usable from Python).

When no gallery is supplied, behavior is **unchanged** (emits embeddings) — fully backward
compatible.

## Non-goals

- Person *detection* / cropping. Queries and gallery are assumed to be person crops, as the
  ReID model expects (consistent with current predict/val behavior).
- Benchmark evaluation metrics (mAP/CMC) — that lives in `ReidValidator`. This command is for
  ad-hoc/general retrieval, so the Market-1501-style filename parsing (PID/camera extraction,
  correct-match green/red coloring, junk `-1` filtering) is **dropped**. Tiles are labeled with
  rank + similarity only.
- Approximate-NN indexing (faiss etc.). Exact cosine over an in-memory matrix is sufficient at
  the targeted gallery scale.

## CLI

```bash
# Single query image against a gallery folder, top-5
yolo reid predict model=yolo26l-reid.pt source=query.jpg gallery=path/to/gallery/ imgsz=448 topk=5

# A folder (or video) of queries, with an embedding cache for repeat runs
yolo reid predict model=yolo26l-reid.pt source=queries/ gallery=gallery/ topk=10 reid_cache=gal.pt
```

### New arguments

| Arg          | Type        | Default | Meaning |
|--------------|-------------|---------|---------|
| `gallery`    | str \| None | `None`  | Folder recursively scanned for images to rank against. When `None`, predict emits embeddings (legacy behavior). |
| `topk`       | int         | `5`     | Number of ranked matches returned/plotted per query. Clamped to gallery size. |
| `reid_cache` | str \| None | `None`  | Optional `.pt` path. If it exists and is valid, gallery embeddings are loaded from it; otherwise the gallery is embedded and saved there. |

`topk` / `gallery` have no collision with existing args. `reid_cache` is namespaced to avoid
clashing with the existing `cache` train arg.

All three are added to `ultralytics/cfg/default.yaml` and to `TASK_CUSTOM_KEYS["reid"]` in
`ultralytics/cfg/__init__.py` so the CLI dict-alignment check accepts them.

### Cache validity

The cache file stores `{paths: list[str], embs: Tensor (N×D), model: str, imgsz: int}`. On load,
if the stored gallery path list, model identifier, or `imgsz` does not match the current request,
the cache is treated as stale, a warning is logged, and the gallery is re-embedded and rewritten.

## Architecture (Approach A — predictor-native streaming)

The gallery is treated as an **index built once**; the `source` is the **query (or queries)
streamed** through the existing predict loop.

### Shared retrieval engine

Refactor the ranking core out of `ReIDVisualizer` into a small reusable engine (generic, no PID
parsing). Responsibilities:

- `embed_paths(paths) -> Tensor (N×D)`: **batched** embedding of a list of image paths (replaces
  the current one-image-at-a-time loop) returning an L2-normalized matrix.
- `load_or_build_gallery(gallery, cache) -> (paths, embs)`: scan gallery, load/validate/rebuild
  cache, return paths + embedding matrix.
- `rank(query_emb, gallery_embs, topk) -> list[(index, score)]`: exact cosine similarity
  (`gallery_embs @ query`), top-k via argsort.

Both the standalone `ReIDVisualizer` solution and the new `ReidPredictor` path call this engine,
so the two stay consistent. `ReIDVisualizer` keeps its existing public API
(`rank`/`visualize`/`__call__`) but loses the Market-specific helpers (or makes them opt-in
internally — not exposed via the CLI command).

### ReidPredictor changes

1. **Build the gallery index once.** Before the query loop (after `setup_model`, e.g. on the
   first `postprocess` call or a dedicated lazy init), if `self.args.gallery` is set, call
   `load_or_build_gallery` and store `self.gallery_paths` + `self.gallery_embs` on the predictor.
2. **Rank each query in `postprocess`.** The existing per-image flow yields a query embedding;
   when a gallery index is present, compute top-k and attach the matches to that query's
   `Results`. When no gallery, the current embedding-only `Results` is produced unchanged.

This reuses the streaming engine, so a query *folder* or *video* works for free, and the gallery
is embedded only once per run.

### Results & output plumbing

- Add a typed field to `Results`, `Results.matches`, holding the ranked matches for a query —
  a lightweight container of `(path, score)` pairs (mirrors how `embeddings` was added as a
  typed field rather than overloading `probs`). `None` when retrieval was not run.
- `ReidPredictor` overrides `write_results` to (a) append the ranked list to the per-query
  console line and (b) save the montage via `plot_reid_retrieval`, which already accepts one row
  per query (`[(query_path, label, color), (match_path, label, color), ...]`). With PID parsing
  dropped, all match tiles use a single neutral border color and a `#rank sim=0.xxxx` label.
- Montage is saved under the standard predict `save_dir` (e.g. `runs/reid/predict/`), one file
  per query.

## Data flow

```
source (query img / dir / video)            gallery (folder)        reid_cache (.pt, optional)
        |                                          |                         |
        |                                  load_or_build_gallery <-----------+
        |                                          |  (batched embed, once)
        v                                          v
  stream loop  --preprocess-->  model  --embed-->  query_emb
        |                                          |
        |                                  rank(query_emb, gallery_embs, topk)
        |                                          |
        v                                          v
  postprocess  ----------------------------> Results.matches = [(path, score), ...]
        |
        +--> write_results: console ranked list + montage (plot_reid_retrieval)
        +--> yields Results (Python API)
```

## Error handling

- `gallery` path missing / contains no images → clear `FileNotFoundError`/`RuntimeError`
  (reuse the engine's existing checks).
- `topk` > gallery size → clamp to gallery size, log a debug note.
- A gallery image that fails to load/embed → skipped with a warning (existing behavior), so one
  bad file does not abort the run.
- Stale/incompatible `reid_cache` → warn and rebuild rather than erroring.
- Query that produces no embedding → skipped with a warning.

## Testing

**Unit**
- Engine `rank()` on tiny synthetic embeddings with a known nearest-neighbor order → asserts
  ordering and scores.
- Cache round-trip: build → save → load returns identical paths/embeddings; mutating the
  recorded `imgsz`/`model`/paths triggers a rebuild.
- `topk` clamping when gallery smaller than `topk`.

**Integration**
- `yolo reid predict source=<small dir> gallery=<small dir> topk=3` (tiny fixture set):
  asserts a montage file is written, the console line includes ranked matches, and
  `Results.matches` is populated with `topk` entries sorted by descending score.
- No-gallery call still yields `Results.embeddings` and `Results.matches is None`
  (backward-compat guard).

## Affected files

- `ultralytics/models/yolo/reid/predict.py` — gallery index build + ranking in `postprocess`,
  `write_results` override.
- `ultralytics/solutions/reid_visualizer.py` — refactor ranking core into shared engine, add
  batched embedding + cache, drop/opt-out Market PID parsing.
- `ultralytics/engine/results.py` — add typed `Results.matches` field.
- `ultralytics/cfg/default.yaml` — add `gallery`, `topk`, `reid_cache`.
- `ultralytics/cfg/__init__.py` — add the three keys to `TASK_CUSTOM_KEYS["reid"]`.
- `ultralytics/utils/plotting.py` — reuse `plot_reid_retrieval` (likely no change; verify
  neutral-color path).
- Tests under `tests/` — unit + integration as above.
