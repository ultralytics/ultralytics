# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file. `docs/AGENTS.md` covers the documentation tree.

Ultralytics (`ultralytics` on PyPI, AGPL-3.0) is the official Python package for YOLO-family vision models — detection, instance and semantic segmentation, depth, classification, pose, oriented boxes, and tracking — plus training, validation, benchmarking, export to 20+ deployment formats, and the `yolo` CLI. Supported floors are Python>=3.8 with PyTorch>=1.8. The version lives in `ultralytics/__init__.py` (`__version__`).

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

## PR Review

- Require a reproducible production bug or a broadly useful feature; close incorrect, niche, speculative, or AI-generated bloat.
- Treat new arguments and PRs with more than 50 net added non-documentation code lines, counting tests and offsetting additions with deletions, as high-barrier exceptions. The PR author must strongly justify why the complete change cannot be smaller or use an existing owner, and defend how it avoids duplication and improves maintainability or scalability.
- Delete mock tests, including monkeypatched platform or device state. Prefer focused validation of the real code path on real CI targets, adding only minimal regression coverage for a high-risk gap.
- Review the full live diff independently; approvals, comments, descriptions, and green CI are supporting evidence, not proof.
- Reject compatibility shims, duplicated helpers, dead code, unrelated cleanup, and complexity that does not pay for itself.
- Preserve an existing implementation when a maintainer explicitly requests it remain temporarily disabled at its owner
  with a linked tracking issue; require the issue to document the evidence needed to re-enable it.
- Require production-ready behavior across supported tasks, platforms, versions, and integrations affected by the owner change.
- Remind unsigned contributors to complete the CLA. Do not close PRs opened by Ultralytics organization team members.
- Merge only the exact cold-reviewed live head after terminal-green checks and zero unresolved review threads.
- Fix accepted contributions on their existing PR branches; never create replacement or follow-up PRs for review repairs.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Dev install (editable); tests also need export/solutions extras
uv pip install -e ".[dev,export-base,export-openvino,solutions]"

# All tests with coverage, matching ci.yml's Tests job (CI also sets YOLO_AUTOINSTALL=false and drops -n auto on ARM)
pytest -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml tests/ --export-env base

# Single file / single test
pytest tests/test_python.py
pytest tests/test_python.py::test_predict_img -v

# Include slow tests (excluded by default in tests/conftest.py)
pytest --slow tests/

# Format and lint (source of truth: [tool.ruff] in pyproject.toml, line length 120)
ruff format . && ruff check --fix .

# Fastest end-to-end smoke test (auto-downloads yolo26n.pt, runs on 2 local asset images)
yolo predict model=yolo26n.pt

# Docs: see docs/AGENTS.md (python docs/build_reference.py, python docs/build_docs.py)
```

- CI (`ci.yml`) runs tests on Python 3.13 across ubuntu-latest, macos-26, windows-latest, and ubuntu-24.04-arm, plus a floor job on Python 3.8 with torch 1.8.0. Other workflows: `format.yml` (Ultralytics Actions: ruff, docformatter, prettier, codespell, license headers, automated PR review), `docs.yml` (Publish Docs), `publish.yml` (Publish to PyPI), `docker.yml`, `conda-check-prs.yml`, `cla.yml`, `links.yml`, `fuzz.yml`, `merge-main-into-prs.yml`, `stale.yml`, `mirror.yml`.
- `pyproject.toml` pytest `addopts` includes `--doctest-modules`, so pointing pytest at `ultralytics/` runs docstring doctests — CI only runs `tests/`, so package doctests are NOT exercised in CI.
- `tests/test_exports.py` is partitioned by `--export-env` (env ids from `export_formats()`); omitting the flag runs ALL export formats. GPU tests live in `tests/test_cuda.py` and skip without CUDA.

## Architecture

**Request flow.** The `yolo` CLI (`entrypoint` in `ultralytics/cfg/__init__.py`; `pyproject.toml` maps both `yolo` and `ultralytics` to it) and the Python `Model` facade (`ultralytics/engine/model.py`: `.train()`, `.val()`, `.predict()`, `.track()`, `.export()`, `.benchmark()`, `.tune()`) both build an `overrides` dict, merge it with `cfg/default.yaml` through `get_cfg`, then `Model._smart_load(key)` resolves `self.task_map[self.task][key]` for `key in {"model", "trainer", "validator", "predictor"}` and runs that task class. Every model family defines its own `task_map` in `ultralytics/models/<family>/model.py`.

- `ultralytics/engine/` — model-agnostic base classes. A new task subclasses these and overrides hooks; do not fork the loops.
  - `trainer.py` `BaseTrainer`: hooks `get_model`, `get_validator`, `get_dataloader`, `build_dataset`, `preprocess_batch`, `label_loss_items`, `set_model_attributes`, `plot_training_samples`, `plot_training_labels`. Loop is `_setup_train` → `_do_train` → `final_eval`; resume is `check_resume`/`resume_training`; optimizer in `build_optimizer`. `MultiTrainer` trains on multiple datasets.
  - `validator.py` `BaseValidator`: hooks `preprocess`, `postprocess`, `init_metrics`, `update_metrics`, `finalize_metrics`, `get_stats`, `print_results`, `pred_to_json`, `eval_json`; prediction/GT matching in `match_predictions`.
  - `predictor.py` `BasePredictor`: hooks `preprocess`, `inference`, `pre_transform`, `postprocess`, `write_results`; the loop is `stream_inference`, and `setup_model` wraps weights in `AutoBackend`.
  - `exporter.py` `Exporter`: `export_formats()` is the format registry (one row per format; drives the CLI `format=` arg, `AutoBackend` detection, benchmarks, and `tests/test_exports.py`). Each `export_<fmt>` method is decorated with `@try_export`; heavy per-format code lives in `ultralytics/utils/export/<fmt>.py`. Keep export-format behavior in that module: bind format-specific code onto the head at export time (as `tf_wrapper` does for `kpts_decode` and `_get_decode_boxes`) or set an attribute the head reads — do not add new `self.format` branches to `ultralytics/nn/modules/head.py`.
  - `results.py` `Results` with `Boxes`, `Masks`, `Keypoints`, `Probs`, `OBB`, `SemanticMask`, `DepthMap` (`plot`, `save_txt`, `summary`). `tuner.py` `Tuner` (hyperparameter evolution, writes `tune_results.ndjson`).
- `ultralytics/models/<family>/` — `yolo/` has one subpackage per task (`detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, `obb`), each with `train.py`, `val.py`, `predict.py`; also `rtdetr/`, `sam/`, `fastsam/`, `nas/`, `llm.py`. `YOLO.__init__` swaps its class to `YOLOWorld` or `YOLOE` from the filename stem (`-world`, `yoloe`) and to `RTDETR` after loading when the head class name contains `RTDETR`. `SAM` is lazy-imported so `import ultralytics` never pulls torchvision.
- `ultralytics/nn/` — `tasks.py`: `parse_model` builds a model from YAML by resolving layer names to classes in `modules/`; task models `DetectionModel`, `SegmentationModel`, `PoseModel`, `OBBModel`, `DepthModel`, `SemanticSegmentationModel`, `ClassificationModel`, `RTDETRDetectionModel`, `WorldModel`, `YOLOEModel`; `torch_safe_load` (restricted unpickling, `ULTRALYTICS_SAFE_LOAD`) and `load_checkpoint` with `temporary_modules` for renamed-class back-compat, `Ensemble`, `guess_model_task`, `yaml_model_load`. `modules/`: `conv.py`, `block.py`, `head.py` (`Detect`, `Segment`/`Segment26`, `Pose`/`Pose26`, `OBB`/`OBB26`, `Depth`, `Classify`, `SemanticSegment`, `WorldDetect`, `YOLOEDetect`, `RTDETRDecoder`, `v10Detect`), `transformer.py`, `activation.py`. `autobackend.py` `AutoBackend`: `_model_type` picks the format by matching `export_formats()` suffixes against the filename, then `forward`/`warmup` are uniform across every format; the per-format runtime classes live in `backends/`, all subclassing `BaseBackend` in `backends/base.py`.
- `ultralytics/cfg/` — `default.yaml` defines ALL train/val/predict/export args (single source of truth). `__init__.py` holds `TASKS`, `MODES`, `TASK2DATA`/`TASK2MODEL`/`TASK2METRIC`, `get_cfg` (the single merge point), `CFG_FLOAT_KEYS`/`CFG_FRACTION_KEYS`/`CFG_INT_KEYS`/`CFG_BOOL_KEYS`/`CFG_STR_KEYS` type gates, `check_dict_alignment` (typo suggestions), `_handle_deprecation`, `handle_yolo_settings`, `entrypoint`. Model YAMLs in `models/{11,12,26,v3,v5,v6,v8,v9,v10,rt-detr}/`, dataset YAMLs in `datasets/`, tracker YAMLs in `trackers/`.
- `ultralytics/data/` — `build.py` (`build_yolo_dataset`, `build_dataloader`, `InfiniteDataLoader`), `base.py` (`BaseDataset`: `get_img_files`, `load_image`, RAM/disk image caching); `YOLODataset.cache_labels` writes a `labels.cache` beside the labels and `load_dataset_cache_file` in `data/utils.py` reads it, `dataset.py` (`YOLODataset` plus `DepthDataset`, `SemanticDataset`, `GroundingDataset`, `YOLOMultiModalDataset`, `ClassificationDataset`), `augment.py` (`Compose`, `Mosaic`, `RandomPerspective`, `LetterBox`, `Albumentations`, `Format`, `v8_transforms`), `utils.py` (`check_det_dataset`, `check_cls_dataset`, `verify_image_label`), `loaders.py` (predict sources: files, streams, screenshots, tensors), `converter.py`, `annotator.py`, `split.py`.
- `ultralytics/utils/` — `__init__.py`: `LOGGER`, `SETTINGS` (`SettingsManager`, persisted under `USER_CONFIG_DIR`; `yolo settings` CLI), `DEFAULT_CFG`, `ROOT`, `ASSETS`, `WEIGHTS_DIR`, `DATASETS_DIR`, `RANK`/`LOCAL_RANK`, `ONLINE`, `yaml_load`/`yaml_save`, `colorstr`. `checks.py` (`check_requirements` auto-installs unless `YOLO_AUTOINSTALL=false`, `check_version`, `check_imgsz`, `check_file`, `check_amp`, `check_yolo`), `downloads.py` (`attempt_download_asset`, `safe_download`, `GITHUB_ASSETS_NAMES` from `ultralytics/assets` releases), `torch_utils.py` (device selection, EMA, fuse, profiling), `ops.py` (box/mask ops, coordinate conversions), `nms.py`, `metrics.py`, `loss.py`, `tal.py` (task-aligned assigner), `plotting.py`, `benchmarks.py`, `autobatch.py`, `autodevice.py`, `dist.py` (DDP launch), `callbacks/` (`base.py` event hooks plus `clearml`, `comet`, `dvc`, `mlflow`, `platform`, `raytune`, `tensorboard`, `wb` integrations, excluded from coverage), `export/` per-format exporters.
- `ultralytics/trackers/` — `track.py` registers `on_predict_*` callbacks; `bot_sort.py`, `byte_tracker.py`, `oc_sort.py`, `deep_oc_sort.py`, `fast_tracker.py`, `track_tracker.py` over `basetrack.py`.
- `ultralytics/solutions/` — end-user apps (`object_counter.py`, `heatmap.py`, `speed_estimation.py`, `ai_gym.py`, `parking_management.py`, ...) subclassing `BaseSolution` in `solutions.py`.
- `tests/` — `test_python.py` (core API), `test_cli.py`, `test_engine.py`, `test_exports.py`, `test_cuda.py`, `test_solutions.py`, `test_integrations.py`, `test_ndjson_converter.py`; `conftest.py` adds `--slow`; shared constants (`MODEL`, `CFG`, `SOURCE`, `TASK_MODEL_DATA`) in `tests/__init__.py`.

**Where to look.**

- Training loop, AMP, EMA, resume, DDP → `engine/trainer.py`, `utils/torch_utils.py`, `utils/dist.py`; losses → `utils/loss.py`, `utils/tal.py`.
- Wrong mAP/metrics or val output → `engine/validator.py`, `models/yolo/<task>/val.py`, `utils/metrics.py`.
- Predict results, plotting, saving → `engine/predictor.py`, `engine/results.py`, `utils/plotting.py`; input sources → `data/loaders.py`.
- New or changed argument → `cfg/default.yaml`, the matching `CFG_*_KEYS` type set, `_handle_deprecation` if renaming, and `docs/en/usage/cfg.md`.
- New layer → `nn/modules/block.py` and register in `nn/tasks.py` `parse_model`; new head → `nn/modules/head.py`.
- New export format → `export_formats()` row, `export_<fmt>` method, `utils/export/<fmt>.py`, `AutoBackend` loader, `tests/test_exports.py`.
- Dataset load/label/cache errors → `data/utils.py`, `data/base.py`, `data/dataset.py`; augmentation → `data/augment.py`.
- Download or asset failures → `utils/downloads.py`, `utils/checks.py` `check_file`.
- Adding a task or family: Trainer/Validator/Predictor triplet wired into `task_map`, a model class in `nn/tasks.py`, a head in `nn/modules/head.py`, and a YAML in `cfg/models/`.

## Conventions

- Ultralytics-owned PyPI packages use `MAJOR.MINOR.PATCH` versions only; no suffixes.
- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Google-style docstrings with types in parentheses (`arg1 (int): ...`); Ruff enforces `convention = "google"` and formats docstring code blocks; the Actions bot also runs docformatter, prettier (YAML/JSON/Markdown), and codespell — expect bot commits on PR branches. Format markdown exactly as the bot does, never with unpinned defaults: `npx prettier@3.8.5 --tab-width 4 --print-width 120 --write` for `docs/**/*.md` (the documentation dialect requires 4-space list continuation; prettier's default tab width 2 breaks rendering) and the same command without `--tab-width` for markdown outside `docs/`.
- Tests hit the live network: weights (e.g. `yolo26n.pt`) and assets auto-download from GitHub releases; shared constants (`MODEL`, `CFG`, `SOURCE`) live in `tests/__init__.py`, with `MODEL` deliberately under a "path with spaces" directory.
- Releases: bump `__version__` in `ultralytics/__init__.py`; on push to main, `publish.yml` detects the increment, then tags, creates the GitHub release, and publishes to PyPI (gated to the ultralytics repo and glenn-jocher).
- Tasks and modes are listed in one canonical order everywhere — tables, navs, prose, code, and the Ultralytics Platform: `detect, segment, semantic, depth, classify, pose, obb` and `train, val, predict, export, track, benchmark`. `TASKS` and `MODES` in `ultralytics/cfg/__init__.py` are ordered tuples that define it; never introduce a different ordering.
