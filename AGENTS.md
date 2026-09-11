# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file. `docs/AGENTS.md` covers the documentation tree; read it whenever a change touches a public Python API, because docs reference pages are generated from docstrings.

Ultralytics (`ultralytics` on PyPI, AGPL-3.0) is the official Python package for YOLO-family vision models — detection, instance and semantic segmentation, depth, classification, pose, oriented boxes, and tracking — plus training, validation, benchmarking, export to 20+ deployment formats, and the `yolo` CLI. Supported floors are Python>=3.8 with PyTorch>=1.8. The version lives in `ultralytics/__init__.py` (`__version__`).

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

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

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Dev install (editable) inside an activated virtualenv; tests also need export/solutions extras.
# [dev] does not include ruff — install it explicitly.
uv pip install -e ".[dev,export-base,export-openvino,solutions]" ruff

# Default (non-slow) suite with the base export environment, matching ci.yml's Tests job
# (CI also runs python tests/cache_test_assets.py first, sets YOLO_AUTOINSTALL=false, caps macOS at 2 workers, and runs ARM single-process)
pytest -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml tests/ --export-env base

# Single file / single test
pytest tests/test_python.py
pytest tests/test_python.py::test_predict_img -v

# Include slow tests (excluded by default in tests/conftest.py)
pytest --slow tests/ --export-env base

# Format and lint (source of truth: [tool.ruff] in pyproject.toml, line length 120)
ruff format . && ruff check --fix .

# Fastest end-to-end smoke test (auto-downloads yolo26n.pt, runs on 2 local asset images)
yolo predict model=yolo26n.pt

# Docs: see docs/AGENTS.md (python docs/build_reference.py, python docs/build_docs.py)
```

- Tests hit the live network: weights (e.g. `yolo26n.pt`) and assets auto-download from GitHub releases into `WEIGHTS_DIR`, with `MODEL` deliberately under a "path with spaces" directory. Even a focused pytest run executes session cleanup (`tests/conftest.py` `pytest_sessionfinish`) that deletes `*.onnx`/`*.torchscript` files and `*.mlpackage`/`*_openvino_model` directories under `WEIGHTS_DIR`, and unconditionally deletes `bus.jpg`, `yolo26n.onnx`, and `yolo26n.torchscript` from the current working directory, so run tests from a scratch cwd with a dedicated weights directory. Export tests should use the `isolated_model` fixture (or call `isolated_model_path(tmp_path, model)`, a plain helper, for other weights) to avoid xdist filename races.
- `pyproject.toml` pytest `addopts` includes `--doctest-modules`, so pointing pytest at `ultralytics/` runs docstring doctests — CI only runs `tests/`, so package doctests are NOT exercised in CI.
- `tests/test_exports.py` is partitioned by `--export-env` (env ids from `EXPORT_ENVS` in `engine/exporter.py`); omitting the flag removes that filter only — slow, platform, and dependency skips still apply, and the flag never installs anything. `.github/scripts/create-export-env.py --list` shows the isolated environments and `--env <id>` builds one and runs its smoke exports. GPU tests live in `tests/test_cuda.py` and skip without CUDA.
- Workflows in `.github/workflows/`: `ci.yml` (Tests on Python 3.13 across ubuntu-latest, macos-26, windows-latest, ubuntu-24.04-arm, plus a Python 3.8 / torch 1.8.0 floor job; Benchmarks; GPU), `format.yml` (Ultralytics Actions: ruff, docformatter, prettier, codespell, license headers, automated PR review — expect bot commits on PR branches), `docs.yml` (runs `ruff check --extend-select F,I,D,UP,RUF,FA` for docstring rules and pushes "Auto-update Ultralytics Docs Reference" commits to the branch), `publish.yml` (Publish to PyPI), `docker.yml`, `conda-check-prs.yml`, `cla.yml`, `links.yml`, `fuzz.yml`, `merge-main-into-prs.yml`, `stale.yml`, `mirror.yml`.

## Conventions

- Ultralytics-owned PyPI packages use `MAJOR.MINOR.PATCH` versions only; no suffixes.
- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Google-style docstrings with types in parentheses (`arg1 (int): ...`); `[tool.ruff.lint.pydocstyle] convention = "google"` configures the docstring rules and `docs.yml` enables them with `--extend-select D`, so a bare `ruff check` does not reproduce that pass. Ruff formats docstring code blocks; the Actions bot also runs docformatter, prettier (YAML/JSON/Markdown), and codespell. Format markdown exactly as the bot does, never with unpinned defaults: `npx prettier@3.8.5 --tab-width 4 --print-width 120 --write` for `docs/**/*.md` (the documentation dialect requires 4-space list continuation; prettier's default tab width 2 breaks rendering) and the same command without `--tab-width` for markdown outside `docs/`.
- Releases: bump `__version__` in `ultralytics/__init__.py`; on push to main, `publish.yml` detects the increment, then tags, creates the GitHub release, and publishes to PyPI (gated to the ultralytics repo and glenn-jocher).
- Tasks and modes are listed in one canonical order everywhere — tables, navs, prose, code, and the Ultralytics Platform: `detect, segment, semantic, depth, classify, pose, obb` and `train, val, predict, export, track, benchmark`. `TASKS` and `MODES` in `ultralytics/cfg/__init__.py` are ordered tuples that define it; never introduce a different ordering.
- Public precision is the single `quantize` arg (`_handle_deprecation` maps legacy `half`/`int8` onto it); `nms` is tri-state (`None` external NMS, `True` embed NMS on export, `False` NMS-free head where supported; legacy `end2end` maps here). Older examples that use `half=`, `int8=`, or `end2end=` are obsolete — don't copy them into new code or docs.

## Architecture

Paths below are relative to `ultralytics/` unless they start with `docs/`, `tests/`, or `.github/`.

**Request flow.** The `yolo` CLI (`entrypoint` in `cfg/__init__.py`; `pyproject.toml` maps both `yolo` and `ultralytics` to it) parses `key=value` args and calls `getattr(model, mode)(**overrides)`. The Python `Model` facade (`engine/model.py`: `.train()`, `.val()`, `.predict()`, `.track()`, `.export()`, `.benchmark()`, `.tune()`) is the single entry for both paths. For train/val/predict it picks the task class with `Model._smart_load(key)`, which returns `self.task_map[self.task][key]` for `key in {"model", "trainer", "validator", "predictor"}`; the class constructor then merges the overrides with `cfg/default.yaml` through `get_cfg`. `export()` constructs `Exporter` directly, `benchmark()` calls `utils.benchmarks.benchmark`, and `tune()` runs `Tuner` or Ray Tune. Every `Model` subclass defines `task_map` in `models/<family>/model.py`; entries vary by family (SAM exposes only a predictor, NAS and FastSAM a predictor and validator, YOLO the full set). `models/llm.py` `LLM` is not a `Model` subclass and has no `task_map`. Method signatures also carry Python-only controls not in the YAML (`predict(stream=, predictor=)`, `val(validator=)`, `train(trainer=)`), and Python `predict` defaults to `save=False` while the CLI saves.

- `engine/` — model-agnostic base classes. A new task subclasses these and overrides hooks; do not fork the loops.
  - `trainer.py` `BaseTrainer`: hooks `get_model`, `get_validator`, `get_dataloader`, `build_dataset`, `preprocess_batch`, `label_loss_items`, `set_model_attributes`, `plot_training_samples`, `plot_training_labels`. Loop is `_setup_train` → `_do_train` → `final_eval`; resume is `check_resume`/`resume_training`; `build_optimizer` picks the optimizer (`optimizer=auto` chooses `MuSGD` from `optim/muon.py` for long runs, else `AdamW`). `MultiTrainer` fine-tunes the base model on each dataset in series and returns `{dataset: metrics}` — not joint training. Knowledge distillation is wrapped by `nn/distill_model.py`.
  - `validator.py` `BaseValidator`: hooks `preprocess`, `postprocess`, `init_metrics`, `update_metrics`, `finalize_metrics`, `get_stats`, `print_results`, `pred_to_json`, `eval_json`; prediction/GT matching in `match_predictions`.
  - `predictor.py` `BasePredictor`: hooks `preprocess` (BGR arrays → normalized RGB BCHW; tensor inputs are validated by `data/loaders.py` `LoadTensor`), `inference`, `pre_transform`, `postprocess`, `write_results`; the loop is `stream_inference`, and `setup_model` wraps weights in `AutoBackend`. `Model.predict()` caches its predictor and only rebuilds it when setup args (device, `channels_last`, `nms`, ...) change, so repeated-call bugs live in that reuse path; `stream=True` returns a lazy generator that must be consumed, the default materializes every result in memory.
  - `exporter.py` `Exporter`: `export_formats()` is the format registry (one row per format; drives the CLI `format=` arg, `AutoBackend` detection, benchmarks, and `tests/test_exports.py`), and `EXPORT_ENVS` assigns rows to isolated CI environments. Each `export_<fmt>` method is decorated with `@try_export`; conversion implementations live under `utils/export/` (follow the `export_*` method to its import — related formats share a module, e.g. every TensorFlow variant uses `utils/export/tensorflow.py`). Keep export-format behavior there: bind format-specific code onto the head at export time (as `tf_wrapper` does for `kpts_decode` and `_get_decode_boxes`) or set an attribute the head reads — do not add new `self.format` branches to `nn/modules/head.py`.
  - `results.py` `Results` with `Boxes`, `Masks`, `Keypoints`, `Probs`, `OBB`, `SemanticMask`, `DepthMap` (`plot`, `save_txt`, `summary`). `tuner.py` `Tuner` (hyperparameter evolution, writes `tune_results.ndjson`).
- `models/<family>/` — `yolo/` has one subpackage per task (`detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, `obb`), each with `train.py`, `val.py`, `predict.py`; also `rtdetr/`, `sam/`, `fastsam/`, `nas/`, `llm.py`, and `models/utils/loss.py` for the RT-DETR losses. `YOLO.__init__` swaps its class to `YOLOWorld` or `YOLOE` from the filename stem (`-world`, `yoloe`), and after loading to `RTDETR` when the head class name — or, for exported files, the `head` metadata read by `BaseBackend.read_metadata` — contains `RTDETR`; a loaded `WorldModel` also swaps to `YOLOWorld`. `SAM` is lazy-imported so `import ultralytics` never pulls torchvision.
- `nn/` — `tasks.py`: `BaseModel.forward()` routes a batch dict to `loss()` (which builds the task criterion via `init_criterion()`) and a tensor to prediction. `parse_model` builds a model from YAML by resolving layer names through this module's `globals()` (`nn.*` and `torchvision.ops.*` have separate lookups) and consults the `base_modules`/`repeat_modules` sets for channel and repeat handling. Task models: `DetectionModel`, `SegmentationModel`, `PoseModel`, `OBBModel`, `DepthModel`, `SemanticSegmentationModel`, `ClassificationModel`, `RTDETRDetectionModel`, `WorldModel`, `YOLOEModel`, `YOLOESegModel`, `Ensemble`. Checkpoints: `torch_safe_load` (restricted unpickling is opt-in via `ULTRALYTICS_SAFE_LOAD` or `safe_only=True` and silently falls back to normal loading when the torch serialization API is too old) and `load_checkpoint`, with `temporary_modules` for renamed-class back-compat; also `guess_model_task`, `yaml_model_load`. `modules/`: `conv.py`, `block.py`, `head.py` (`Detect`, `Segment`/`Segment26`, `Pose`/`Pose26`, `OBB`/`OBB26`, `Depth`, `Classify`, `SemanticSegment`, `WorldDetect`, `YOLOEDetect`, `RTDETRDecoder`, `v10Detect`), `transformer.py`, `activation.py`. `autobackend.py` `AutoBackend`: `_model_type` matches `export_formats()` suffixes against the filename (with special cases for QNN, LiteRT, legacy CoreML, DNN, and Triton URLs), `_BACKEND_MAP` maps the format to a runtime class in `backends/` (all subclass `BaseBackend` in `backends/base.py`), and `forward`/`warmup` present one interface — layouts, precision, and which formats actually warm up differ per backend.
- `cfg/` — `default.yaml` defines the shared configuration schema and defaults (loaded once into `DEFAULT_CFG` in `utils/__init__.py`). `__init__.py` holds `TASKS`, `MODES`, `TASK2DATA`/`TASK2CALIBRATIONDATA`/`TASK2MODEL`/`TASK2METRIC`, `get_cfg` (the single merge point), the `CFG_FLOAT_KEYS`/`CFG_FRACTION_KEYS`/`CFG_INT_KEYS`/`CFG_BOOL_KEYS`/`CFG_STR_KEYS` type gates, `check_dict_alignment` (typo suggestions), `_handle_deprecation` (every legacy arg rename), `handle_yolo_settings`, `entrypoint`. Model YAMLs in `models/{11,12,26,v3,v5,v6,v8,v9,v10,rt-detr}/`, dataset YAMLs in `datasets/`, tracker YAMLs in `trackers/`.
- `data/` — `build.py` (`build_yolo_dataset`, `build_dataloader`, `InfiniteDataLoader`; predict source dispatch in `check_source` and `load_inference_source`), `base.py` (`BaseDataset`: `get_img_files`, `load_image`, RAM/disk image caching), `dataset.py` (`YOLODataset` plus `DepthDataset`, `SemanticDataset`, `GroundingDataset`, `YOLOMultiModalDataset`, `ClassificationDataset`), `augment.py` (`Compose`, `Mosaic`, `RandomPerspective`, `LetterBox`, `Albumentations`, `Format`/`SemanticFormat`/`DepthFormat`, `v8_transforms`), `utils.py` (`check_det_dataset`, `check_cls_dataset`, `verify_image_label`, `get_hash`), `loaders.py` (files, streams, screenshots, tensors), `converter.py`, `annotator.py`, `split.py`. File-based dataset subclasses reuse `YOLODataset.cache_labels()` through the `get_label_files`, `get_cache_hash`, `verify_args`, `result_to_label`, and `scan_summary` hooks — extend those instead of duplicating the scan/cache loop. Label cache: `cache_labels` writes `<labels-dir>.cache` next to the labels directory (e.g. `labels/train.cache`), validated by `DATASET_CACHE_VERSION` and `get_cache_hash()`; `get_hash()` hashes paths plus total file size, not contents or mtimes, so a same-size label edit can keep a stale cache — delete the `.cache` file when labels change. Geometric augmentations extend `BaseTransform`: compute parameters once in `get_params()` and apply them to images, `Instances` (`utils/instance.py`), semantic masks, and depth alike — an image-only change silently corrupts semantic/depth supervision.
- `utils/` — `__init__.py`: `LOGGER`, `SETTINGS` (`SettingsManager`, persisted under `USER_CONFIG_DIR`; `yolo settings` CLI), `DEFAULT_CFG`, `ROOT`, `ASSETS`, `WEIGHTS_DIR`, `DATASETS_DIR`, `RANK`/`LOCAL_RANK`, `ONLINE`, `YAML.load`/`YAML.save`, `colorstr`. `checks.py` (`check_requirements` auto-installs unless `YOLO_AUTOINSTALL=false`, `check_version`, `check_imgsz`, `check_file`, `check_amp`, `check_yolo`), `downloads.py` (`attempt_download_asset`, `safe_download`, `GITHUB_ASSETS_NAMES` from `ultralytics/assets` releases), `torch_utils.py` (device selection, EMA, fuse, profiling), `ops.py` (box/mask ops, coordinate conversions), `nms.py` (shared detection postprocessing; end-to-end BNC predictions and ordinary BCN predictions take different paths), `metrics.py`, `loss.py`, `tal.py` (task-aligned assigner), `plotting.py`, `benchmarks.py`, `autobatch.py`, `autodevice.py`, `dist.py` (DDP launch), `export/` per-format conversion code, `callbacks/` (`base.py` event registry plus `clearml`, `comet`, `dvc`, `mlflow`, `platform`, `raytune`, `tensorboard`, `wb`, excluded from coverage). Callback gotchas: `optimizer_step`, `on_before_zero_grad`, and `on_params_update` are registered placeholders that nothing emits by default, and `add_integration_callbacks()` only attaches the training integrations when the instance's class name contains `Trainer` — check the `run_callbacks()` call site, not just the registry.
- `trackers/` — `track.py` registers `on_predict_*` callbacks; `bot_sort.py`, `byte_tracker.py`, `oc_sort.py`, `deep_oc_sort.py`, `fast_tracker.py`, `track_tracker.py` over `basetrack.py`.
- `solutions/` — end-user apps (`object_counter.py`, `heatmap.py`, `speed_estimation.py`, `ai_gym.py`, `parking_management.py`, ...) subclassing `BaseSolution` in `solutions.py`.
- `tests/` — `test_python.py` (core API), `test_cli.py`, `test_engine.py`, `test_exports.py`, `test_cuda.py`, `test_solutions.py`, `test_integrations.py`, `test_ndjson_converter.py`; `conftest.py` adds `--slow` and `--export-env`; shared constants (`MODEL`, `CFG`, `SOURCE`, `TASK_MODEL_DATA`) in `tests/__init__.py`; `cache_test_assets.py` pre-downloads shared weights.

## Where to look

- Training loop, AMP, EMA, resume, DDP → `engine/trainer.py`, `utils/torch_utils.py`, `utils/dist.py`; losses → `utils/loss.py`, `utils/tal.py`, `models/utils/loss.py` (RT-DETR), the model's `init_criterion()` in `nn/tasks.py`.
- Wrong mAP/metrics or val output → `engine/validator.py`, `models/yolo/<task>/val.py`, `utils/metrics.py`.
- Predict results, plotting, saving → `engine/predictor.py`, `engine/results.py`, `utils/plotting.py`; source routing → `data/build.py` `check_source`/`load_inference_source`, `data/loaders.py`; NMS/postprocess → `utils/nms.py`.
- New or changed argument → `cfg/default.yaml`, the matching `CFG_*_KEYS` type set, `_handle_deprecation` if renaming, then `docs/en/usage/cfg.md` and the shared tables in `docs/macros/` (see `docs/AGENTS.md`).
- New layer → the matching module under `nn/modules/` (`conv.py`, `block.py`, `head.py`, `transformer.py`), export it from `nn/modules/__init__.py`, import it into `nn/tasks.py` so `parse_model` can resolve it, and touch `base_modules`/`repeat_modules` only if the layer needs channel or repeat handling.
- New export format → `export_formats()` row plus an `EXPORT_ENVS` recipe, `export_<fmt>` method, conversion code under `utils/export/`, a runtime class in `nn/backends/` exported from `nn/backends/__init__.py` and registered in `AutoBackend._BACKEND_MAP`, then `tests/test_exports.py`.
- New trainable YOLO task → Trainer/Validator/Predictor triplet under `models/yolo/<task>/` wired into `task_map`, a model class with `init_criterion` in `nn/tasks.py`, a head in `nn/modules/head.py`, a YAML in `cfg/models/`, and registrations in `TASKS`, `TASK2DATA`, `TASK2CALIBRATIONDATA`, `TASK2MODEL`, `TASK2METRIC`; check `guess_model_task()`, `build_yolo_dataset()`, and the `Results` type it produces.
- Dataset load/label/cache errors → `data/utils.py`, `data/base.py`, `data/dataset.py` (delete the stale `.cache`); augmentation → `data/augment.py`, `utils/instance.py`.
- Download or asset failures → `utils/downloads.py`, `utils/checks.py` `check_file`.
