# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics (`ultralytics` on PyPI, AGPL-3.0) is the official Python package for YOLO-family vision models — detection, instance and semantic segmentation, classification, pose, oriented boxes, and tracking — plus training, validation, benchmarking, export to 20 deployment formats, and the `yolo` CLI. Supported floors are Python>=3.8 with PyTorch>=1.8.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

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
uv pip install -e ".[dev,export-base,solutions]"

# All tests with coverage, matching ci.yml's Tests job (CI also sets YOLO_AUTOINSTALL=false and drops -n auto on ARM)
pytest -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml tests/ --export-env base

# Single file / single test
pytest tests/test_python.py
pytest tests/test_python.py::test_predict_img -v

# Include slow tests (excluded by default in tests/conftest.py)
pytest --slow tests/

# Format and lint (source of truth: [tool.ruff] in pyproject.toml, line length 120)
ruff format . && ruff check --fix .

# Regenerate docs/en/reference/ after adding/removing/renaming public APIs (docs.yml runs this)
python docs/build_reference.py

# Fastest end-to-end smoke test (auto-downloads yolo26n.pt, runs on 2 local asset images)
yolo predict model=yolo26n.pt
```

- CI (`ci.yml`) runs tests on Python 3.13 across ubuntu-latest, macos-26, windows-latest, and ubuntu-24.04-arm, plus a floor job on Python 3.8 with torch 1.8.0.
- `pyproject.toml` pytest `addopts` includes `--doctest-modules`, so pointing pytest at `ultralytics/` runs docstring doctests — CI only runs `tests/`, so package doctests are NOT exercised in CI.
- `tests/test_exports.py` is partitioned by `--export-env` (env ids from `export_formats()`); omitting the flag runs ALL export formats. GPU tests live in `tests/test_cuda.py` and skip without CUDA.

## Architecture

The user-facing `Model` facade in `ultralytics/engine/model.py` (`.train()`, `.val()`, `.predict()`, `.export()`, `.track()`) lazily dispatches to task-specific components through each model family's `task_map` property.

- `ultralytics/engine/` — model-agnostic core: `BaseTrainer`, `BaseValidator`, `BasePredictor`, `Exporter`, and `Results`.
- `ultralytics/models/` — families (yolo, rtdetr, sam, fastsam, nas) subclass the engine per task, e.g. `models/yolo/detect/{train,val,predict}.py`; `YOLO.__init__` morphs into `YOLOWorld`, `YOLOE`, or `RTDETR` based on the checkpoint/YAML filename.
- `ultralytics/nn/` — `tasks.py` builds models from YAMLs (`parse_model`), `modules/` is the layer zoo referenced by name in YAMLs, `autobackend.py` gives unified inference across all export formats.
- `ultralytics/cfg/` — `default.yaml` defines ALL train/val/predict/export args (the `overrides` dict flows through `get_cfg` everywhere), plus model/dataset/tracker YAMLs, the `yolo` CLI `entrypoint`, and arg deprecation via `_handle_deprecation`.
- `ultralytics/data/`, `ultralytics/utils/`, `ultralytics/solutions/`, `ultralytics/trackers/` — datasets/augmentation, shared utilities and lifecycle `callbacks/` (integration loggers, excluded from coverage), end-user apps, and BoT-SORT/ByteTrack.

Adding a task or family means a Trainer/Validator/Predictor triplet wired into `task_map`, a model class in `nn/tasks.py`, and a YAML in `cfg/models/`.

## Conventions

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Google-style docstrings with types in parentheses (`arg1 (int): ...`); Ruff enforces `convention = "google"` and formats docstring code blocks; the Actions bot also runs docformatter, prettier (YAML/JSON/Markdown), and codespell — expect bot commits on PR branches. Format markdown exactly as the bot does, never with unpinned defaults: `npx prettier@3.6.2 --tab-width 4 --print-width 120 --write` for `docs/**/*.md` (mkdocs requires 4-space list continuation; prettier's default tab width 2 breaks rendering) and the same command without `--tab-width` for markdown outside `docs/`.
- Tests hit the live network: weights (e.g. `yolo26n.pt`) and assets auto-download from GitHub releases; shared constants (`MODEL`, `CFG`, `SOURCE`) live in `tests/__init__.py`, with `MODEL` deliberately under a "path with spaces" directory.
- Releases: bump `__version__` in `ultralytics/__init__.py`; on push to main, `publish.yml` detects the increment, then tags, creates the GitHub release, and publishes to PyPI (gated to the ultralytics repo and glenn-jocher).
- Docs: docs.ultralytics.com is published from a separate portal repo, so relative `.md` cross-file links are the correct convention in `docs/en/`.
