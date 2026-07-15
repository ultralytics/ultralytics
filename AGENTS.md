# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics (`ultralytics` on PyPI, AGPL-3.0) is the official Python package for YOLO-family vision models — detection, instance and semantic segmentation, classification, pose, oriented boxes, and tracking — plus training, validation, benchmarking, export to 19 deployment formats, and the `yolo` CLI. Supported floors are Python>=3.8 with PyTorch>=1.8.

## Core Principles (CRITICAL)

**Delete > Replace > Add.** Before writing any change, answer in order: what can I delete? what can I replace? only then, what must I add?

The most common agent failure in this repo is reaching for the locally-safest edit — a new guard, flag, or helper — instead of fixing ownership. These tripwires override that instinct:

1. **Never guard a symptom — relocate the trigger.** A fix that adds a condition to suppress bad behavior (a staleness check, an is-initialized flag, a skip-first-call guard, a try/except around broken logic) is wrong by default. Find the code path that should own the behavior, move the logic there, and delete the code that got it wrong. Example: a warning fired from stale state; the right fix was not a recency guard — it deleted the stale detection and moved the trigger into the code path that observes the event live.
2. **Bugfixes are net-negative by default.** A bugfix that adds more lines than it removes needs a one-sentence justification in the PR body naming why deletion and relocation were impossible.
3. **Search the repo before creating anything.** Before building a feature or helper, search the whole package — it likely exists (`ultralytics/utils/` holds most shared helpers). If two modules grow the same logic, consolidate into the shared utility and delete the duplicates. Avoid premature abstraction — three similar lines beat a helper nobody else calls.
4. **Deletion beats caution.** Zero regression means understanding the code you remove, not leaving it in place as insurance. Keeping broken or duplicated code "to be safe" is itself the regression: it is how repos rot. All changes must still ship debugged, validated, and production ready.

**Output gate:** every PR body must contain a `Deleted:` line naming the code removed (functions, branches, files, config). Features must name what they reused or consolidated. `Deleted: nothing` demands the rule-2 justification.

**Review gate:** adversarial reviewers must answer two questions before LGTM: (a) what could have been deleted instead of added? (b) does any added condition suppress a symptom rather than relocate a trigger? A finding on either blocks LGTM.

**This file is code — additions require deletions.** To add a rule here, remove or merge one. When everything is emphasized, nothing is.

**NEVER push to `main`. NEVER force push.** Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Record the pulled live PR-head SHA. Inside the implementation session, run one reviewer covering Core Principles/deduplication/minimalism, production readiness, and performance on the full diff; collect all findings, batch fixes, then reuse it for `<recorded-sha>..HEAD` and affected invariants. After automation, pull and repeat delta review until local and live heads match. Run one final cold full-diff review, require LGTM with no findings, record its SHA, and hand off or merge only while it remains the live head.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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
- `tests/test_exports.py` is partitioned by `--export-env` (env ids from `export_formats()`); omitting the flag runs ALL export formats, so pass `--export-env base` to match CI. GPU tests live in `tests/test_cuda.py` and skip without CUDA.

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
