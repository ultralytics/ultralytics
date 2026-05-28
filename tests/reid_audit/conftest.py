"""Pytest config for ReID audit regression tests.

These tests cover the 5 batches of fixes from the 30-touchpoint audit:
  Batch 2:  Typed Embeddings field on Results
  Batch 3:  Exporter integration for ReID
  Batch 4:  Centralization + DRY (build_yolo_dataset, gallery dataloader, CLIP detection)
  Batch 9:  Docs (not exercised by unit tests)
  Batch 10: Dataset YAMLs

Each test is written so it would have FAILED before the corresponding fix —
i.e. it locks in the contract the audit identified as missing.

Tests are independent so `pytest -n auto` can parallelize them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root (parent of `ultralytics/`) is importable, regardless of where pytest runs.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def market1501_root() -> Path:
    """Path to a local Market-1501 copy. Skip tests needing it if missing."""
    p = Path(os.environ.get("MARKET1501_ROOT", "/root/.cache/autoresearch/Market-1501-v15.09.15"))
    if not p.exists():
        pytest.skip(f"Market-1501 not present at {p}")
    return p


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
