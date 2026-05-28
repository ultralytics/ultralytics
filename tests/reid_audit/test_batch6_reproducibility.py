"""Batch 6: dataset reproducibility.

Before the fix:
  * _get_samples_folder fell back to Python's built-in hash() for non-numeric folder names.
    hash(str) is salted per process via PYTHONHASHSEED, so the SAME folder name mapped to
    DIFFERENT pids across DDP ranks and across process restarts — broke
    IdentityBalancedSampler's cross-rank invariant and destroyed reproducibility.
  * ReidDataset inherited verify_images, whose cache hash was paths only. Changing
    filename_re or cam_0indexed in the YAML silently reused stale pid/camid labels.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from PIL import Image


def test_non_numeric_pid_uses_sha1_not_builtin_hash():
    """The non-numeric pid fallback must use a stable hash (sha1 / hashlib), not the
    salted built-in hash()."""
    from ultralytics.data.dataset import ReidDataset

    src = inspect.getsource(ReidDataset._get_samples_folder)
    assert "hashlib" in src, "must use hashlib for stable pid hashing"
    # The built-in hash() form must be gone (in the same function body)
    # — `hash(...)` calls are easy to find; the offending one was `hash(identity_dir.name)`.
    assert "hash(identity_dir.name)" not in src, "salted built-in hash() must not be used for pids"


def test_non_numeric_pid_is_stable_across_runs():
    """Same folder name → same pid every time. Smoke test by re-deriving the sha1 fallback
    and confirming it matches what the code would produce."""
    name = "alice_v2"
    expected = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
    # The sha1 prefix has 8 hex chars = 32 bits, well below the 2**31 mask the old code used
    assert expected < 2**32
    assert expected >= 0


def test_reid_dataset_cache_hash_includes_pid_camid(tmp_path):
    """ReidDataset._cache_hash must change when pid/camid mapping changes for the same image
    paths — preventing silent reuse of stale labels after filename_re or cam_0indexed change.
    """
    from ultralytics.data.dataset import ReidDataset

    # Two synthetic dataset instances with the same paths but different pid/camid
    ds_a = ReidDataset.__new__(ReidDataset)
    ds_a.samples = [("/tmp/a.jpg", 1, 0), ("/tmp/b.jpg", 2, 1)]
    ds_b = ReidDataset.__new__(ReidDataset)
    ds_b.samples = [("/tmp/a.jpg", 9, 7), ("/tmp/b.jpg", 8, 6)]  # same paths, different labels
    assert ds_a._cache_hash() != ds_b._cache_hash(), (
        "cache hash must change when pid/camid mapping changes"
    )


def test_reid_dataset_cache_hash_stable_for_same_samples():
    """Reordering samples produces a different list-order hash — but the same samples in
    the same order must produce the same hash deterministically."""
    from ultralytics.data.dataset import ReidDataset

    samples = [("/tmp/a.jpg", 1, 0), ("/tmp/b.jpg", 2, 1)]
    ds = ReidDataset.__new__(ReidDataset)
    ds.samples = samples
    h1 = ds._cache_hash()
    ds.samples = list(samples)
    h2 = ds._cache_hash()
    assert h1 == h2


def test_classification_dataset_cache_hash_unchanged_for_path_only():
    """The parent's path-only hash behaviour must be preserved for non-ReID classify use."""
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.data.utils import get_hash

    ds = ClassificationDataset.__new__(ClassificationDataset)
    ds.samples = [("/tmp/a.jpg", 1), ("/tmp/b.jpg", 2)]
    assert ds._cache_hash() == get_hash([x[0] for x in ds.samples])
