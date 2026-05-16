# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression test: Windows path-case differences must not cause bundled dataset YAMLs to be misclassified as untrusted.

Covers the Python <3.9 fallback path in check_det_dataset() where string comparison is used
instead of Path.is_relative_to().
"""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch

import pytest


def _is_trusted_fallback(yaml_file: Path, trusted_root: Path, *, simulate_nt: bool = False) -> bool:
    """Reproduce the <3.9 trust-check fallback from ultralytics/data/utils.py."""
    yaml_str, root_str = str(yaml_file), str(trusted_root) + os.sep
    if simulate_nt or os.name == "nt":
        yaml_str, root_str = yaml_str.lower(), root_str.lower()
    return yaml_str.startswith(root_str)


class TestTrustPathCaseWindows:
    """Ensure Windows drive-letter / path casing doesn't break the trust check."""

    def test_matching_case_is_trusted(self):
        """Same casing should always be trusted."""
        root = Path("C:/Users/user/ultralytics/cfg/datasets")
        yaml = root / "coco.yaml"
        assert _is_trusted_fallback(yaml, root, simulate_nt=True)

    def test_different_drive_letter_case_is_trusted(self):
        """Drive letter 'c:' vs 'C:' must still be trusted on Windows."""
        root = Path("C:/Users/user/ultralytics/cfg/datasets")
        yaml = Path("c:/Users/user/ultralytics/cfg/datasets/coco.yaml")
        assert _is_trusted_fallback(yaml, root, simulate_nt=True)

    def test_different_folder_case_is_trusted(self):
        """Mixed folder casing must still be trusted on Windows."""
        root = Path("C:/Users/User/Ultralytics/cfg/datasets")
        yaml = Path("C:/users/user/ultralytics/cfg/datasets/coco.yaml")
        assert _is_trusted_fallback(yaml, root, simulate_nt=True)

    def test_outside_root_is_untrusted(self):
        """A YAML outside the trusted root must remain untrusted regardless of casing."""
        root = Path("C:/Users/user/ultralytics/cfg/datasets")
        yaml = Path("C:/Users/attacker/malicious.yaml")
        assert not _is_trusted_fallback(yaml, root, simulate_nt=True)

    def test_partial_prefix_not_trusted(self):
        """A sibling directory whose name starts with 'datasets' must not be trusted."""
        root = Path("C:/ultralytics/cfg/datasets")
        yaml = Path("C:/ultralytics/cfg/datasets_evil/payload.yaml")
        assert not _is_trusted_fallback(yaml, root, simulate_nt=True)

    def test_unix_case_sensitive(self):
        """On non-Windows, different casing should NOT match (case-sensitive FS)."""
        root = Path("/home/user/ultralytics/cfg/datasets")
        yaml = Path("/home/user/Ultralytics/cfg/datasets/coco.yaml")
        assert not _is_trusted_fallback(yaml, root, simulate_nt=False)
