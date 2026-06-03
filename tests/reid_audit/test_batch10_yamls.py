"""Batch 10: dataset & model YAMLs.

What would have failed before the fix:
  * Combined-ReID.yaml had hardcoded /home/rick/.cache/... paths — wouldn't run elsewhere.
  * yolo26l-reid-2psa-ls1.yaml was byte-identical to yolo26-reid-2psa-ls1.yaml.
  * Market-1501.yaml's download block assigned a dead GDrive URL to an unused variable.
  * yolo26* ReID YAMLs hardcoded nc=751 without a comment that data=*.yaml overrides it.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CFG_DS = REPO / "ultralytics/cfg/datasets"
CFG_M26 = REPO / "ultralytics/cfg/models/26"


def test_combined_reid_yaml_deleted():
    assert not (CFG_DS / "Combined-ReID.yaml").exists(), "Combined-ReID.yaml should be deleted (research artifact)"


def test_yolo26l_reid_2psa_ls1_yaml_deleted():
    """The byte-identical duplicate of yolo26-reid-2psa-ls1.yaml should be gone."""
    assert not (CFG_M26 / "yolo26l-reid-2psa-ls1.yaml").exists()


def test_market1501_yaml_has_working_download_block():
    """Market-1501.yaml's download block must actually call download() with a non-dead URL."""
    text = (CFG_DS / "Market-1501.yaml").read_text()
    assert "download:" in text
    assert "drive.google.com" not in text, "Dead GDrive URL must be replaced with a working mirror"
    assert "download(" in text, "download block must actually invoke download(); not be a no-op script"
    # HF mirror called out in user memory:
    assert "huggingface.co" in text


def test_market1501_yaml_loads_and_has_required_keys():
    d = yaml.safe_load((CFG_DS / "Market-1501.yaml").read_text())
    assert d["path"] == "Market-1501-v15.09.15"
    assert d["nc"] == 751
    for k in ("train", "val", "gallery", "filename_re"):
        assert k in d, f"Market-1501.yaml missing required key {k}"


def test_msmt17_yaml_loads():
    d = yaml.safe_load((CFG_DS / "MSMT17.yaml").read_text())
    assert d["nc"] == 1041
    assert d["filename_re"] == "msmt17"
    for k in ("train", "val", "gallery"):
        assert k in d


def test_dukemtmc_yaml_loads():
    d = yaml.safe_load((CFG_DS / "DukeMTMC-reID.yaml").read_text())
    assert d["nc"] == 702
    assert d["filename_re"] == "dukemtmc"


def test_yolo26_reid_yamls_have_nc_override_comment():
    """All shipping yolo26* reid yamls should signal that data=*.yaml overrides nc."""
    expected = [
        "yolo26-reid.yaml",
        "yolo26-reid-2psa.yaml",
        "yolo26-reid-2psa-ls1.yaml",
        "yolo26-reid-ibn.yaml",
        "yolo26l-reid-3psa.yaml",
    ]
    for fn in expected:
        text = (CFG_M26 / fn).read_text()
        # locate the `nc: 751` line and confirm the override note is in the same comment
        for line in text.splitlines():
            if line.lstrip().startswith("nc:"):
                assert "overridden by data" in line.lower(), (
                    f"{fn}: nc line lacks override comment — readers won't know data=*.yaml -> nc"
                )
                break
        else:
            raise AssertionError(f"{fn}: no nc: line found")


def test_all_reid_dataset_yamls_have_agpl_header():
    """Every shipped dataset YAML must begin with the AGPL header line."""
    for fn in ("Market-1501.yaml", "MSMT17.yaml", "DukeMTMC-reID.yaml"):
        first_line = (CFG_DS / fn).read_text().splitlines()[0]
        assert "AGPL" in first_line, f"{fn}: missing AGPL header on line 1"
