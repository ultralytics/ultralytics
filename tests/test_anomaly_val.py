"""Regression test for YOLOA anomaly validation.

Runs per-category ``model.fit`` + ``model.val`` over the full MVTec-YOLO dataset
and checks the final average mAP values match the reference run::

    AVERAGE  mAP10=0.6444  mAP25=0.4759  mAP50=0.1936  mAP10-50=0.4155

Usage:
    pytest tests/test_anomaly_val.py -v -s --slow
    python tests/test_anomaly_val.py
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.models.yolo.anomaly.train_rnd import MVTEC_CATEGORIES
from ultralytics import YOLOA
from yoloa_utils import good_dir

CKPT = Path("/home/laughing/Downloads/best_converted.pt")
ROOT = Path("/home/laughing/codes/datasets/MVTec-YOLO")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reference average metrics from run_yoloa.py val mode on the full MVTec-YOLO dataset.
EXPECTED = {
    "mAP10": 0.6433,
    "mAP25": 0.4760,
    "mAP50": 0.1929,
    "mAP10-50": 0.4155,
}
TOL = 0.001


def _data_yaml(root: Path, cat: str) -> Path | None:
    """Pick the category data yaml, preferring the *_binary variant."""
    for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
        p = root / cat / name
        if p.exists():
            return p
    return None


def _compute_metrics(metrics):
    """Return mAP10/25/50/10-50 from a YOLOA validation metrics object.

    The anomaly validator uses a coarse IoU grid 0.10:0.05:0.50, so
    ``all_ap`` has 9 columns: col0=0.10, col3=0.25, col8=0.50.
    """
    ap = metrics.box.all_ap
    if ap.shape[1] == 9:
        map10 = float(ap[:, 0].mean())
        map25 = float(ap[:, 3].mean())
        map50 = float(ap[:, 8].mean())
        map10_50 = float(ap.mean())
    else:
        # Fallback for non-anomaly validation grids.
        map10 = float(ap[:, 10].mean()) if ap.shape[1] > 10 else float("nan")
        map25 = float(ap[:, 11].mean()) if ap.shape[1] > 11 else float("nan")
        map50 = float(metrics.box.map50)
        map10_50 = float(ap[:, :10].mean())
    return map10, map25, map50, map10_50


@pytest.mark.slow
def test_anomaly_validation_averages():
    """Full MVTec-YOLO val averages must match the reference run."""
    assert CKPT.exists(), f"checkpoint not found: {CKPT}"
    assert ROOT.is_dir(), f"dataset root not found: {ROOT}"

    model = YOLOA(str(CKPT))
    rows = []

    for cat in MVTEC_CATEGORIES:
        yaml = _data_yaml(ROOT, cat)
        assert yaml is not None, f"[{cat}] no data yaml found"

        model.fit(
            source=str(good_dir(ROOT, cat)),
            name=cat,
            batch=8,
            device=DEVICE,
            cache=None,  # avoid polluting disk; rebuild deterministically each run
        )

        metrics = model.val(
            data=str(yaml),
            iou=0.1,
            end2end=False,
            single_cls=True,
            device=DEVICE,
        )

        map10, map25, map50, map10_50 = _compute_metrics(metrics)
        rows.append(
            {"category": cat, "mAP10": map10, "mAP25": map25, "mAP50": map50, "mAP10-50": map10_50}
        )
        print(
            f"[{cat}] mAP10={map10:.4f} mAP25={map25:.4f} "
            f"mAP50={map50:.4f} mAP10-50={map10_50:.4f}"
        )

    avg = {k: float(np.mean([r[k] for r in rows])) for k in EXPECTED}
    print(
        f"\nAVERAGE  mAP10={avg['mAP10']:.4f}  mAP25={avg['mAP25']:.4f}  "
        f"mAP50={avg['mAP50']:.4f}  mAP10-50={avg['mAP10-50']:.4f}"
    )

    for key, expected in EXPECTED.items():
        assert abs(avg[key] - expected) < TOL, (
            f"{key} average {avg[key]:.4f} differs from expected {expected:.4f} "
            f"(tolerance {TOL})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
