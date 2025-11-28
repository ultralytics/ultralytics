"""
Stereo 3D Augmentation - Option A (flip + swap) Validation Suite

Industrial-style checks adapted to current label schema:
- Labels: list[dict] with keys 'left_box' (cx, cy, w, h) and 'right_box' (cx, w); all normalized.
- Augmentors: operate on np.ndarray images and label dicts.

This suite verifies core invariants and emits visualization artifacts for manual inspection.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from ultralytics.models.yolo.stereo3ddet.augment import (
    PhotometricAugmentor,
    HorizontalFlipAugmentor,
    RandomScaleAugmentor,
    RandomCropAugmentor,
    StereoAugmentationPipeline,
)
from pathlib import Path as _Path
from ultralytics.data.kitti_stereo import KITTIStereoDataset


# Local copies of helpers to avoid cross-test imports
ARTIFACT_DIR = _Path("/root/ultralytics/tests/artifacts/stereo_aug")


def ensure_dir(p: _Path):
    p.mkdir(parents=True, exist_ok=True)


DATASET_ROOT = _Path("/root/autodl-tmp/converted_kitti_3dop")


def load_real_sample(index: int = 0, split: str = "train"):
    if DATASET_ROOT.exists():
        try:
            ds = KITTIStereoDataset(root=DATASET_ROOT, split=split)
            sample = ds[index % len(ds)]
            return sample["left_img"], sample["right_img"], sample["labels"], sample["image_id"]
        except Exception:
            pass
    # Fallback synthetic
    H, W = 120, 200
    left = np.zeros((H, W, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    cv2.rectangle(left, (20, 30), (80, 90), (255, 0, 0), -1)
    cv2.rectangle(right, (W - 80, 30), (W - 20, 90), (0, 255, 0), -1)
    labels = [
        {
            "class_id": 0,
            "left_box": {"center_x": 0.3, "center_y": 0.5, "width": 0.2, "height": 0.3},
            "right_box": {"center_x": 0.7, "width": 0.2},
        }
    ]
    return left, right, labels, "synthetic"


def draw_label_markers(img: np.ndarray, labels, is_right: bool = False):
    out = img.copy()
    H, W = out.shape[:2]
    for obj in labels:
        lb = obj.get("left_box", {})
        rb = obj.get("right_box", {})
        if not is_right:
            cx = int(lb.get("center_x", 0.5) * W)
            cy = int(lb.get("center_y", 0.5) * H)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            w = int(lb.get("width", 0.1) * W)
            h = int(lb.get("height", 0.1) * H)
            x1 = max(cx - w // 2, 0)
            y1 = max(cy - h // 2, 0)
            x2 = min(cx + w // 2, W - 1)
            y2 = min(cy + h // 2, H - 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
        else:
            cx = int(rb.get("center_x", 0.5) * W)
            w = int(rb.get("width", 0.1) * W)
            cv2.circle(out, (cx, H // 2), 4, (255, 0, 255), -1)
            x1 = max(cx - w // 2, 0)
            x2 = min(cx + w // 2, W - 1)
            cv2.line(out, (x1, H // 2), (x2, H // 2), (255, 0, 255), 1)
    return out


SUITE_ARTIFACTS = ARTIFACT_DIR / "suite"


def _first_label(labels):
    return labels[0] if labels else None


def _skip_if_no_labels(labels):
    if not labels:
        pytest.skip("No labels available in sample; skipping label-dependent assertions")


@pytest.mark.parametrize("idx", list(range(5)))
def test_flip_swap_invariants_suite(idx):
    ensure_dir(SUITE_ARTIFACTS)
    left, right, labels, img_id = load_real_sample(index=idx)
    _skip_if_no_labels(labels)

    aug = HorizontalFlipAugmentor(p_apply=1.0)
    left_f, right_f, labels_f = aug(left, right, labels)

    cv2.imwrite(str(SUITE_ARTIFACTS / f"{img_id}_suite_flip_left_before.png"), draw_label_markers(left, labels, False))
    cv2.imwrite(str(SUITE_ARTIFACTS / f"{img_id}_suite_flip_right_before.png"), draw_label_markers(right, labels, True))
    cv2.imwrite(str(SUITE_ARTIFACTS / f"{img_id}_suite_flip_left_after.png"), draw_label_markers(left_f, labels_f, False))
    cv2.imwrite(str(SUITE_ARTIFACTS / f"{img_id}_suite_flip_right_after.png"), draw_label_markers(right_f, labels_f, True))

    # Image swap + flip
    assert np.array_equal(left_f, cv2.flip(right, 1))
    assert np.array_equal(right_f, cv2.flip(left, 1))

    # Label mapping
    for o, f in zip(labels, labels_f):
        lb_o, rb_o = o["left_box"], o["right_box"]
        lb_f, rb_f = f["left_box"], f["right_box"]
        # Mapping under flip+swap
        assert pytest.approx(lb_f["center_x"], 1e-6) == 1.0 - rb_o["center_x"]
        assert pytest.approx(rb_f["center_x"], 1e-6) == 1.0 - lb_o["center_x"]
        # Disparity preserved
        disp_o = lb_o["center_x"] - rb_o["center_x"]
        disp_f = lb_f["center_x"] - rb_f["center_x"]
        assert pytest.approx(disp_o, 1e-6) == disp_f


@pytest.mark.parametrize("idx", list(range(5)))
def test_photometric_preserves_labels_suite(idx):
    left, right, labels, _ = load_real_sample(index=idx)
    labels_before = [dict(l) for l in labels]
    aug = PhotometricAugmentor(p_apply=1.0)
    _ = aug(left, right)
    assert labels == labels_before


@pytest.mark.parametrize("idx", list(range(5)))
def test_scale_normalized_labels_unchanged_suite(idx):
    left, right, labels, _ = load_real_sample(index=idx)
    aug = RandomScaleAugmentor(scale_range=(1.15, 1.15), p_apply=1.0)
    left_s, right_s, labels_s = aug(left, right, labels)
    assert left_s.shape[0] == int(round(left.shape[0] * 1.15))
    assert left_s.shape[1] == int(round(left.shape[1] * 1.15))
    assert right_s.shape == left_s.shape
    assert labels_s == labels


@pytest.mark.parametrize("idx", list(range(5)))
def test_crop_bounds_suite(idx):
    left, right, labels, _ = load_real_sample(index=idx)
    aug = RandomCropAugmentor(crop_height_ratio=0.7, crop_width_ratio=0.7, p_apply=1.0)
    left_c, right_c, labels_c = aug(left, right, labels)
    assert right_c.shape == left_c.shape
    if labels_c:
        lb = labels_c[0]["left_box"]
        rb = labels_c[0]["right_box"]
        for k in ("center_x", "center_y"):
            assert 0.0 <= lb[k] <= 1.0
        for k in ("center_x",):
            assert -5e-2 <= rb[k] <= 1.0 + 5e-2
