import os
from pathlib import Path

import numpy as np
import cv2
import pytest

from ultralytics.models.yolo.stereo3ddet.augment import (
    PhotometricAugmentor,
    HorizontalFlipAugmentor,
    RandomScaleAugmentor,
    RandomCropAugmentor,
    StereoAugmentationPipeline,
)
from ultralytics.data.kitti_stereo import KITTIStereoDataset


ARTIFACT_DIR = Path("/root/ultralytics/tests/artifacts/stereo_aug")
DATASET_ROOT = Path("/root/autodl-tmp/converted_kitti_3dop")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_real_sample(index: int = 0, split: str = "train"):
    """Load a real KITTI stereo sample via KITTIStereoDataset.

    Falls back to synthetic if dataset not present.
    """
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


@pytest.mark.parametrize("seed", [7])
def test_photometric_augmentor(seed):
    ensure_dir(ARTIFACT_DIR)
    left, right, labels, img_id = load_real_sample()
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_photometric_left_before.png"), left)
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_photometric_right_before.png"), right)

    aug = PhotometricAugmentor(p_apply=1.0, seed=seed)
    left_a, right_a = aug(left, right)

    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_photometric_left_after.png"), left_a)
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_photometric_right_after.png"), right_a)

    assert left_a.shape == left.shape and right_a.shape == right.shape
    assert not np.array_equal(left_a, left) or not np.array_equal(right_a, right)

class TestHorizontalFlipAugmentor:

    def test_disparity_preserved(self):
        left, right, labels, _ = load_real_sample()
        aug = HorizontalFlipAugmentor(p_apply=1.0)
        left_f, right_f, labels_f = aug(left, right, labels)

        for obj_o, obj_f in zip(labels, labels_f):
            lb_o = obj_o["left_box"]
            rb_o = obj_o["right_box"]
            lb_f = obj_f["left_box"]
            rb_f = obj_f["right_box"]

            disp_o = (lb_o["center_x"] - rb_o["center_x"])
            disp_f = (lb_f["center_x"] - rb_f["center_x"])
            # Option A (flip + swap): disparity preserved
            assert pytest.approx(disp_o, 1e-6) == disp_f
    
    def test_height_consistency(self):
        left, right, labels, _ = load_real_sample()
        aug = HorizontalFlipAugmentor(p_apply=1.0)
        left_f, right_f, labels_f = aug(left, right, labels)

        for obj_o, obj_f in zip(labels, labels_f):
            lb_o = obj_o["left_box"]
            lb_f = obj_f["left_box"]

            assert pytest.approx(lb_o["center_y"], 1e-6) == lb_f["center_y"]
            assert pytest.approx(lb_o["height"], 1e-6) == lb_f["height"]
    
    def test_depth_preserved(self):
        left, right, labels, _ = load_real_sample()
        aug = HorizontalFlipAugmentor(p_apply=1.0)
        left_f, right_f, labels_f = aug(left, right, labels)

        for obj_o, obj_f in zip(labels, labels_f):
            lb_o = obj_o["left_box"]
            rb_o = obj_o["right_box"]
            lb_f = obj_f["left_box"]
            rb_f = obj_f["right_box"]

            disp_o = (lb_o["center_x"] - rb_o["center_x"])
            disp_f = (lb_f["center_x"] - rb_f["center_x"])
            # Option A: disparity preserved => depth ~ 1/disp preserved
            if disp_o == 0 and disp_f == 0:
                continue
            depth_o = float('inf') if disp_o == 0 else (1.0 / disp_o)
            depth_f = float('inf') if disp_f == 0 else (1.0 / disp_f)
            assert pytest.approx(depth_o, 1e-6) == depth_f
    
    def test_3d_left_right_mapping(self):
        left, right, labels, _ = load_real_sample()
        aug = HorizontalFlipAugmentor(p_apply=1.0)
        left_f, right_f, labels_f = aug(left, right, labels)

        for obj_o, obj_f in zip(labels, labels_f):
            lb_o = obj_o["left_box"]
            rb_o = obj_o["right_box"]
            lb_f = obj_f["left_box"]
            rb_f = obj_f["right_box"]
            # Option A: flip + swap mapping
            assert pytest.approx(lb_f["center_x"], 1e-6) == 1.0 - rb_o["center_x"]
            assert pytest.approx(rb_f["center_x"], 1e-6) == 1.0 - lb_o["center_x"]
    
    def test_principal_point_changes(self):
        # Option A mapping for centers under flip + swap
        left, right, labels, _ = load_real_sample()
        W = left.shape[1]
        aug = HorizontalFlipAugmentor(p_apply=1.0)
        left_f, right_f, labels_f = aug(left, right, labels)
        for obj_o, obj_f in zip(labels, labels_f):
            lb_o = obj_o["left_box"]
            lb_f = obj_f["left_box"]
            rb_o = obj_o["right_box"]
            rb_f = obj_f["right_box"]
            assert pytest.approx(lb_f["center_x"], 1e-6) == 1.0 - rb_o["center_x"]
            assert pytest.approx(rb_f["center_x"], 1e-6) == 1.0 - lb_o["center_x"]
    

def test_horizontal_flip_augmentor():
    ensure_dir(ARTIFACT_DIR)
    left, right, labels, img_id = load_real_sample()
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_hflip_left_before.png"), draw_label_markers(left, labels, is_right=False))
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_hflip_right_before.png"), draw_label_markers(right, labels, is_right=True))

    aug = HorizontalFlipAugmentor(p_apply=1.0)
    left_f, right_f, labels_f = aug(left, right, labels)

    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_hflip_left_after.png"), draw_label_markers(left_f, labels_f, is_right=False))
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_hflip_right_after.png"), draw_label_markers(right_f, labels_f, is_right=True))

    assert np.array_equal(left_f, cv2.flip(right, 1))
    assert np.array_equal(right_f, cv2.flip(left, 1))

    lb = labels[0]["left_box"]
    rb = labels[0]["right_box"]
    lb_f = labels_f[0]["left_box"]
    rb_f = labels_f[0]["right_box"]
    assert pytest.approx(lb_f["center_x"], 1e-6) == 1.0 - rb["center_x"]
    assert pytest.approx(rb_f["center_x"], 1e-6) == 1.0 - lb["center_x"]

class TestScaleAugmentor:

    def test_baseline_unchanged(self):
        # baseline should be the same after scale=1.0
        left, right, labels, _ = load_real_sample()
        aug = RandomScaleAugmentor(scale_range=(1.0, 1.0), p_apply=1.0)
        left_s, right_s, labels_s = aug(left, right, labels)
        assert left_s.shape == left.shape
        assert right_s.shape == right.shape
        assert labels_s == labels

    def test_focal_length_scaled(self):
        # f'_x = s * f_x
        pass

    def test_depth_preserved_after_scale(self):
        # z' = (s * f_x * B) / (s * disparity) = z
        pass

def test_random_scale_augmentor():
    ensure_dir(ARTIFACT_DIR)
    left, right, labels, img_id = load_real_sample()
    aug = RandomScaleAugmentor(scale_range=(1.2, 1.2), p_apply=1.0)
    left_s, right_s, labels_s = aug(left, right, labels)
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_rscale_left_after.png"), draw_label_markers(left_s, labels_s, is_right=False))
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_rscale_right_after.png"), draw_label_markers(right_s, labels_s, is_right=True))

    assert left_s.shape[0] == int(round(left.shape[0] * 1.2))
    assert left_s.shape[1] == int(round(left.shape[1] * 1.2))
    assert right_s.shape == left_s.shape
    assert labels_s == labels


def test_random_crop_augmentor():
    ensure_dir(ARTIFACT_DIR)
    left, right, labels, img_id = load_real_sample()
    np.random.seed(123)
    aug = RandomCropAugmentor(crop_height_ratio=0.5, crop_width_ratio=0.5, p_apply=1.0)
    left_c, right_c, labels_c = aug(left, right, labels)
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_rcrop_left_after.png"), draw_label_markers(left_c, labels_c, is_right=False))
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_rcrop_right_after.png"), draw_label_markers(right_c, labels_c, is_right=True))

    expected_h = int(round(left.shape[0] * 0.5))
    expected_w = int(round(left.shape[1] * 0.5))
    assert left_c.shape[0] == expected_h and left_c.shape[1] == expected_w
    assert right_c.shape == left_c.shape
    lb_c = labels_c[0]["left_box"]
    rb_c = labels_c[0]["right_box"]
    assert 0.0 <= lb_c["center_x"] <= 1.0
    assert 0.0 <= lb_c["center_y"] <= 1.0
    assert 0.0 <= rb_c["center_x"] <= 1.0


def test_stereo_augmentation_pipeline():
    ensure_dir(ARTIFACT_DIR)
    left, right, labels, img_id = load_real_sample()
    pipe = StereoAugmentationPipeline(
        photometric=PhotometricAugmentor(p_apply=0.0),
        hflip=HorizontalFlipAugmentor(p_apply=1.0),
        rscale=RandomScaleAugmentor(scale_range=(1.0, 1.0), p_apply=1.0),
        rcrop=RandomCropAugmentor(p_apply=0.0),
    )
    left_o, right_o, labels_o, _ = pipe.augment(left, right, labels, calibration=None)

    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_pipeline_left_after.png"), draw_label_markers(left_o, labels_o, is_right=False))
    cv2.imwrite(str(ARTIFACT_DIR / f"{img_id}_pipeline_right_after.png"), draw_label_markers(right_o, labels_o, is_right=True))

    assert np.array_equal(left_o, cv2.flip(right, 1))
    assert np.array_equal(right_o, cv2.flip(left, 1))
