import numpy as np

from ultralytics.data.augment import DepthColorJitter, DepthFormat, DepthRandomFlip, DepthRandomScale


def test_depth_transforms_live_in_augment():
    from ultralytics.data.augment import DepthFormat, DepthRandomFlip, DepthRandomScale, DepthColorJitter  # noqa: F401

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    out = DepthFormat()({"img": img, "depth": depth})
    assert out["img"].shape == (3, 32, 32)            # CHW
    assert tuple(out["depth"].shape) == (1, 32, 32)   # (1,H,W)
    assert out["depth"].dtype.is_floating_point


def test_depth_random_flip_flips_both():
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    depth = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    out = DepthRandomFlip(p=1.0)({"img": img.copy(), "depth": depth.copy()})
    assert np.array_equal(out["img"], np.ascontiguousarray(np.fliplr(img)))
    assert np.array_equal(out["depth"], np.ascontiguousarray(np.fliplr(depth)))


def test_depth_random_scale_outputs_target_size():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    depth = np.ones((40, 40), dtype=np.float32)
    out = DepthRandomScale(scale_range=(1.5, 1.5), target_size=32, p=1.0)({"img": img, "depth": depth})
    assert out["img"].shape[:2] == out["depth"].shape[:2]


def test_depth_color_jitter_preserves_shape_and_dtype():
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    depth = np.ones((16, 16), dtype=np.float32)
    out = DepthColorJitter()({"img": img, "depth": depth})
    assert out["img"].shape == (16, 16, 3) and out["img"].dtype == np.uint8
    assert np.array_equal(out["depth"], depth)   # depth unchanged by color jitter


def _sparse_depth(h, w, val=10.0):
    """Sparse depth map: mostly zero (invalid) with a block of valid metric depth."""
    d = np.zeros((h, w), dtype=np.float32)
    d[h // 4 : h // 2, w // 4 : w // 2] = val
    return d


def test_depth_format_resize_does_not_blend_sparse_depth():
    """Resizing sparse depth must not create intermediate near-zero values (no bilinear blend)."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)        # forces depth resize 16->32
    depth = _sparse_depth(16, 16, val=10.0)
    out = DepthFormat()({"img": img, "depth": depth})["depth"].numpy()
    blended = ((out > 1e-6) & (out < 9.0)).sum()       # values between background(0) and valid(10)
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation"


def test_depth_random_scale_does_not_blend_sparse_depth():
    """DepthRandomScale resize must preserve sparse-depth values (no bilinear blend)."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    depth = _sparse_depth(40, 40, val=10.0)
    out = DepthRandomScale(scale_range=(1.5, 1.5), target_size=32, p=1.0)({"img": img, "depth": depth})["depth"]
    blended = ((out > 1e-6) & (out < 9.0)).sum()
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation"
