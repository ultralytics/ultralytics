import numpy as np

from ultralytics.data.augment import DepthColorJitter, DepthFormat, DepthRandomFlip, RandomPerspective


def test_depth_transforms_live_in_augment():
    from ultralytics.data.augment import DepthFormat, DepthRandomFlip, DepthColorJitter  # noqa: F401

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


def test_random_perspective_warps_depth_with_nearest_interpolation():
    """RandomPerspective should warp depth without inventing intermediate sparse values.

    Labels carry empty cls/instances like DepthDataset provides (depth label files are
    backgrounds) — RandomPerspective.apply_instances requires both keys unconditionally.
    """
    from ultralytics.utils.instance import Instances

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = _sparse_depth(16, 16, val=10.0)
    labels = {
        "img": img.copy(),
        "depth": depth.copy(),
        "cls": np.zeros((0, 1), dtype=np.float32),
        "instances": Instances(np.zeros((0, 4), dtype=np.float32), segments=np.zeros((0, 1000, 2), dtype=np.float32)),
    }
    transform = RandomPerspective(
        degrees=0.0,
        translate=0.0,
        scale=(1.0, 1.0),
        shear=0.0,
        perspective=0.0,
        size=(32, 32),
    )
    out = transform(labels)
    assert out["img"].shape[:2] == out["depth"].shape[:2] == (32, 32)
    blended = ((out["depth"] > 1e-6) & (out["depth"] < 9.0)).sum()
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation"
