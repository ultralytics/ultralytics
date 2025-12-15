# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Pytest configuration and fixtures for Stereo 3D Object Detection tests."""

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.data.stereo.pair import StereoImagePair


@pytest.fixture
def sample_calibration():
    """Create a sample KITTI-style calibration for testing."""
    return CalibrationParameters(
        fx=721.5377,
        fy=721.5377,
        cx=609.5593,
        cy=172.8540,
        baseline=0.54,
        image_width=1242,
        image_height=375,
    )


@pytest.fixture
def sample_calibration_dict(sample_calibration):
    """Provide calibration parameters as a plain dict for functions that expect mapping inputs."""
    return {
        "fx": sample_calibration.fx,
        "fy": sample_calibration.fy,
        "cx": sample_calibration.cx,
        "cy": sample_calibration.cy,
        "baseline": sample_calibration.baseline,
        "image_width": sample_calibration.image_width,
        "image_height": sample_calibration.image_height,
    }


@pytest.fixture
def sample_stereo_pair(sample_calibration):
    """Create a sample stereo image pair for testing."""
    left_img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    return StereoImagePair(
        left_image=left_img,
        right_image=right_img,
        image_id="000000",
        calibration=sample_calibration,
    )


@pytest.fixture
def sample_box3d():
    """Create a canonical Car box for tests that need a single Box3D."""
    return Box3D(
        center_3d=(10.0, 2.0, 30.0),
        dimensions=(3.88, 1.63, 1.53),  # Car dimensions [L, W, H]
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
        bbox_2d=(100.0, 100.0, 200.0, 200.0),  # [x1, y1, x2, y2]
    )


@pytest.fixture
def sample_boxes3d():
    """Create a heterogeneous list of Box3D instances for visualization tests."""
    return [
        Box3D(
            center_3d=(10.0, 1.8, 28.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.15,
            class_label="Car",
            class_id=0,
            confidence=0.92,
            bbox_2d=(90.0, 110.0, 210.0, 220.0),
        ),
        Box3D(
            center_3d=(6.0, 1.6, 18.0),
            dimensions=(0.8, 0.6, 1.7),
            orientation=-0.4,
            class_label="Pedestrian",
            class_id=1,
            confidence=0.81,
            bbox_2d=(140.0, 120.0, 160.0, 200.0),
        ),
        Box3D(
            center_3d=(12.5, 1.7, 32.0),
            dimensions=(1.7, 0.6, 1.8),
            orientation=0.9,
            class_label="Cyclist",
            class_id=2,
            confidence=0.88,
            bbox_2d=(220.0, 130.0, 260.0, 210.0),
        ),
    ]


@pytest.fixture
def sample_stereo_batch():
    """Create a sample batch of stereo images for testing."""
    batch_size = 2
    height, width = 96, 320  # Downsampled size
    # 6 channels: RGB left (3) + RGB right (3)
    return torch.randn(batch_size, 6, height, width)


@pytest.fixture
def device():
    """Get the device for testing (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

