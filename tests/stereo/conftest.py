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
    """Create a sample 3D bounding box for testing."""
    return Box3D(
        center_3d=(10.0, 2.0, 30.0),
        dimensions=(3.88, 1.63, 1.53),  # Car dimensions [L, W, H]
        orientation=0.0,
        class_label=0,  # Car
        confidence=0.95,
        bbox_2d=(100, 100, 200, 200),  # [x1, y1, x2, y2]
    )


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

