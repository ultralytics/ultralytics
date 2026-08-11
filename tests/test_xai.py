from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from ultralytics.utils.xai import YOLO_XAI_Extractor, generate_gradcam_heatmap, validate_heatmap


def test_generate_gradcam_heatmap():
    """Test Grad-CAM generation, output shape, and normalization."""
    activations = torch.randn(1, 4, 8, 8)
    gradients = torch.randn(1, 4, 8, 8)

    heatmap = generate_gradcam_heatmap(activations, gradients, image_shape=(32, 32))

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (32, 32)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_generate_gradcam_heatmap_removes_negative_values():
    """Test that negative Grad-CAM activations are removed by ReLU."""
    activations = -torch.ones(1, 2, 4, 4)
    gradients = torch.ones(1, 2, 4, 4)

    heatmap = generate_gradcam_heatmap(activations, gradients, image_shape=(16, 16))

    assert np.allclose(heatmap, 0.0)


class DummyNetwork(nn.Module):
    """Small network used to test XAI hooks without loading a YOLO checkpoint."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class DummyYOLO:
    """Minimal YOLO-like wrapper exposing a .model attribute."""

    def __init__(self):
        self.model = DummyNetwork()


def test_yolo_xai_extractor_hooks():
    """Test XAI forward/backward hooks and hook removal."""
    model = DummyYOLO()
    extractor = YOLO_XAI_Extractor(model, target_layer_index=-2)

    assert extractor.activations is None
    assert extractor.gradients is None

    x = torch.randn(1, 3, 16, 16, requires_grad=True)

    output = extractor(x)

    assert extractor.activations is not None
    assert output.shape[0] == 1

    output.sum().backward()

    assert extractor.gradients is not None
    assert extractor.activations.shape == extractor.gradients.shape

    assert len(extractor.target_layer._forward_hooks) > 0
    assert len(extractor.target_layer._backward_hooks) > 0

    extractor.remove_hooks()

    assert len(extractor.target_layer._forward_hooks) == 0
    assert len(extractor.target_layer._backward_hooks) == 0


class FakeBoxes:
    """Minimal boxes object used to test confidence extraction."""

    def __init__(self, confidence=0.8, class_id=0):
        self.conf = torch.tensor([confidence], dtype=torch.float32)
        self.cls = torch.tensor([class_id], dtype=torch.float32)


class FakeModel:
    """Minimal model returning deterministic detection results."""

    def __call__(self, image, verbose=False):
        mean_value = float(image.mean()) / 255.0
        confidence = max(0.0, min(1.0, mean_value))

        result = SimpleNamespace(boxes=FakeBoxes(confidence=confidence, class_id=0))
        return [result]


def test_validate_heatmap():
    """Test Deletion and Insertion faithfulness evaluation."""
    model = FakeModel()

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[8:24, 8:24] = 1.0

    audc, auic = validate_heatmap(
        model=model,
        img_bgr=image,
        heatmap=heatmap,
        target_class_idx=0,
    )

    assert np.isfinite(audc)
    assert np.isfinite(auic)
    assert 0.0 <= audc <= 1.0
    assert 0.0 <= auic <= 1.0


def test_validate_heatmap_with_different_dimensions():
    """Test heatmap resizing when image and heatmap dimensions differ."""
    model = FakeModel()

    image = np.full((64, 48, 3), 150, dtype=np.uint8)
    heatmap = np.random.default_rng(0).random((16, 16), dtype=np.float32)

    audc, auic = validate_heatmap(
        model=model,
        img_bgr=image,
        heatmap=heatmap,
        target_class_idx=0,
    )

    assert np.isfinite(audc)
    assert np.isfinite(auic)


def test_validate_heatmap_no_target_class():
    """Test that missing target classes produce zero confidence."""

    class EmptyTargetModel:
        def __call__(self, image, verbose=False):
            boxes = FakeBoxes(confidence=0.8, class_id=1)
            return [SimpleNamespace(boxes=boxes)]

    model = EmptyTargetModel()

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    heatmap = np.ones((32, 32), dtype=np.float32)

    audc, auic = validate_heatmap(
        model=model,
        img_bgr=image,
        heatmap=heatmap,
        target_class_idx=0,
    )

    assert audc == pytest.approx(0.0)
    assert auic == pytest.approx(0.0)


def test_validate_heatmap_empty_boxes():
    """Test that images with no detections return zero confidence."""

    class EmptyBoxesModel:
        def __call__(self, image, verbose=False):
            boxes = SimpleNamespace(
                conf=torch.empty(0),
                cls=torch.empty(0),
            )
            return [SimpleNamespace(boxes=boxes)]

    model = EmptyBoxesModel()

    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    heatmap = np.ones((32, 32), dtype=np.float32)

    audc, auic = validate_heatmap(
        model=model,
        img_bgr=image,
        heatmap=heatmap,
        target_class_idx=0,
    )

    assert audc == pytest.approx(0.0)
    assert auic == pytest.approx(0.0)
