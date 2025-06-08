# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import pytest

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import Metric


class TestSizeSpecificMetrics:
    """Test size-specific mAP metrics functionality."""

    def test_metric_initialization(self):
        """Test that size-specific metrics initialize correctly."""
        metric = Metric()

        # Test default initialization
        assert hasattr(metric, "size_specific_metrics")
        assert metric.size_specific_metrics == [0.0, 0.0, 0.0]

        # Test properties
        assert metric.map_small == 0.0
        assert metric.map_medium == 0.0
        assert metric.map_large == 0.0

    def test_update_size_metrics(self):
        """Test updating size-specific metrics."""
        metric = Metric()

        # Update with test values
        metric.update_size_metrics(0.45, 0.67, 0.89)

        # Verify updates
        assert metric.map_small == 0.45
        assert metric.map_medium == 0.67
        assert metric.map_large == 0.89
        assert metric.size_specific_metrics == [0.45, 0.67, 0.89]

    def test_properties_return_correct_values(self):
        """Test that properties return correct values from size_specific_metrics."""
        metric = Metric()

        # Set values directly
        metric.size_specific_metrics = [0.1, 0.2, 0.3]

        # Test properties return correct values
        assert metric.map_small == 0.1
        assert metric.map_medium == 0.2
        assert metric.map_large == 0.3

    @pytest.mark.skipif(not check_requirements("pycocotools", install=False), reason="pycocotools not installed")
    def test_coco_size_metrics_integration(self):
        """Test that COCO validation populates size-specific metrics."""
        from ultralytics.models.yolo.detect import DetectionValidator

        # Setup validation with COCO dataset (using coco8 for testing)
        args = {"model": "yolo11n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
        validator = DetectionValidator(args=args)
        validator()

        # Check that size metrics are accessible
        assert hasattr(validator.metrics.box, "map_small")
        assert hasattr(validator.metrics.box, "map_medium")
        assert hasattr(validator.metrics.box, "map_large")

        # For COCO datasets, should be able to get real values or 0.0
        # (values might be 0.0 if no objects of that size in test dataset)
        assert isinstance(validator.metrics.box.map_small, float)
        assert isinstance(validator.metrics.box.map_medium, float)
        assert isinstance(validator.metrics.box.map_large, float)

    def test_non_coco_defaults(self):
        """Test that non-COCO datasets default to 0.0."""
        # This tests the default behavior without COCO evaluation
        metric = Metric()

        # Should remain at defaults for non-COCO
        assert metric.map_small == 0.0
        assert metric.map_medium == 0.0
        assert metric.map_large == 0.0
