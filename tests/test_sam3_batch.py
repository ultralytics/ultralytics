# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest

from ultralytics.utils import ASSETS, ASSETS_URL, ONLINE, WEIGHTS_DIR

# SAM3 weights path - requires Meta license agreement to download
SAM3_WEIGHTS = WEIGHTS_DIR / "sam3_s.pt"
# Test video URL
VIDEO_URL = f"{ASSETS_URL}/decelera_portrait_min.mov"


@pytest.mark.slow
@pytest.mark.skipif(not SAM3_WEIGHTS.exists(), reason="SAM3 weights not available (requires Meta license)")
class TestSAM3BatchInference:
    """Test SAM3 batch inference functionality."""

    @pytest.fixture
    def predictor(self):
        """Create SAM3 semantic predictor."""
        from ultralytics.models.sam import SAM3SemanticPredictor

        predictor = SAM3SemanticPredictor(overrides={"model": SAM3_WEIGHTS})
        predictor.setup_model()
        return predictor

    @pytest.fixture
    def test_images(self):
        """Get test image paths."""
        return [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"]

    # Legacy single-image API tests
    def test_legacy_set_image_then_call(self, predictor, test_images):
        """Test legacy API: set_image() then __call__(text=...)."""
        predictor.set_image(test_images[0])
        result = predictor(text=["person", "car"])
        assert result is not None

    def test_legacy_set_image_with_bboxes(self, predictor, test_images):
        """Test legacy API with bboxes."""
        predictor.set_image(test_images[0])
        result = predictor(bboxes=[[100, 100, 300, 300]])
        assert result is not None

    # New batch API tests
    def test_batch_same_text_all_images(self, predictor, test_images):
        """Test batch API with same text prompts for all images."""
        results = predictor(test_images, text=["person", "car"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_per_image_text(self, predictor, test_images):
        """Test batch API with different text prompts per image."""
        results = predictor(test_images, text=[["person"], ["person", "tie"]])
        assert len(results) == 2

    def test_batch_with_batch_size(self, predictor, test_images):
        """Test batch API with explicit batch_size."""
        results = predictor(test_images, text=["person"], batch_size=2)
        assert len(results) == 2

    def test_batch_single_image_returns_single_result(self, predictor, test_images):
        """Test that single image returns single Result, not list."""
        result = predictor(test_images[0], text=["person"])
        # Single image should return single Result, not list
        from ultralytics.engine.results import Results

        assert isinstance(result, Results)

    def test_batch_many_images(self, predictor, test_images):
        """Test batch API with many images."""
        images = test_images * 4  # 8 images
        results = predictor(images, text=["person"], batch_size=4)
        assert len(results) == 8

    def test_original_set_image_unchanged(self, predictor, test_images):
        """Ensure original single-image set_image API still works."""
        predictor.set_image(test_images[0])
        assert predictor.features is not None

    def test_inference_features_api(self, predictor, test_images):
        """Test inference_features API for feature reuse."""
        import cv2

        predictor.set_image(test_images[0])
        src_shape = cv2.imread(str(test_images[0])).shape[:2]
        masks, _boxes = predictor.inference_features(predictor.features, src_shape=src_shape, text=["person"])
        # Should return masks and boxes (may be None if no detections)
        assert masks is None or hasattr(masks, "shape")


@pytest.mark.slow
@pytest.mark.skipif(not SAM3_WEIGHTS.exists(), reason="SAM3 weights not available (requires Meta license)")
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
class TestSAM3VideoInference:
    """Test SAM3 video semantic predictor functionality."""

    @pytest.fixture
    def video_predictor(self):
        """Create SAM3 video semantic predictor."""
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        predictor = SAM3VideoSemanticPredictor(overrides={"model": SAM3_WEIGHTS})
        predictor.setup_model()
        return predictor

    def test_video_inference_with_text(self, video_predictor):
        """Test video inference with text prompts."""
        results = list(video_predictor(VIDEO_URL, text=["person"], stream=True))
        assert len(results) > 0
        # Each result should be a Results object
        from ultralytics.engine.results import Results

        assert all(isinstance(r, Results) for r in results)

    def test_video_inference_stream(self, video_predictor):
        """Test video streaming mode."""
        count = 0
        for result in video_predictor(VIDEO_URL, text=["person"], stream=True):
            count += 1
            if count >= 3:  # Process at least 3 frames
                break
        assert count >= 3

    def test_video_tracking_multiple_classes(self, video_predictor):
        """Test video tracking with multiple text classes."""
        results = []
        for result in video_predictor(VIDEO_URL, text=["person", "car"], stream=True):
            results.append(result)
            if len(results) >= 5:
                break
        assert len(results) >= 5
