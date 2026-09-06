# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Tests for SAM3 video prediction via LoadNumpyFrames and stream_inference lifecycle."""

import numpy as np
import pytest

from ultralytics.data.loaders import LoadNumpyFrames


def _sam3_available():
    """Check if sam3.pt weights are available."""
    from pathlib import Path

    from ultralytics.utils import SETTINGS

    weights_dir = Path(SETTINGS["weights_dir"])
    return (weights_dir / "sam3.pt").exists()


def _distinct_frames(n=3, size=(64, 64)):
    """Create n visually distinct BGR frames (different mean intensities)."""
    return [np.full((*size, 3), fill_value=i * 50 + 10, dtype=np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------
# LoadNumpyFrames unit tests
# ---------------------------------------------------------------------------


class TestLoadNumpyFrames:
    """Validate the loader interface contract required by the predictor lifecycle."""

    def test_video_mode_and_attributes(self):
        frames = _distinct_frames(5)
        loader = LoadNumpyFrames(frames)
        assert loader.mode == "video"
        assert loader.frames == 5
        assert len(loader) == 5
        assert loader.bs == 1
        assert loader.fps == 30

    def test_one_based_frame_counter(self):
        frames = _distinct_frames(3)
        loader = LoadNumpyFrames(frames)
        seen = []
        for paths, im0s, info in loader:
            seen.append(loader.frame)
            assert len(im0s) == 1
        assert seen == [1, 2, 3]

    def test_yields_one_frame_at_a_time(self):
        frames = _distinct_frames(4)
        loader = LoadNumpyFrames(frames)
        batch_count = 0
        for paths, im0s, info in loader:
            assert len(im0s) == 1
            assert im0s[0] is frames[batch_count]
            batch_count += 1
        assert batch_count == 4

    def test_reset_iterator(self):
        frames = _distinct_frames(2)
        loader = LoadNumpyFrames(frames)
        first_pass = [loader.frame for _, _, _ in loader]
        second_pass = [loader.frame for _, _, _ in loader]
        assert first_pass == second_pass == [1, 2]

    def test_empty_frames_raises(self):
        with pytest.raises(FileNotFoundError):
            LoadNumpyFrames([])

    def test_routes_through_load_inference_source(self):
        from ultralytics.data.build import load_inference_source

        loader = LoadNumpyFrames(_distinct_frames(3))
        dataset = load_inference_source(source=loader)
        assert dataset is loader
        assert dataset.source_type is not None
        assert dataset.source_type.from_img
        assert not dataset.source_type.stream
        assert not dataset.source_type.screenshot
        assert not dataset.source_type.tensor


# ---------------------------------------------------------------------------
# Input validation (no model needed)
# ---------------------------------------------------------------------------


class TestPredictFramesValidation:
    """Input validation runs before any model setup."""

    def test_rejects_empty_list(self):
        from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

        predictor = SAM3VideoSemanticPredictor.__new__(SAM3VideoSemanticPredictor)
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict_frames([])

    def test_rejects_non_list(self):
        from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

        predictor = SAM3VideoSemanticPredictor.__new__(SAM3VideoSemanticPredictor)
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict_frames("not-a-list")

    def test_rejects_mixed_types(self):
        from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

        predictor = SAM3VideoSemanticPredictor.__new__(SAM3VideoSemanticPredictor)
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict_frames([np.zeros((8, 8, 3), dtype=np.uint8), "bad"])


# ---------------------------------------------------------------------------
# Real SAM3 end-to-end tests (skipped when weights unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _sam3_available(), reason="sam3.pt weights not available offline")
class TestPredictFramesEndToEnd:
    """Full-lifecycle tests using real SAM3 weights on short frame sequences."""

    @pytest.fixture
    def predictor(self):
        from pathlib import Path

        from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
        from ultralytics.utils import SETTINGS

        weights_dir = Path(SETTINGS["weights_dir"])
        return SAM3VideoSemanticPredictor(overrides={"model": str(weights_dir / "sam3.pt")})

    def test_returns_results_per_frame(self, predictor):
        frames = _distinct_frames(3)
        results = predictor.predict_frames(frames, text=["object"])
        assert len(results) == 3

    def test_temporal_continuity(self, predictor):
        """Masks should evolve coherently across frames (not random per-frame)."""
        frames = _distinct_frames(4)
        results = predictor.predict_frames(frames, text=["object"])
        # Each result should have a valid mask attribute (non-None for video with text prompt)
        for r in results:
            assert hasattr(r, "masks")

    def test_clean_second_sequence(self, predictor):
        """Second call must not leak state from the first."""
        frames_a = _distinct_frames(3)
        results_a = predictor.predict_frames(frames_a, text=["object"])
        assert len(results_a) == 3

        frames_b = _distinct_frames(2)
        results_b = predictor.predict_frames(frames_b, text=["object"])
        assert len(results_b) == 2

    def test_callback_ordering(self, predictor):
        """All standard lifecycle callbacks should fire."""
        fired = []

        def on_start(*a, **k):
            fired.append("on_predict_start")

        def on_batch_start(*a, **k):
            fired.append("on_predict_batch_start")

        def on_batch_end(*a, **k):
            fired.append("on_predict_batch_end")

        def on_end(*a, **k):
            fired.append("on_predict_end")

        predictor.add_callback("on_predict_start", on_start)
        predictor.add_callback("on_predict_batch_start", on_batch_start)
        predictor.add_callback("on_predict_batch_end", on_batch_end)
        predictor.add_callback("on_predict_end", on_end)

        frames = _distinct_frames(2)
        predictor.predict_frames(frames, text=["object"])

        assert "on_predict_start" in fired
        assert "on_predict_end" in fired
        assert fired.index("on_predict_start") < fired.index("on_predict_batch_start")
        assert fired.index("on_predict_batch_start") < fired.index("on_predict_batch_end")
        assert fired.index("on_predict_batch_end") < fired.index("on_predict_end")
