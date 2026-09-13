# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for SAM3 video prediction via LoadNumpyFrames and the stream_inference lifecycle."""

import cv2
import numpy as np
import pytest

from ultralytics.data.loaders import LoadNumpyFrames
from ultralytics.utils import ASSETS, WEIGHTS_DIR


def _sam3_available():
    """Check if sam3.pt weights are available."""
    return (WEIGHTS_DIR / "sam3.pt").exists()


def _distinct_frames(n=3, size=(64, 64)):
    """Create n distinct BGR frames from a real image with a moving marker, mimicking a short sequence."""
    base = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), size)
    frames = []
    for i in range(n):
        frame = base.copy()
        x = (i + 1) * size[1] // (n + 1)
        cv2.rectangle(frame, (x - 5, 10), (x + 5, 30), (0, 0, 255), -1)
        frames.append(frame)
    return frames


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


@pytest.mark.skipif(not _sam3_available(), reason="sam3.pt weights not available offline")
class TestPredictFramesEndToEnd:
    """Full-lifecycle tests using real SAM3 weights on short frame sequences."""

    @pytest.fixture
    def predictor(self):
        from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

        return SAM3VideoSemanticPredictor(overrides={"model": str(WEIGHTS_DIR / "sam3.pt"), "vid_stride": 10})

    def test_returns_results_per_frame(self, predictor):
        frames = _distinct_frames(3)
        results = predictor.predict_frames(frames, text=["bus"])
        assert len(results) == 3

    def test_temporal_continuity(self, predictor):
        """The tracker accumulates one state per frame, preserving memory across the sequence."""
        frames = _distinct_frames(4)
        predictor.predict_frames(frames, text=["bus"])
        assert len(predictor.inference_state["tracker_inference_states"]) == len(frames)

    def test_clean_second_sequence(self, predictor):
        """A second call starts clean instead of leaking the first sequence's memory."""
        predictor.predict_frames(_distinct_frames(3), text=["bus"])
        predictor.predict_frames(_distinct_frames(2), text=["bus"])
        assert len(predictor.inference_state["tracker_inference_states"]) == 2

    def test_callback_ordering(self, predictor):
        """All standard lifecycle callbacks fire in order through stream_inference."""
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

        predictor.predict_frames(_distinct_frames(2), text=["bus"])

        assert "on_predict_start" in fired
        assert "on_predict_end" in fired
        assert fired.index("on_predict_start") < fired.index("on_predict_batch_start")
        assert fired.index("on_predict_batch_start") < fired.index("on_predict_batch_end")
        assert fired.index("on_predict_batch_end") < fired.index("on_predict_end")

    def test_existing_video_input(self, predictor, solution_assets):
        """The normal on-disk video path still drives the same lifecycle (no regression)."""
        video = str(solution_assets("demo_video"))
        results = predictor(video, text=["object"])
        assert len(results) > 0
        assert len(predictor.inference_state["tracker_inference_states"]) == len(results)
