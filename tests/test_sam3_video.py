# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor, _NumpyVideoLoader


def _tiny_frames(n=3):
    """Create n tiny synthetic BGR frames."""
    return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(n)]


def test_numpy_video_loader_sequence():
    """Shim exposes a video-like interface with a 1-based frame counter."""
    frames = _tiny_frames(3)
    loader = _NumpyVideoLoader(frames)
    assert loader.mode == "video"
    assert loader.frames == 3
    assert len(loader) == 3
    seen = []
    for paths, im0s, _ in loader:
        seen.append(loader.frame)
        assert len(im0s) == 1  # one frame at a time (SAM pre_transform asserts this)
        assert im0s[0] is frames[len(seen) - 1]
    assert seen == [1, 2, 3]


def test_predict_frames_validates_input():
    """Non-empty lists of ndarrays only; invalid input raises ValueError."""
    predictor = SAM3VideoSemanticPredictor.__new__(SAM3VideoSemanticPredictor)
    predictor.inference_state = {}
    with pytest.raises(ValueError):
        predictor.predict_frames([])
    with pytest.raises(ValueError):
        predictor.predict_frames("not-a-list")
    with pytest.raises(ValueError):
        predictor.predict_frames([np.zeros((8, 8, 3), dtype=np.uint8), "not-an-array"])


def _stubbed_predictor():
    """Predictor with heavy model/tracker parts stubbed; real init_state + shim retained."""
    predictor = SAM3VideoSemanticPredictor.__new__(SAM3VideoSemanticPredictor)
    predictor.inference_state = {"stale": "memory"}
    predictor.model = MagicMock()
    predictor.args = SimpleNamespace(imgsz=640)
    predictor.imgsz = [640, 640]
    tracker_model = MagicMock()
    tracker_model.memory_encoder.mask_downsampler.interpol_size = 144
    tracker = SimpleNamespace(imgsz=None, model=tracker_model, _bb_feat_sizes=None)
    predictor.tracker = tracker
    predictor.interpol_size = None
    predictor.stride = 14
    predictor._lock = threading.Lock()
    predictor.run_callbacks = lambda *a, **k: None
    predictor.done_warmup = True
    return predictor


def test_predict_frames_resets_stale_state_and_preserves_sequence():
    """Repeated calls start clean while tracker memory grows within one call."""
    predictor = _stubbed_predictor()
    seen_frames, seen_text = [], {}

    def fake_inference(im, text=None, **kwargs):
        seen_frames.append(predictor.dataset.frame)
        seen_text["text"] = text
        predictor.inference_state["tracker_inference_states"].append(object())  # simulate memory growth
        return {"dummy": True}

    predictor.preprocess = lambda im0s: im0s
    predictor.inference = fake_inference
    predictor.postprocess = lambda preds, im, im0s: [f"result-{predictor.dataset.frame}"]

    frames = _tiny_frames(3)
    results = predictor.predict_frames(frames, text=["cat"])
    assert results == ["result-1", "result-2", "result-3"]
    assert seen_frames == [1, 2, 3]  # 1-based video-like sequence
    assert seen_text["text"] == ["cat"]  # text forwarded to inference machinery
    assert "stale" not in predictor.inference_state
    assert len(predictor.inference_state["tracker_inference_states"]) == 3

    results = predictor.predict_frames(frames, text=["cat"])  # stale state must not leak across calls
    assert len(predictor.inference_state["tracker_inference_states"]) == 3  # rebuilt, not 6
    assert len(results) == 3


def test_predict_frames_full_model():
    """End-to-end predict_frames with sam3.pt weights; skipped when weights/network are unavailable."""
    pytest.importorskip("cv2")
    from ultralytics.utils import WEIGHTS_DIR

    weights = WEIGHTS_DIR / "sam3.pt"
    if not weights.exists():
        pytest.skip("sam3.pt weights not available offline")
    predictor = SAM3VideoSemanticPredictor(overrides={"model": str(weights)})
    frames = [np.full((256, 256, 3), fill_value=i * 40, dtype=np.uint8) for i in range(2)]
    results = predictor.predict_frames(frames, text=["object"])
    assert len(results) == 2
