# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Regression test for index collision in the second-association path across
all integrated trackers (BYTETrack, BoT-SORT, TrackTrack, FastTrack, OC-SORT, DeepOCSort).

This test reproduces the exact scenario described in the PR:
- 3 independent non-overlapping targets with stable bounding boxes.
- Only confidence scores change between frames (common in real detection due to environmental factors).
- Frame 1: all high-score → first association.
- Frame 2: one target drops to low-score → second association.

The bug occurs because the second-association path used a local subset index
(np.arange) instead of the original detection-set index, causing index collision
(e.g. track2_idx=0 colliding with track0_idx).

The test is parameterized over all supported tracker configs so that any
regression in any tracker's second-association logic is caught automatically.
"""

import numpy as np
import pytest

from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# All tracker configurations to test
TRACKER_CONFIGS = [
    "bytetrack.yaml",
    "botsort.yaml",
    "tracktrack.yaml",
    "fasttrack.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
]


class MockResults:
    """Minimal mock of ultralytics Results for tracker unit testing."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.array(xywh, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.float32)
        self.data = np.concatenate(
            [self.xywh, self.conf.reshape(-1, 1), self.cls.reshape(-1, 1)], axis=1
        )

    def __getitem__(self, idx):
        """Obtain specified detection data."""
        return MockResults(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        """Return the number of detections."""
        return len(self.data)


@pytest.fixture
def tracker(request):
    """Return a fresh tracker instance parameterized by config file.

    Iterates over all integrated tracker configurations defined in TRACKER_CONFIGS.
    """
    config_file = request.param

    tracker_path = check_yaml(config_file)
    cfg = IterableSimpleNamespace(**YAML.load(tracker_path))

    cfg.track_high_thresh = 0.6  # as specified in the deduction
    cfg.track_low_thresh = 0.1  # as specified in the deduction
    cfg.match_thresh = 0.8  # high enough that tiny bbox shifts still match
    cfg.new_track_thresh = 0.6  # activate tracks on first frame
    if hasattr(cfg, "use_byte"):
        cfg.use_byte = True

    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    return tracker


@pytest.mark.parametrize("tracker", TRACKER_CONFIGS, indirect=True)
def test_second_association_preserves_detection_set_index(tracker):
    """
    Cover the [0, 1, 0] -> [0, 1, 2] case from the PR description.

    This test runs against every integrated tracker (BYTETrack, BoT-SORT,
    TrackTrack, FastTrack, OC-SORT, DeepOCSort) to ensure that none of them
    regress on the second-association index-collision bug.

    Deduction steps
    ---------------
    0) Frame 1 – trajectory initialization
       scores = {"obj0": 0.90, "obj1": 0.88, "obj2": 0.72}
       All confidences > track_high_thresh (0.6)
       results        = [obj0, obj1, obj2]
       tracks (after update) idx = [0, 1, 2]

    1) Frame 2 – second-round matching
       scores = {"obj0": 0.91, "obj1": 0.86, "obj2": 0.56}
       obj0, obj1  > 0.6  → first association  → results       = [obj0, obj1]
       obj2  in (0.1, 0.6) → second association → results_second = [obj2]
       obj2 matches the remaining unmatched track (track2).

    Bug (old code)
    --------------
    init_track used ``np.arange(len(bboxes))`` for the second-association subset,
    so track2 received the *local* subset index 0 instead of the original detection
    index 2.  Final idx vector became [0, 1, 0] — a collision.

    Fix (current code)
    ------------------
    The original detection indices are propagated via explicit ``np.where(...)``
    arrays, so track2 keeps its original index 2.  Final idx vector must be
    [0, 1, 2] (order may vary, but the set must be {0, 1, 2} with no duplicates).
    """
    # ---------- Frame 1 : initialise three tracks ----------
    # Three independent non-overlapping targets, all high-score.
    frame1 = MockResults(
        xywh=[
            [100.0, 100.0, 50.0, 50.0],  # obj0 original detection index 0
            [200.0, 200.0, 50.0, 50.0],  # obj1 original detection index 1
            [300.0, 300.0, 50.0, 50.0],  # obj2 original detection index 2
        ],
        conf=[0.90, 0.88, 0.72],  # all > track_high_thresh (0.5)
        cls=[0, 0, 0],
    )
    tracks1 = tracker.update(frame1)
    assert len(tracks1) == 3, "All three high-score detections should be activated in frame 1"
    assert set(tracks1[:, -1].astype(int)) == {0, 1, 2}, "Frame-1 indices should be the original detection indices"

    # ---------- Frame 2 : one target drops to low-score ----------
    # Bounding boxes stay almost the same (tiny shift so IoU is still high).
    frame2 = MockResults(
        xywh=[
            [101.0, 101.0, 50.0, 50.0],  # obj0 high score (0.91)
            [201.0, 201.0, 50.0, 50.0],  # obj1 high score (0.82)
            [301.0, 301.0, 50.0, 50.0],  # obj2 low score (0.56), enters second association
        ],
        conf=[0.91, 0.86, 0.56],  # obj2 in (track_low_thresh, track_high_thresh) = (0.1, 0.5)
        cls=[0, 0, 0],
    )
    tracks2 = tracker.update(frame2)
    assert len(tracks2) == 3, "All three tracks should survive in frame 2"

    # The last column of the returned array is the original detection index (idx)
    idxs = tracks2[:, -1].astype(int)

    # Core assertion: the [0, 1, 0] bug must not happen.
    assert len(set(idxs)) == len(idxs), (
        f"Index collision detected in second-association path: {idxs}. "
        f"Expected unique indices, got duplicates."
    )

    # All original detection indices {0, 1, 2} must be preserved after second association.
    assert set(idxs) == {0, 1, 2}, (
        f"Original detection indices lost after second association: {idxs}. "
        f"Expected {{0, 1, 2}}, got {set(idxs)}."
    )
