# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import OrderedDict

import numpy as np


class TrackState:
    """Enumeration of possible object tracking states."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking, handling basic track attributes and operations."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # Multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Return the last frame ID of the track."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Increment and return the global track ID counter."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Activate the track with the provided arguments."""
        raise NotImplementedError

    def predict(self):
        """Predict the next state of the track."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Update the track with new observations."""
        raise NotImplementedError

    def mark_lost(self):
        """Mark the track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self):
        """Mark the track as removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the global track ID counter."""
        BaseTrack._count = 0
