# Ultralytics YOLO 🚀, AGPL-3.0 license
"""This module defines the base classes and structures for object tracking in YOLO."""

from collections import OrderedDict

import numpy as np


class TrackState:
    """
    Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """
    Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (list): List of features extracted from the object for tracking.
        curr_feature (any): The current feature of the object being tracked.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.
    """

    _count = 0

    def __init__(self):
        """Initializes a new track with unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

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
        """Abstract method to activate the track with provided arguments."""
        raise NotImplementedError

    def predict(self):
        """Abstract method to predict the next state of the track."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Abstract method to update the track with new observations."""
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
