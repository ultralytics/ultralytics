# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque
from math import sqrt
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class SpeedEstimator(BaseSolution):
    """A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using tracking
    data in video streams. Speed is calculated based on pixel displacement over time and converted to real-world units
    using a configurable meters-per-pixel scale factor.

    Attributes:
        fps (float): Video frame rate for time calculations.
        frame_count (int): Global frame counter for tracking temporal information.
        trk_frame_ids (dict): Maps track IDs to their first frame index.
        spd (dict): Final speed per object in km/h once locked.
        trk_hist (dict): Maps track IDs to deque of position history.
        locked_ids (set): Track IDs whose speed has been finalized.
        max_hist (int): Required frame history before computing speed.
        meter_per_pixel (float): Real-world meters represented by one pixel for scene scale conversion.
        max_speed (int): Maximum allowed object speed; values above this will be capped.

    Methods:
        process: Process input frames to estimate object speeds based on tracking data.
        store_tracking_history: Store the tracking history for an object.
        extract_tracks: Extract tracks from the current frame.
        display_output: Display the output with annotations.

    Examples:
        Initialize speed estimator and process a frame
        >>> estimator = SpeedEstimator(meter_per_pixel=0.04, max_speed=120)
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("Speed Estimation", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SpeedEstimator object with speed estimation parameters and data structures.

        Args:
            **kwargs (Any): Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self.fps = self.CFG["fps"]  # Video frame rate for time calculations
        self.frame_count = 0  # Global frame counter
        self.trk_frame_ids = {}  # Track ID → first frame index
        self.spd = {}  # Final speed per object (km/h), once locked
        self.trk_hist = {}  # Track ID → deque of (time, position)
        self.locked_ids = set()  # Track IDs whose speed has been finalized
        self.max_hist = self.CFG["max_hist"]  # Required frame history before computing speed
        self.meter_per_pixel = self.CFG["meter_per_pixel"]  # Scene scale, depends on camera details
        self.max_speed = self.CFG["max_speed"]  # Maximum speed adjustment

    def process(self, im0) -> SolutionResults:
        """Process an input frame to estimate object speeds based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing with shape (H, W, C) for RGB images.

        Returns:
            (SolutionResults): Contains processed image `plot_im` and `total_tracks` (number of tracked objects).

        Examples:
            Process a frame for speed estimation
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, _, _ in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:  # Initialize history if new track found
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
                self.trk_frame_ids[track_id] = self.frame_count

            if track_id not in self.locked_ids:  # Update history until speed is locked
                trk_hist = self.trk_hist[track_id]
                trk_hist.append(self.track_line[-1])

                # Compute and lock speed once enough history is collected
                if len(trk_hist) == self.max_hist:
                    p0, p1 = trk_hist[0], trk_hist[-1]  # First and last points of track
                    dt = (self.frame_count - self.trk_frame_ids[track_id]) / self.fps  # Time in seconds
                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]  # Pixel displacement
                        pixel_distance = sqrt(dx * dx + dy * dy)  # Calculate pixel distance
                        meters = pixel_distance * self.meter_per_pixel  # Convert to meters
                        self.spd[track_id] = int(
                            min((meters / dt) * 3.6, self.max_speed)
                        )  # Convert to km/h and store final speed
                        self.locked_ids.add(track_id)  # Prevent further updates
                        self.trk_hist.pop(track_id, None)  # Free memory
                        self.trk_frame_ids.pop(track_id, None)  # Remove frame start reference

            if track_id in self.spd:
                speed_label = f"{self.spd[track_id]} km/h"
                annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # Draw bounding box

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return results with processed image and tracking summary
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
