# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from time import time

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class SpeedEstimator(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.

    Attributes:
        spd (Dict[int, float]): Dictionary storing speed data for tracked objects.
        trkd_ids (List[int]): List of tracked object IDs that have already been speed-estimated.
        trk_pt (Dict[int, float]): Dictionary storing previous timestamps for tracked objects.
        trk_pp (Dict[int, Tuple[float, float]]): Dictionary storing previous positions for tracked objects.
        region (List[Tuple[int, int]]): List of points defining the speed estimation region.
        track_line (List[Tuple[float, float]]): List of points representing the object's track.
        r_s (LineString): LineString object representing the speed estimation region.

    Methods:
        initialize_region: Initializes the speed estimation region.
        process: Processes input frames to estimate object speeds.
        store_tracking_history: Stores the tracking history for an object.
        extract_tracks: Extracts tracks from the current frame.
        display_output: Displays the output with annotations.

    Examples:
        >>> estimator = SpeedEstimator()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("Speed Estimation", results.plot_im)
    """

    def __init__(self, **kwargs):
        """
        Initialize the SpeedEstimator object with speed estimation parameters and data structures.

        Args:
            **kwargs (Any): Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self.initialize_region()  # Initialize speed region

        self.spd = {}  # Dictionary for speed data
        self.trkd_ids = []  # List for already speed-estimated and tracked IDs
        self.trk_pt = {}  # Dictionary for tracks' previous timestamps
        self.trk_pp = {}  # Dictionary for tracks' previous positions

    def process(self, im0):
        """
        Process an input frame to estimate object speeds based on tracking data.

        Args:
            im0 (np.ndarray): Input image for processing with shape (H, W, C) for RGB images.

        Returns:
            (SolutionResults): Contains processed image `plot_im` and `total_tracks` (number of tracked objects).

        Examples:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.extract_tracks(im0)  # Extract tracks
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        # Draw speed estimation region
        annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store track history

            # Initialize tracking data for new objects
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            # Prepare label with speed if available, otherwise use class name
            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # Draw bounding box

            # Determine if object is crossing the speed estimation region
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            # Calculate speed for objects crossing the region for the first time
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    # Calculate speed based on vertical displacement and time
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

            # Update tracking data for next frame
            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return results with processed image and tracking summary
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
