# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from time import time
from collections import deque
from math import sqrt
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

        self.spd = {}  # Final speed per object (km/h), once locked
        self.trk_hist = {}  # Track ID → deque of (time, position)
        self.locked_ids = set()  # Track IDs whose speed has been finalized
        self.max_hist = kwargs.get(5, "max_hist")  # Required frame history before computing speed
        self.meters_per_pixel = kwargs.get("meter_per_pixel", 0.05)  # Scene scale, depends on camera details
        self.max_speed = kwargs.get("max_speed")  # max_speed adjustment

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
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        current_time = time()

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)
            pos = self.track_line[-1]

            # Initialize history if new track
            if track_id not in self.trk_hist:
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)

            # Keep updating history until speed is locked
            if track_id not in self.locked_ids:
                self.trk_hist[track_id].append((current_time, pos))

                # Once enough history is collected, compute and lock speed
                if len(self.trk_hist[track_id]) == self.max_hist:
                    t0, p0, t1, p1 = self.trk_hist[track_id][0], self.trk_hist[track_id][-1]
                    dt = t1 - t0
                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        pixel_distance = sqrt(dx * dx + dy * dy)
                        meters = pixel_distance * self.meters_per_pixel
                        speed_kmph = ((meters / dt) * 3.6)
                        if self.max_speed is not None:
                            speed_kmph = min(speed_kmph, 120)  # clamp max
                        self.spd[track_id] = int(speed_kmph)
                        self.locked_ids.add(track_id)  # lock this value

            # Draw label: show locked speed if available
            speed_label = (
                f"{self.spd[track_id]} km/h"
                if track_id in self.spd and self.show_labels
                else self.adjust_box_label(cls, conf, track_id)
            )
            annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # Draw bounding box

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return results with processed image and tracking summary
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
