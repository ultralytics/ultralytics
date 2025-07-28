# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class QueueManager(BaseSolution):
    """
    Manages queue counting in real-time video streams based on object tracks.

    This class extends BaseSolution to provide functionality for tracking and counting objects within a specified
    region in video frames.

    Attributes:
        counts (int): The current count of objects in the queue.
        rect_color (Tuple[int, int, int]): RGB color tuple for drawing the queue region rectangle.
        region_length (int): The number of points defining the queue region.
        track_line (List[Tuple[int, int]]): List of track line coordinates.
        track_history (Dict[int, List[Tuple[int, int]]]): Dictionary storing tracking history for each object.

    Methods:
        initialize_region: Initialize the queue region.
        process: Process a single frame for queue management.
        extract_tracks: Extract object tracks from the current frame.
        store_tracking_history: Store the tracking history for an object.
        display_output: Display the processed output.

    Examples:
        >>> cap = cv2.VideoCapture("path/to/video.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        >>>     success, im0 = cap.read()
        >>>     if not success:
        >>>         break
        >>>     results = queue_manager.process(im0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the QueueManager with parameters for tracking and counting objects in a video stream."""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # Queue counts information
        self.rect_color = (255, 255, 255)  # Rectangle color for visualization
        self.region_length = len(self.region)  # Store region length for further usage

    def process(self, im0) -> SolutionResults:
        """
        Process queue management for a single frame of video.

        Args:
            im0 (np.ndarray): Input image for processing, typically a frame from a video stream.

        Returns:
            (SolutionResults): Contains processed image `im0`, 'queue_count' (int, number of objects in the queue) and
                'total_tracks' (int, total number of tracked objects).

        Examples:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = queue_manager.process(frame)
        """
        self.counts = 0  # Reset counts every frame
        self.extract_tracks(im0)  # Extract tracks from the current frame
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)  # Draw region

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # Draw bounding box and counting region
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Cache frequently accessed attributes
            track_history = self.track_history.get(track_id, [])

            # Store previous position of track and check if the object is inside the counting region
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # Display queue counts
        annotator.queue_counts_display(
            f"Queue Counts : {str(self.counts)}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return a SolutionResults object with processed data
        return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
