# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class QueueManager(BaseSolution):
    """
    Manages queue counting in real-time video streams based on object tracks.

    This class extends BaseSolution to provide functionality for tracking and counting objects within a specified
    region in video frames.

    Attributes:
        counts (int): The current count of objects in the queue.
        rect_color (Tuple[int, int, int]): RGB color tuple for drawing the queue region rectangle.
        region_length (int): The number of points defining the queue region.
        annotator (Annotator): An instance of the Annotator class for drawing on frames.
        track_line (List[Tuple[int, int]]): List of track line coordinates.
        track_history (Dict[int, List[Tuple[int, int]]]): Dictionary storing tracking history for each object.

    Methods:
        initialize_region: Initializes the queue region.
        process_queue: Processes a single frame for queue management.
        extract_tracks: Extracts object tracks from the current frame.
        store_tracking_history: Stores the tracking history for an object.
        display_output: Displays the processed output.

    Examples:
        >>> queue_manager = QueueManager(source="video.mp4", region=[100, 100, 200, 200, 300, 300])
        >>> for frame in video_stream:
        ...     processed_frame = queue_manager.process_queue(frame)
        ...     cv2.imshow("Queue Management", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initializes the QueueManager with parameters for tracking and counting objects in a video stream."""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # Queue counts Information
        self.rect_color = (255, 255, 255)  # Rectangle color
        self.region_length = len(self.region)  # Store region length for further usage

    def process_queue(self, im0):
        """
        Processes the queue management for a single frame of video.

        Args:
            im0 (numpy.ndarray): Input image for processing, typically a frame from a video stream.

        Returns:
            (numpy.ndarray): Processed image with annotations, bounding boxes, and queue counts.

        This method performs the following steps:
        1. Resets the queue count for the current frame.
        2. Initializes an Annotator object for drawing on the image.
        3. Extracts tracks from the image.
        4. Draws the counting region on the image.
        5. For each detected object:
           - Draws bounding boxes and labels.
           - Stores tracking history.
           - Draws centroids and tracks.
           - Checks if the object is inside the counting region and updates the count.
        6. Displays the queue count on the image.
        7. Displays the processed output.

        Examples:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> processed_frame = queue_manager.process_queue(frame)
        """
        self.counts = 0  # Reset counts every frame
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2
        )  # Draw region

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # Cache frequently accessed attributes
            track_history = self.track_history.get(track_id, [])

            # store previous position of track and check if the object is inside the counting region
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # Display queue counts
        self.annotator.queue_counts_display(
            f"Queue Counts : {str(self.counts)}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
