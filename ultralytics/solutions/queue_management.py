# Ultralytics YOLO 🚀, AGPL-3.0 license

from shapely.geometry import Point

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator, colors


class QueueManager(BaseSolution):
    """A class to manage the queue in a real-time video stream based on object tracks."""

    def __init__(self, **kwargs):
        """Initializes the QueueManager with specified parameters for tracking and counting objects."""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # Queue counts Information
        self.rect_color = (255, 255, 255)  # Rectangle color
        self.region_length = len(self.region)  # Store region length for further usage

    def process_queue(self, im0):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
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
            prev_position = track_history[-2] if len(track_history) > 1 else None
            if self.region_length >= 3 and prev_position and self.r_s.contains(Point(self.track_line[-1])):
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
