# Ultralytics YOLO 🚀, AGPL-3.0 license

from shapely.geometry import LineString, Point

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self, **kwargs):
        """Initialization function for Count class, a child class of BaseSolution class, can be used for counting the
        objects.
        """
        super().__init__(**kwargs)

        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_counts = {}  # Dictionary for counts, categorized by object class
        self.region_initialized = False  # Bool variable for region initialization

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

    def count_objects(self, track_line, box, track_id, prev_position, cls):
        """
        Helper function to count objects within a polygonal region.

        Args:
            track_line (dict): last 30 frame track record
            box (list): Bounding box data for specific track in current frame
            track_id (int): track ID of the object
            prev_position (tuple): last frame position coordinates of the track
            cls (int): Class index for classwise count updates
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        centroid = self.r_s.centroid
        dx = (box[0] - prev_position[0]) * (centroid.x - prev_position[0])
        dy = (box[1] - prev_position[1]) * (centroid.y - prev_position[1])

        if len(self.region) >= 3 and self.r_s.contains(Point(track_line[-1])):
            self.counted_ids.append(track_id)
            # For polygon region
            if dx > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

        elif len(self.region) < 3 and LineString([prev_position, box[:2]]).intersects(self.l_s):
            self.counted_ids.append(track_id)
            # For linear region
            if dx > 0 and dy > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts if not already present.

        Args:
            cls (int): Class index for classwise count updates
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        Helper function to display object counts on the frame.

        Args:
            im0 (ndarray): The input image or frame
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """
        Processes input data (frames or object tracks) and updates counts.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # Store track history
            self.store_classwise_counts(cls)  # store classwise counts in dict

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # store previous position of track for object counting
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            self.count_objects(self.track_line, box, track_id, prev_position, cls)  # Perform object counting

        self.display_counts(im0)  # Display the counts on the frame
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
