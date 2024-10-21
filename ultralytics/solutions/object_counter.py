# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
    specified region in a video stream. It supports both polygonal and linear regions for counting.

    Attributes:
        in_count (int): Counter for objects moving inward.
        out_count (int): Counter for objects moving outward.
        counted_ids (List[int]): List of IDs of objects that have been counted.
        classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class.
        region_initialized (bool): Flag indicating whether the counting region has been initialized.
        show_in (bool): Flag to control display of inward count.
        show_out (bool): Flag to control display of outward count.

    Methods:
        count_objects: Counts objects within a polygonal or linear region.
        store_classwise_counts: Initializes class-wise counts if not already present.
        display_counts: Displays object counts on the frame.
        count: Processes input data (frames or object tracks) and updates counts.

    Examples:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = counter.count(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
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
        Counts objects within a polygonal or linear region based on their tracks.

        Args:
            track_line (Dict): Last 30 frame track record for the object.
            box (List[float]): Bounding box coordinates [x1, y1, x2, y2] for the specific track in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(track_line, box, track_id, prev_position, cls)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        # dx = (box[0] - prev_position[0]) * (centroid.x - prev_position[0])
        # dy = (box[1] - prev_position[1]) * (centroid.y - prev_position[1])

        curr_pos = prev_position
        prev_pos = self.track_history[track_id][-3] if len(self.track_history[track_id]) > 2 else None
        if prev_pos is None:
            return

        start_point = self.region[0]
        end_point = self.region[1]
        # A((curr_pos[0]-prev_pos[0]), (curr_pos[1]-prev_pos[1]))
        # B((end_point[0]-start_point[0]), (end_point[1]-start_point[1]))
        d = (curr_pos[0] - prev_pos[0]) * (end_point[1] - start_point[1]) - (curr_pos[1] - prev_pos[1]) * (
            end_point[0] - start_point[0]
        )
        dx = 1
        if d < 0:
            dy = 1
        else:
            dy = -1

        if len(self.region) >= 3 and self.r_s.contains(self.Point(track_line[-1])):
            self.counted_ids.append(track_id)
            # For polygon region
            if dx > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

        elif len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
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
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'IN' and 'OUT' counts to zero if the class is not already present.

        Examples:
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
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
        Processes input data (frames or object tracks) and updates object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.

        Returns:
            (numpy.ndarray): The processed image with annotations and count information.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
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
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history
            self.store_classwise_counts(cls)  # store classwise counts in dict

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )

            # store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_line, box, track_id, prev_position, cls)  # Perform object counting

        self.display_counts(im0)  # Display the counts on the frame
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
