# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        line_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
        """
        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
        self.counting_region = None

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = names  # Classes names
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}

        # Tracks info
        self.track_history = defaultdict(list)
        self.draw_tracks = draw_tracks

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize counting region
        if len(self.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        # Define the counting line segment
        self.counting_line_segment = LineString(
            [
                (self.reg_pts[0][0], self.reg_pts[0][1]),
                (self.reg_pts[1][0], self.reg_pts[1][1]),
            ]
        )

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        # Annotator Init and region drawing
        annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        annotator.draw_region(reg_pts=self.reg_pts, color=(104, 0, 123), thickness=self.tf * 2)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                annotator.box_label(box, label=self.names[cls], color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    annotator.draw_centroid_and_tracks(
                        track_line,
                        color=colors(int(track_id), True),
                        track_thickness=self.tf,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1

                # Count objects using line
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        # Check if the object's movement segment intersects the counting line
                        if LineString([(prev_position[0], prev_position[1]), (box[0], box[1])]).intersects(
                            self.counting_line_segment
                        ):
                            self.count_ids.append(track_id)

                            # Determine the direction of movement (IN or OUT)
                            dx = (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0])
                            dy = (box[1] - prev_position[1]) * (self.counting_region.centroid.y - prev_position[1])
                            if dx > 0 and dy > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            annotator.display_analytics(self.im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(classes_names)
