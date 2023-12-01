# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements('shapely>=2.0.0')

from shapely.geometry import Polygon
from shapely.geometry.point import Point


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region Information
        self.reg_pts = None
        self.counting_region = None
        self.region_color = (255, 255, 255)

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False

    def set_args(self,
                 classes_names,
                 reg_pts,
                 region_color=None,
                 line_thickness=2,
                 track_thickness=2,
                 view_img=False,
                 draw_tracks=False):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            region_color (tuple): color for region line
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.reg_pts = reg_pts
        self.counting_region = Polygon(self.reg_pts)
        self.names = classes_names
        self.region_color = region_color if region_color else self.region_color

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        # global is_drawing, selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if isinstance(point, (tuple, list)) and len(point) >= 2:
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
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
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=(0, 255, 0))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(int(cls), True))  # Draw bounding box

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            track_line.pop(0) if len(track_line) > 30 else None

            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(track_line,
                                                        color=(0, 255, 0),
                                                        track_thickness=self.track_thickness)

            # Count objects
            if self.counting_region.contains(Point(track_line[-1])):
                if track_id not in self.counting_list:
                    self.counting_list.append(track_id)
                    if box[0] < self.counting_region.centroid.x:
                        self.out_counts += 1
                    else:
                        self.in_counts += 1

        if self.view_img:
            incount_label = 'InCount : ' + f'{self.in_counts}'
            outcount_label = 'OutCount : ' + f'{self.out_counts}'
            self.annotator.count_labels(in_count=incount_label, out_count=outcount_label)
            cv2.namedWindow('Ultralytics YOLOv8 Object Counter')
            cv2.setMouseCallback('Ultralytics YOLOv8 Object Counter', self.mouse_event_for_region,
                                 {'region_points': self.reg_pts})
            cv2.imshow('Ultralytics YOLOv8 Object Counter', self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        if tracks[0].boxes.id is None:
            return
        self.extract_and_process_tracks(tracks)


if __name__ == '__main__':
    ObjectCounter()
