# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements('shapely>=2.0.0')

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

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
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(self,
                 classes_names,
                 reg_pts,
                 count_reg_color=(255, 0, 255),
                 line_thickness=2,
                 track_thickness=2,
                 view_img=False,
                 draw_tracks=False,
                 count_txt_thickness=2,
                 count_txt_color=(0, 0, 0),
                 count_color=(255, 255, 255),
                 track_color=(0, 255, 0),
                 region_thickness=5,
                 line_dist_thresh=15):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print('Line Counter Initiated.')
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print('Region Counter Initiated.')
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print('Invalid Region points provided, region_points can be 2 or 4')
            print('Using Line Counter Now')
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh

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

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=str(track_id) + ':' + self.names[cls],
                                     color=colors(int(cls), True))  # Draw bounding box

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(track_line,
                                                        color=self.track_color,
                                                        track_thickness=self.track_thickness)

            # Count objects
            if len(self.reg_pts) == 4:
                if self.counting_region.contains(Point(track_line[-1])):
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if box[0] < self.counting_region.centroid.x:
                            self.out_counts += 1
                        else:
                            self.in_counts += 1

            elif len(self.reg_pts) == 2:
                distance = Point(track_line[-1]).distance(self.counting_region)
                if distance < self.line_dist_thresh:
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if box[0] < self.counting_region.centroid.x:
                            self.out_counts += 1
                        else:
                            self.in_counts += 1

        incount_label = 'In Count : ' + f'{self.in_counts}'
        outcount_label = 'OutCount : ' + f'{self.out_counts}'
        self.annotator.count_labels(in_count=incount_label,
                                    out_count=outcount_label,
                                    count_txt_size=self.count_txt_thickness,
                                    txt_color=self.count_txt_color,
                                    color=self.count_color)

        if self.env_check and self.view_img:
            cv2.namedWindow('Ultralytics YOLOv8 Object Counter')
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
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
        return self.im0


if __name__ == '__main__':
    ObjectCounter()
