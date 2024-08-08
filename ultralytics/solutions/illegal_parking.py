import time
from collections import defaultdict, deque
from typing import Union

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class IllegalParking(object):
    """A class for judging illegal parking in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names: dict,
        polygon_coords: list = None,
        view_img: bool = False,
        line_thickness: int = 2,
        polygon_thickness: int = 5,
        parking_threshold: int = 50,
        object_iou_threshold: float = 0.80,
    ):
        """
        Initializes the IllegalParking with the given parameters.

        Args:
            names: Dictionary of class names.
            polygon_coords: List of polygon coordinates for judging illegal parking.Defaults to None.
            view_img: Whether to display the image with annotations. Defaults to False.
            line_thickness: Thickness of the lines for drawing boxes and tracks. Defaults to 2.
            polygon_thickness: Thickness of the polygon lines. Defaults to 5.
            parking_threshold: Threshold for the number of frames to park. Defaults to 50.
            object_iou_threshold: The intersection over union(iou) threshold of the object at different times.
                Defaults to 0.80.
        """
        # Visual & image information
        self.im0 = None
        self.annotator = None
        self.view_img = view_img

        # polygons information
        self.polygon_coords = polygon_coords
        self.polygon_thickness = polygon_thickness

        # Tracking information
        self.clss = None
        self.names = names
        self.boxes = None
        self.trk_ids = None
        self.line_thickness = line_thickness

        # Illegal parking information
        self.parking_threshold = parking_threshold
        self.object_iou_threshold = object_iou_threshold
        self.parking_ids_set = set()
        self.trk_start_time = defaultdict(int)
        self.trk_window = defaultdict(list)

        # Check if the environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize polygons
        self.polygons = []
        for coords in self.polygon_coords:
            if len(coords) >= 3:
                polygon = Polygon(coords)
                self.polygons.append(polygon)
            else:
                print(f"Invalid Polygon coordinates {coords} provided, coords must >= 3 for polygons.")
        if not self.polygons:
            print("Polygons is empty, Initiated as the whole of images. ")

    @staticmethod
    def iou(boxes: Union[list, deque], box: Union[list, deque]):
        """
        Calculate the Intersection over Union (IoU) between a set of boxes and a single box using a vectorized approach.

        Args:
            boxes: Shape (N, 4), where N is the number of bounding boxes,
                and each box is represented as [x1, y1, x2, y2].
            box: Shape (4,), representing a single bounding box as [x1, y1, x2, y2].

        Returns:
            Return an array of IoU values for each box in boxes with respect to the given box.
        """
        boxes = np.asarray(boxes)
        box = np.asarray(box)

        x1_inter = np.maximum(boxes[:, 0], box[0])
        y1_inter = np.maximum(boxes[:, 1], box[1])
        x2_inter = np.minimum(boxes[:, 2], box[2])
        y2_inter = np.minimum(boxes[:, 3], box[3])

        inter_width = np.maximum(0, x2_inter - x1_inter)
        inter_height = np.maximum(0, y2_inter - y1_inter)

        inter_area = inter_width * inter_height

        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        single_box_area = (box[2] - box[0]) * (box[3] - box[1])

        union_area = box_area + single_box_area - inter_area
        iou = inter_area / union_area
        return iou

    def extract_tracks(self, tracks: list):
        """
        Extracts results from the provided tracking data.

        Args:
            tracks: List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().tolist()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def plot_box(self, track_id: int, box: list, cls: int):
        """
        Plot bounding box.

        Args:
            track_id: Object track id.
            box: Object bounding box data.
            cls(str): Object class code.
        """
        is_parking = track_id in self.parking_ids_set
        parking_label = (
            f"{self.names[int(cls)]}: {int(time.time() - self.trk_start_time[track_id])} s."
            if (is_parking)
            else self.names[int(cls)]
        )
        bbox_color = colors(int(track_id)) if is_parking else (255, 0, 255)
        self.annotator.box_label(box, parking_label, bbox_color)

    def inside(self, point: list) -> bool:
        """
        Determine if a point is inside any of a set of polygons.

        Args:
            point: a coordinate point.
        """
        point = Point(point)
        for polygon in self.polygons:
            if polygon.contains(point):
                return True
        return False

    def found_illegal_parking(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Found the illegal parking objects based on tracking data.

        Args:
            im0 (ndarray): Image.
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple, optional): Color to use when drawing regions. Defaults to (255, 0, 0).

        Returns:
            (ndarray): The image with annotated boxes and tracks.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return im0

        self.extract_tracks(tracks)
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)
        if self.polygons:
            for polygon in self.polygons:
                reg_pts = list(polygon.exterior.coords)
                self.annotator.draw_region(reg_pts=reg_pts, color=region_color, thickness=self.polygon_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            is_inside = True
            if self.polygons:
                is_inside = self.inside(center)

            if is_inside:
                self.trk_window[trk_id].append(box)
                is_gt_parking_threshold = len(self.trk_window[trk_id]) > self.parking_threshold
                if is_gt_parking_threshold:
                    self.trk_window[trk_id].pop(0)
                is_parking = (
                    is_gt_parking_threshold
                    and (IllegalParking.iou(self.trk_window[trk_id], box) >= self.object_iou_threshold).all()
                )
                if is_parking:
                    self.parking_ids_set.add(trk_id)
                    self.trk_start_time.setdefault(trk_id, time.time())
            else:
                if trk_id in self.parking_ids_set:
                    self.parking_ids_set.remove(trk_id)
                    self.trk_start_time.pop(trk_id)

            self.plot_box(trk_id, box, cls)

        if self.view_img and self.env_check:
            self.display_frames()
        return im0

    def display_frames(self):
        """Displays the current frame."""
        cv2.imshow("Ultralytics Illegal Parking", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    my_names = {0: "truck", 1: "car"}
    illegal_parking = IllegalParking(my_names)
