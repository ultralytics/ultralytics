# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None
        self.view_img = False
        self.shape = "circle"

        self.names = None  # Classes names

        # Image information
        self.imw = None
        self.imh = None
        self.im0 = None
        self.view_in_counts = True
        self.view_out_counts = True

        # Heatmap colormap and heatmap np array
        self.colormap = None
        self.heatmap = None
        self.heatmap_alpha = 0.5

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None
        self.track_history = defaultdict(list)

        # Region & Line Information
        self.count_reg_pts = None
        self.counting_region = None
        self.line_dist_thresh = 15
        self.region_thickness = 5
        self.region_color = (255, 0, 255)

        # Object Counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.line_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        # Decay factor
        self.decay_factor = 0.99

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        imw,
        imh,
        classes_names=None,
        colormap=cv2.COLORMAP_JET,
        heatmap_alpha=0.5,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        count_reg_pts=None,
        count_txt_thickness=2,
        count_txt_color=(255, 255, 255),
        fontsize=0.8,
        line_color=(255, 255, 255),
        count_reg_color=(255, 0, 255),
        region_thickness=5,
        line_dist_thresh=15,
        decay_factor=0.99,
        shape="circle",
        cls_txtdisplay_gap=50,
    ):
        """
        Configures the heatmap colormap, width, height and display parameters.

        Args:
            colormap (cv2.COLORMAP): The colormap to be set.
            imw (int): The width of the frame.
            imh (int): The height of the frame.
            classes_names (dict): Classes names
            heatmap_alpha (float): alpha value for heatmap display
            view_img (bool): Flag indicating frame display
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            count_reg_pts (list): Object counting region points
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            fontsize (float): Text display font size
            line_color (RGB color): count highlighter line color
            count_reg_color (RGB color): Color of object counting region
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            decay_factor (float): value for removing heatmap area after object passed
            shape (str): Heatmap shape, rect or circle shape supported
            cls_txtdisplay_gap (int): Display gap between each class count
        """
        self.names = classes_names
        self.imw = imw
        self.imh = imh
        self.heatmap_alpha = heatmap_alpha
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.colormap = colormap

        # Region and line selection
        if count_reg_pts is not None:
            if len(count_reg_pts) == 2:
                print("Line Counter Initiated.")
                self.count_reg_pts = count_reg_pts
                self.counting_region = LineString(self.count_reg_pts)
            elif len(count_reg_pts) >= 3:
                print("Polygon Counter Initiated.")
                self.count_reg_pts = count_reg_pts
                self.counting_region = Polygon(self.count_reg_pts)
            else:
                print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
                print("Using Line Counter Now")
                self.counting_region = LineString(self.count_reg_pts)

        # Heatmap new frame
        self.heatmap = np.zeros((int(self.imh), int(self.imw)), dtype=np.float32)

        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.fontsize = fontsize
        self.line_color = line_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.decay_factor = decay_factor
        self.line_dist_thresh = line_dist_thresh
        self.shape = shape
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

        # shape of heatmap, if not selected
        if self.shape not in {"circle", "rect"}:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print("Using Circular shape now")
            self.shape = "circle"

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            self.heatmap = np.zeros((int(self.imh), int(self.imw)), dtype=np.float32)
            if self.view_img and self.env_check:
                self.display_frames()
            return im0
        self.heatmap *= self.decay_factor  # decay factor
        self.extract_results(tracks)
        self.annotator = Annotator(self.im0, self.count_txt_thickness, None)

        if self.count_reg_pts is not None:
            # Draw counting region
            if self.view_in_counts or self.view_out_counts:
                self.annotator.draw_region(
                    reg_pts=self.count_reg_pts, color=self.region_color, thickness=self.region_thickness
                )

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    if len(self.names[cls]) > 5:
                        self.names[cls] = self.names[cls][:5]
                    self.class_wise_count[self.names[cls]] = {"in": 0, "out": 0}

                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0 : self.heatmap.shape[0], 0 : self.heatmap.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += (
                        2 * mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    )

                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.count_reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["in"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["out"] += 1

                # Count objects using line
                elif len(self.count_reg_pts) == 2:
                    is_inside = (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)

                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["in"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["out"] += 1

        else:
            for box, cls in zip(self.boxes, self.clss):
                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0 : self.heatmap.shape[0], 0 : self.heatmap.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += (
                        2 * mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    )

                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)

        label = "Ultralytics Analytics \t"

        for key, value in self.class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    label = None
                elif not self.view_in_counts:
                    label += f"{str.capitalize(key)}: IN {value['in']} \t"
                elif not self.view_out_counts:
                    label += f"{str.capitalize(key)}: OUT {value['out']} \t"
                else:
                    label += f"{str.capitalize(key)}: IN {value['in']} OUT {value['out']} \t"

        label = label.rstrip()
        label = label.split("\t")

        if self.count_reg_pts is not None and label is not None:
            self.annotator.display_counts(
                counts=label,
                tf=self.count_txt_thickness,
                fontScale=self.fontsize,
                txt_color=self.count_txt_color,
                line_color=self.line_color,
                classwise_txtgap=self.cls_txtdisplay_gap,
            )

        self.im0 = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)

        if self.env_check and self.view_img:
            self.display_frames()

        return self.im0

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Heatmap", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    Heatmap()
