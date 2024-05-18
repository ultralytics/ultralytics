# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements

check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Point, Polygon


class Solutions:

    def __init__(self,
                 classes_names=None,
                 reg_pts=None,
                 count_reg_color=(255, 0, 255),
                 count_txt_color=(255, 255, 255),
                 count_bg_color=(104, 31, 17),
                 line_thickness=2,
                 track_thickness=2,
                 view_img=False,
                 view_in_counts=True,
                 view_out_counts=True,
                 draw_tracks=False,
                 track_color=(255, 144, 31),
                 region_thickness=5,
                 line_dist_thresh=15,
                 count_type="classwise",
                 imw=None,
                 imh=None,
                 colormap=cv2.COLORMAP_JET,
                 heatmap_alpha=0.5,
                 count_reg_pts=None,
                 decay_factor=0.99,
                 shape="circle",
                 kpts_to_check=None,
                 pose_type="pushup",
                 pose_up_angle=145.0,
                 pose_down_angle=90.0,
                 spdl_dist_thresh=10,
                 ):
        """
        The Ultralytics solutions modules are designed to address various complex and prevalent needs,
        such as object counting, workout monitoring, heatmaps, speed estimation, object cropping,
        object blurring, instance segmentation, distance calculation, and more features are on the way.
        This function initializes the solution arguments according to the user's preferences.
        Args:
            classes_names (dict): Model classes names
            reg_pts (list): Initial list of points defining the counting region.
            count_reg_color (tuple): Color of object counting region
            count_txt_color (tuple): Count text color
            count_bg_color (tuple): Count highlighter line color
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Track line thickness for each track
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            draw_tracks (Bool): draw tracks
            track_color (tuple): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            count_type (str): Object counting type i.e. classwise or line counting
            colormap (cv2.COLORMAP): The colormap to be set opencv colormaps.
            heatmap_alpha (float): Alpha value for heatmap display
            count_reg_pts (list): Initial list of points defining the counting region.
            decay_factor (float): Value for removing heatmap area after object passed
            shape (str): Heatmap shape, rect or circle shape supported
            kpts_to_check (list): 3 keypoints for counting
        """

        # Print the solution initialized arguments data
        self.sargs_dict = {}
        solution = {k: v for k, v in locals().items() if k != 'self'}
        print(f"Solution: {solution}")

        # Display info if image width and height provided for heatmap
        if imw is not None:
            print("imw is deprecated and will be removed in future, shape will be handle automatically")
        if imh is not None:
            print("imh is deprecated and will be removed in future, shape will be handle automatically")

        self.classes_names = classes_names      # Store model classes names

        # Set counting region points, count_reg_pts is deprecated
        if count_reg_pts is not None:
            self.reg_pts = count_reg_pts
        else:
            self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts

        # Configure region points
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

        # Set color attributes i.e. region, background, text and track color
        self.count_reg_color = count_bg_color if count_reg_color is None else count_reg_color
        self.count_bg_color = count_bg_color
        self.count_txt_color = count_txt_color
        self.track_color = track_color

        # Set text and plotting objects, track and region thickness
        self.tf = line_thickness
        self.tt = track_thickness
        self.rt = region_thickness

        # Set display attributes i.e, image, in count, out count, track display
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.draw_tracks = draw_tracks

        self.ld_thresh = line_dist_thresh   # line distance threshold for line counting
        self.count_type = count_type    # set count type i.e. classwise or line

        # Set heatmap parameters
        self.colormap = cv2.COLORMAP_JET if colormap is None else colormap
        self.heatmap_alpha = heatmap_alpha
        self.decay_factor = decay_factor
        self.shape = shape

        # Set workouts monitoring parameters for ai_gym app
        self.workouts_counts = None
        self.stage = None
        self.angle = None
        self.kpts_to_check = [6, 8, 10] if kpts_to_check is None else kpts_to_check
        self.pose_type = "pushup" if pose_type is None else pose_type
        self.pose_up_angle = pose_up_angle
        self.pose_down_angle = pose_down_angle

        # Set speed estimation parameters for speed estimation app
        self.trk_previous_times = {}
        self.trk_previous_points = {}
        self.trk_idslist = []
        self.dist_data = {}
        spdl_dist_thresh = 10,

        # self.pixels_per_meter = pixels_per_meter

        self.im0 = None
        self.heatmap_im0 = None
        self.annotator = None
        self.counted_ids = []
        self.clswise_dict = {}

        self.boxes = None
        self.clss = None
        self.track_ids = None
        self.in_count = 0
        self.out_count = 0

        self.window_name = "Ultralytics Solutions"
        self.track_history = defaultdict(list)
        self.env_check = check_imshow(warn=True)


    def extract_tracks(self, tracks):
        """
        Extract tracks for advanced analytics, including bounding boxes, classes, and tracking IDs.
        Args:
            tracks (list): list of tracking data that comes from model.Track
        """

        if tracks[0].boxes.id is not None:
            self.boxes = tracks[0].boxes.xyxy.cpu()
            self.clss = tracks[0].boxes.cls.cpu().tolist()
            self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

    def object_counts(self, prev_position, box, cls, track_id, track_line):

        if self.count_type == "classwise":
            if self.classes_names[cls] not in self.clswise_dict:
                self.clswise_dict[self.classes_names[cls]] = {"IN": 0, "OUT": 0}

        if len(self.reg_pts) >= 3:
            is_inside = self.counting_region.contains(Point(track_line[-1]))
            if prev_position is not None and is_inside and track_id not in self.counted_ids:
                self.counted_ids.append(track_id)
                if (box[0] - prev_position[0]) * (self.counting_region.centroid.x -
                                                  prev_position[0]) > 0:
                    self.in_count += 1
                    if self.count_type == "classwise":
                        self.clswise_dict[self.classes_names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    if self.count_type == "classwise":
                        self.clswise_dict[self.classes_names[cls]]["OUT"] += 1

        elif len(self.reg_pts) == 2:
            if prev_position is not None and track_id not in self.counted_ids:
                distance = Point(track_line[-1]).distance(self.counting_region)
                if distance < self.ld_thresh and track_id not in self.counted_ids:
                    self.counted_ids.append(track_id)
                    if (box[0] - prev_position[0]) * (self.counting_region.centroid.x -
                                                      prev_position[0]) > 0:
                        self.in_count += 1
                        if self.count_type == "classwise":
                            self.clswise_dict[self.classes_names[cls]]["IN"] += 1
                    else:
                        self.out_count += 1
                        if self.count_type == "classwise":
                            self.clswise_dict[self.classes_names[cls]]["OUT"] += 1


    def display_frames(self, im0, window_name):
        """
        Display output image if platform supported display.

        Args:
            im0 (ndarray): The output image to be displayed if the environment supports display.
            window_name (str): The name of the output window; required by OpenCV.
        """

        if self.env_check:
            cv2.imshow(window_name, im0)    # Display output image
            if cv2.waitKey(1) & 0xFF == ord("q"):   # Break Window
                return

    def _convert_value(self, value):
        """
        Convert string representations of tuples to actual tuples, and other necessary conversions

        Args:
            value (str | int | float | bool): The dict value coming from dictionary

        """
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            return tuple(map(int, value[1:-1].split(',')))
        return value

    def set_args(self, **kwargs):
        """
        The 'Set Arguments' function is deprecated. However, if a user is running the code on a machine
        with a previous version of Ultralytics, updating the Ultralytics package will crash the code, this
        function will prevent the code from crashing and ensure smooth execution of the solution.

        Args:
            kwargs (dict): list of arguments user provided to execute solution
        """

        print("set_args is deprecated and will be removed in the future, pass arguments directly inside object i.e counter = object_counter.ObjectCounter(args)")
        for key, value in kwargs.items():
            if key == "imw" or key == "imh":
                print(f"{key} is deprecated and will be removed in the future, shape will be handle directly")
            elif key == "count_reg_pts":
                print(f"{key} is deprecated and will be removed in the future, use reg_pts instead")
                self.sargs_dict["reg_pts"] = value
            else:
                self.sargs_dict[key] = value
        self.__init__(**self.sargs_dict)
