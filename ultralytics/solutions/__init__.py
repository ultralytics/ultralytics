# Ultralytics YOLO 🚀, AGPL-3.0 license
import sys

import cv2
import numpy as np
from collections import defaultdict

from ..utils.plotting import Annotator, colors
from ultralytics.utils.checks import check_imshow, check_requirements

check_requirements("shapely>=2.0.0")
from shapely.geometry import LineString, Point, Polygon

# Dict and list initialization
counted_ids = []
clswise_dict = {}
counting_region = []
track_history = defaultdict(list)
env_check = check_imshow(warn=True)

# Variables initialization
tf: int = 0
in_count: int = 0
out_count: int = 0
d_thresh: int = 14
rg_pts: list = None
h_alpha: float = 0.5,
h_decay: float = 0.99,
cls_names: dict = None
display_in: bool = True
display_out: bool = True
display_img: bool = False
heatshape: str = "circle",
enable_count: bool = True
colormap = cv2.COLORMAP_JET
display_tracks: bool = True
count_type: str = "classwise"
bg_color_rgb: tuple = (104, 31, 17)
txt_color_rgb: tuple = (255, 255, 255)

pixels_per_meter = 10,
pose_type = "push up"
pose_up_angle = 90,
pose_down_angle = 145,
kpts_to_check = [6, 8, 10]


def configure(
        view_img=True,
        draw_tracks=True,
        view_in_counts=True,
        view_out_counts=True,
        enable_counting=True,
        region_pts=None,
        counts_type="line",
        names=None,
        bg_color=(104, 31, 17),
        txt_color=(255, 255, 255),
        line_thickness=2,
        line_dist_thresh=15,
        color_map=None,
        heat_shape="circle",
        heat_decay=0.99,
        heat_alpha=0.5,
        kpts_to_check=None,
        pose_up_angle=145.0,
        pose_down_angle=90.0,
        pose_type="pullup",
        pixels_per_meter=10,

):
    global display_img, bg_color_rgb, txt_color_rgb, cls_names, tf, sf, display_tracks, \
        display_in, display_out, d_thresh, rg_pts, counting_region, count_type, colormap, \
        heatshape, h_alpha, h_decay, enable_count

    display_img = view_img
    display_tracks = draw_tracks
    display_in = view_in_counts
    display_out = view_out_counts
    enable_count = enable_counting

    rg_pts = [(20, 400), (1260, 400)] if region_pts is None else region_pts
    configure_region(rg_pts)

    count_type = counts_type

    cls_names = names
    if cls_names is None:
        print("Model classes name required, you can pass as argument to sol.configure function i.e cls_names=model.names !!!")
        sys.exit()

    bg_color_rgb = bg_color
    txt_color_rgb = txt_color
    tf = line_thickness
    d_thresh = line_dist_thresh
    colormap = cv2.COLORMAP_JET if color_map is None else color_map
    if heat_shape not in {"circle", "rect"}:
        print("Unknown shape value provided, 'circle' & 'rect' supported")
        print("Using circle shape now")
        heatshape = "circle"
    else:
        heatshape = heat_shape
    h_alpha = heat_alpha
    h_decay = heat_decay

    # Distance calculation arguments
    pixels_per_meter = pixels_per_meter

    # Ai gym arguments
    pose_type = pose_type
    pose_up_angle = pose_up_angle
    pose_down_angle = pose_down_angle
    kpts_to_check = kpts_to_check if kpts_to_check is not None else [6, 8, 10]

    # Environment check
    env_check = check_imshow(warn=True)
    track_history = defaultdict(list)

    from . import object_counter, speed_estimation, heatmap, distance_calculation, queue_management


def configure_region(rg_pts):
    global counting_region
    if len(rg_pts) == 2:
        rg_pts = rg_pts
        counting_region = LineString(rg_pts)
    elif len(rg_pts) >= 3:
        rg_pts = rg_pts
        counting_region = Polygon(rg_pts)
    else:
        print("Invalid Region points provided, region_points "
              "must be 2 for lines or >= 3 for polygons.")
        print("Using Line Counter Now")
        rg_pts = [(20, 400), (1260, 400)]
        counting_region = LineString(rg_pts)


def extract_tracks(tracks):
    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        return boxes, clss, track_ids
    else:
        return 0, 0, None


def object_counts(prev_position, box, cls, track_id, track_line):
    global in_count, out_count
    if count_type == "classwise":
        if cls_names[cls] not in clswise_dict:
            clswise_dict[cls_names[cls]] = {"IN": 0, "OUT": 0}

    if len(rg_pts) >= 3:
        is_inside = counting_region.contains(Point(track_line[-1]))
        if prev_position is not None and is_inside and track_id not in counted_ids:
            counted_ids.append(track_id)
            if (box[0] - prev_position[0]) * (counting_region.centroid.x - prev_position[0]) > 0:
                in_count += 1
                if count_type == "classwise":
                    clswise_dict[cls_names[cls]]["IN"] += 1
            else:
                out_count += 1
                if count_type == "classwise":
                    clswise_dict[cls_names[cls]]["OUT"] += 1

    elif len(rg_pts) == 2:
        if prev_position is not None and track_id not in counted_ids:
            distance = Point(track_line[-1]).distance(counting_region)
            if distance < d_thresh and track_id not in counted_ids:
                counted_ids.append(track_id)
                if (box[0] - prev_position[0]) * (counting_region.centroid.x - prev_position[0]) > 0:
                    in_count += 1
                    if count_type == "classwise":
                        clswise_dict[cls_names[cls]]["IN"] += 1
                else:
                    out_count += 1
                    if count_type == "classwise":
                        clswise_dict[cls_names[cls]]["OUT"] += 1
    return in_count, out_count, clswise_dict


def display_frames(im0, window_name):
    """Display output image if platform supported display"""

    if env_check:
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, im0)
        # Break Window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
