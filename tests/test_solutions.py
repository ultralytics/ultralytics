# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# This is file for Ultralytics Solutions tests: https://docs.ultralytics.com/solutions/,
# It includes every solution excluding DistanceCalculation and Security Alarm System.

import cv2
import pytest

from tests import TMP
from ultralytics import solutions
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import safe_download

DEMO_VIDEO = "solutions_ci_demo.mp4"  # for all the solutions, except workout and parking
POSE_VIDEO = "solution_ci_pose_demo.mp4"  # only for workouts monitoring solution
PARKING_VIDEO = "solution_ci_parking_demo.mp4"  # only for parking management solution
PARKING_AREAS_JSON = "solution_ci_parking_areas.json"  # only for parking management solution
PARKING_MODEL = "solutions_ci_parking_model.pt"  # only for parking management solution

# Pre-defined arguments values
SHOW = False
MODEL_FILE = "yolo11n.pt"  # model file used for solutions, except parking management and instance segmentation
REGION = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # for object counting, speed estimation and queue management

# Solution test configs: (name, class, needs_frame_count, video, kwargs)
SOLUTIONS = [
    # Object counting test
    # (
    #     "ObjectCounter",
    #     solutions.ObjectCounter,
    #     False,
    #     DEMO_VIDEO,
    #     {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    # ),
    # Heatmap test
    (
        "Heatmap",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "model": MODEL_FILE, "show": SHOW},
    ),
    # # Heatmap with object counting test
    # (
    #     "HeatmapWithRegion",
    #     solutions.Heatmap,
    #     False,
    #     DEMO_VIDEO,
    #     {"colormap": cv2.COLORMAP_PARULA, "region": REGION, "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Speedestimation test
    # (
    #     "SpeedEstimator",
    #     solutions.SpeedEstimator,
    #     False,
    #     DEMO_VIDEO,
    #     {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Queue management test
    # (
    #     "QueueManager",
    #     solutions.QueueManager,
    #     False,
    #     DEMO_VIDEO,
    #     {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Line analytics test
    # (
    #     "LineAnalytics",
    #     solutions.Analytics,
    #     True,
    #     DEMO_VIDEO,
    #     {"analytics_type": "line", "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Pie analytics test
    # (
    #     "PieAnalytics",
    #     solutions.Analytics,
    #     True,
    #     DEMO_VIDEO,
    #     {"analytics_type": "pie", "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Bar analytics test
    # (
    #     "BarAnalytics",
    #     solutions.Analytics,
    #     True,
    #     DEMO_VIDEO,
    #     {"analytics_type": "bar", "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Area analytics test
    # (
    #     "AreaAnalytics",
    #     solutions.Analytics,
    #     True,
    #     DEMO_VIDEO,
    #     {"analytics_type": "area", "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Trackzone test
    # ("TrackZone", solutions.TrackZone, False, DEMO_VIDEO, {"region": REGION, "model": MODEL_FILE, "show": SHOW}),
    # # Object crop test
    # (
    #     "ObjectCropper",
    #     solutions.ObjectCropper,
    #     False,
    #     DEMO_VIDEO,
    #     {"crop_dir": str(TMP / "cropped-detections"), "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Object blur test
    # (
    #     "ObjectBlurrer",
    #     solutions.ObjectBlurrer,
    #     False,
    #     DEMO_VIDEO,
    #     {"blur_ratio": 0.5, "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Instance segmentation test
    # (
    #     "InstanceSegmentation",
    #     solutions.InstanceSegmentation,
    #     False,
    #     DEMO_VIDEO,
    #     {"model": "yolo11n-seg.pt", "show": SHOW},
    # ),
    # # Visioneye test
    # ("VisionEye", solutions.VisionEye, False, DEMO_VIDEO, {"model": MODEL_FILE, "show": SHOW}),
    # # Regioncounter test
    # (
    #     "RegionCounter",
    #     solutions.RegionCounter,
    #     False,
    #     DEMO_VIDEO,
    #     {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    # ),
    # # Special video case
    # ("AIGym", solutions.AIGym, False, POSE_VIDEO, {"kpts": [5, 11, 13], "show": SHOW}),
]


def process_video(solution, video_path, needs_frame_count=False):
    """Process video with solution, feeding frames and optional frame count."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file {video_path}"

    frame_count = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        frame_count += 1
        args = [im0.copy(), frame_count] if needs_frame_count else [im0.copy()]
        _ = solution(*args)

    cap.release()


# @pytest.mark.slow
@pytest.mark.parametrize("name, solution_class, needs_frame_count, video, kwargs", SOLUTIONS)
def test_solution(name, solution_class, needs_frame_count, video, kwargs):
    """Test individual Ultralytics solution."""
    safe_download(url=f"{ASSETS_URL}/{video}", dir=TMP)
    print(solution_class)
    solution = solution_class(**kwargs)
    process_video(solution, str(TMP / video), needs_frame_count)


# @pytest.mark.slow
# def test_parking_management():
#     """Test ParkingManagement solution."""
#     safe_download(url=f"{ASSETS_URL}/{PARKING_VIDEO}", dir=TMP)
#     safe_download(url=f"{ASSETS_URL}/{PARKING_AREAS_JSON}", dir=TMP)
#     safe_download(url=f"{ASSETS_URL}/{PARKING_MODEL}", dir=TMP)

#     solution = solutions.ParkingManagement(
#         json_file=str(TMP / PARKING_AREAS_JSON), model=str(TMP / PARKING_MODEL), show=False
#     )
#     process_video(solution, str(TMP / PARKING_VIDEO))


# @pytest.mark.slow
# def test_streamlit_inference():
#     """Test streamlit predict live inference solution."""
#     solutions.Inference().inference()
