# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Tests Ultralytics Solutions: https://docs.ultralytics.com/solutions/,
# including every solution excluding DistanceCalculation and Security Alarm System.

import cv2
import pytest

from tests import TMP
from ultralytics import solutions
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import safe_download

# Pre-defined arguments values
SHOW = False
DEMO_VIDEO = "solutions_ci_demo.mp4"  # for all the solutions, except workout, object cropping and parking management
CROP_VIDEO = "decelera_landscape_min.mov"  # for object cropping solution
POSE_VIDEO = "solution_ci_pose_demo.mp4"  # only for workouts monitoring solution
PARKING_VIDEO = "solution_ci_parking_demo.mp4"  # only for parking management solution
PARKING_AREAS_JSON = "solution_ci_parking_areas.json"  # only for parking management solution
PARKING_MODEL = "solutions_ci_parking_model.pt"  # only for parking management solution
MODEL_FILE = "yolo11n.pt"  # model file used for solutions, except parking management and instance segmentation
REGION = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # for object counting, speed estimation and queue management

# Test configs for each solution : (name, class, needs_frame_count, video, kwargs)
SOLUTIONS = [
    (
        "ObjectCounter",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "Heatmap",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "model": MODEL_FILE, "show": SHOW, "region": None},
    ),
    (
        "HeatmapWithRegion",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "region": REGION, "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "SpeedEstimator",
        solutions.SpeedEstimator,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "QueueManager",
        solutions.QueueManager,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "LineAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "line", "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "PieAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "pie", "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "BarAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "bar", "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "AreaAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "area", "model": MODEL_FILE, "show": SHOW},
    ),
    ("TrackZone", solutions.TrackZone, False, DEMO_VIDEO, {"region": REGION, "model": MODEL_FILE, "show": SHOW}),
    (
        "ObjectCropper",
        solutions.ObjectCropper,
        False,
        CROP_VIDEO,
        {"crop_dir": str(TMP / "cropped-detections"), "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "ObjectBlurrer",
        solutions.ObjectBlurrer,
        False,
        DEMO_VIDEO,
        {"blur_ratio": 0.5, "model": MODEL_FILE, "show": SHOW},
    ),
    (
        "InstanceSegmentation",
        solutions.InstanceSegmentation,
        False,
        DEMO_VIDEO,
        {"model": "yolo11n-seg.pt", "show": SHOW},
    ),
    ("VisionEye", solutions.VisionEye, False, DEMO_VIDEO, {"model": MODEL_FILE, "show": SHOW}),
    (
        "RegionCounter",
        solutions.RegionCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL_FILE, "show": SHOW},
    ),
    ("AIGym", solutions.AIGym, False, POSE_VIDEO, {"kpts": [6, 8, 10], "show": SHOW}),
    (
        "ParkingManager",
        solutions.ParkingManagement,
        False,
        PARKING_VIDEO,
        {"model": str(TMP / PARKING_MODEL), "show": SHOW, "json_file": str(TMP / PARKING_AREAS_JSON)},
    ),
    (
        "StreamlitInference",
        solutions.Inference,
        False,
        None,  # streamlit application don't require video file
        {},  # streamlit application don't accept arguments
    ),
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
        im_copy = im0.copy()
        args = [im_copy, frame_count] if needs_frame_count else [im_copy]
        _ = solution(*args)

    cap.release()


@pytest.mark.slow
@pytest.mark.parametrize("name, solution_class, needs_frame_count, video, kwargs", SOLUTIONS)
def test_solution(name, solution_class, needs_frame_count, video, kwargs):
    """Test individual Ultralytics solution."""
    safe_download(url=f"{ASSETS_URL}/{video}", dir=TMP)
    if name == "ParkingManager":
        safe_download(url=f"{ASSETS_URL}/{PARKING_AREAS_JSON}", dir=TMP)
        safe_download(url=f"{ASSETS_URL}/{PARKING_MODEL}", dir=TMP)
    solution = solution_class(**kwargs)
    solution.inference() if name == "StreamlitInference" else process_video(
        solution, str(TMP / video), needs_frame_count
    )
