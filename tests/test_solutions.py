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
REGION = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Solution test configs: (name, class, needs_frame_count, video, kwargs)
SOLUTIONS = [
    # Basic solutions
    (
        "ObjectCounter",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": "yolo11n.pt", "show": False},
    ),
    (
        "Heatmap",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "model": "yolo11n.pt", "show": False},
    ),
    (
        "HeatmapWithRegion",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "model": "yolo11n.pt", "show": False, "region": REGION},
    ),
    (
        "SpeedEstimator",
        solutions.SpeedEstimator,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": "yolo11n.pt", "show": False},
    ),
    (
        "QueueManager",
        solutions.QueueManager,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": "yolo11n.pt", "show": False},
    ),
    # Analytics solutions
    (
        "LineAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "line", "model": "yolo11n.pt", "show": False},
    ),
    (
        "PieAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "pie", "model": "yolo11n.pt", "show": False},
    ),
    (
        "BarAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "bar", "model": "yolo11n.pt", "show": False},
    ),
    (
        "AreaAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "area", "model": "yolo11n.pt", "show": False},
    ),
    # Advanced solutions
    ("TrackZone", solutions.TrackZone, False, DEMO_VIDEO, {"region": REGION, "model": "yolo11n.pt", "show": False}),
    (
        "ObjectCropper",
        solutions.ObjectCropper,
        False,
        DEMO_VIDEO,
        {"model": "yolo11n.pt", "show": False, "crop_dir": str(TMP / "cropped-detections")},
    ),
    (
        "ObjectBlurrer",
        solutions.ObjectBlurrer,
        False,
        DEMO_VIDEO,
        {"blur_ratio": 0.5, "model": "yolo11n.pt", "show": False},
    ),
    (
        "InstanceSegmentation",
        solutions.InstanceSegmentation,
        False,
        DEMO_VIDEO,
        {"model": "yolo11n-seg.pt", "show": False},
    ),
    ("VisionEye", solutions.VisionEye, False, DEMO_VIDEO, {"model": "yolo11n.pt", "show": False}),
    (
        "RegionCounter",
        solutions.RegionCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": "yolo11n.pt", "show": False},
    ),
    # Special video case
    ("AIGym", solutions.AIGym, False, POSE_VIDEO, {"kpts": [5, 11, 13], "show": False}),
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


@pytest.mark.slow
@pytest.mark.parametrize("name, solution_class, needs_frame_count, video, kwargs", SOLUTIONS)
def test_solution(name, solution_class, needs_frame_count, video, kwargs):
    """Test individual Ultralytics solution."""
    safe_download(url=f"{ASSETS_URL}/{video}", dir=TMP)
    solution = solution_class(**kwargs)
    process_video(solution, str(TMP / video), needs_frame_count)


@pytest.mark.slow
def test_parking_management():
    """Test ParkingManagement solution."""
    safe_download(url=f"{ASSETS_URL}/{PARKING_VIDEO}", dir=TMP)
    safe_download(url=f"{ASSETS_URL}/{PARKING_AREAS_JSON}", dir=TMP)
    safe_download(url=f"{ASSETS_URL}/{PARKING_MODEL}", dir=TMP)

    solution = solutions.ParkingManagement(
        json_file=str(TMP / PARKING_AREAS_JSON), model=str(TMP / PARKING_MODEL), show=False
    )
    process_video(solution, str(TMP / PARKING_VIDEO))


# @pytest.mark.slow
def test_streamlit_inference():
    """Test streamlit predict live inference solution."""
    solutions.Inference().inference()
