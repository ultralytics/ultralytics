# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Tests Ultralytics Solutions: https://docs.ultralytics.com/solutions/,
# including every solution excluding DistanceCalculation and Security Alarm System.

import cv2
import numpy as np
import pytest

from tests import MODEL, TMP
from ultralytics import solutions
from ultralytics.utils import ASSETS_URL, IS_RASPBERRYPI, LINUX, checks
from ultralytics.utils.downloads import safe_download

# Pre-defined arguments values
SHOW = False
DEMO_VIDEO = "solutions_ci_demo.mp4"  # for all the solutions, except workout, object cropping and parking management
CROP_VIDEO = "decelera_landscape_min.mov"  # for object cropping solution
POSE_VIDEO = "solution_ci_pose_demo.mp4"  # only for workouts monitoring solution
PARKING_VIDEO = "solution_ci_parking_demo.mp4"  # only for parking management solution
PARKING_AREAS_JSON = "solution_ci_parking_areas.json"  # only for parking management solution
PARKING_MODEL = "solutions_ci_parking_model.pt"  # only for parking management solution
REGION = [(10, 200), (540, 200), (540, 180), (10, 180)]  # for object counting, speed estimation and queue management

# Test configs for each solution : (name, class, needs_frame_count, video, kwargs)
SOLUTIONS = [
    (
        "ObjectCounter",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL, "show": SHOW},
    ),
(
        "ObjectCounter",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL, "show": SHOW},
    ),
    (
        "ObjectCounterwithOBB",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": "yolo11n-obb.pt", "show": SHOW},
    ),
    (
        "Heatmap",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "model": MODEL, "show": SHOW, "region": None},
    ),
    (
        "HeatmapWithRegion",
        solutions.Heatmap,
        False,
        DEMO_VIDEO,
        {"colormap": cv2.COLORMAP_PARULA, "region": REGION, "model": MODEL, "show": SHOW},
    ),
    (
        "SpeedEstimator",
        solutions.SpeedEstimator,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL, "show": SHOW},
    ),
    (
        "QueueManager",
        solutions.QueueManager,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL, "show": SHOW},
    ),
    (
        "LineAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "line", "model": MODEL, "show": SHOW},
    ),
    (
        "PieAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "pie", "model": MODEL, "show": SHOW},
    ),
    (
        "BarAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "bar", "model": MODEL, "show": SHOW},
    ),
    (
        "AreaAnalytics",
        solutions.Analytics,
        True,
        DEMO_VIDEO,
        {"analytics_type": "area", "model": MODEL, "show": SHOW},
    ),
    ("TrackZone", solutions.TrackZone, False, DEMO_VIDEO, {"region": REGION, "model": MODEL, "show": SHOW}),
    (
        "ObjectCropper",
        solutions.ObjectCropper,
        False,
        CROP_VIDEO,
        {"crop_dir": str(TMP / "cropped-detections"), "model": MODEL, "show": SHOW},
    ),
    (
        "ObjectBlurrer",
        solutions.ObjectBlurrer,
        False,
        DEMO_VIDEO,
        {"blur_ratio": 0.5, "model": MODEL, "show": SHOW},
    ),
    (
        "InstanceSegmentation",
        solutions.InstanceSegmentation,
        False,
        DEMO_VIDEO,
        {"model": "yolo11n-seg.pt", "show": SHOW},
    ),
    ("VisionEye", solutions.VisionEye, False, DEMO_VIDEO, {"model": MODEL, "show": SHOW}),
    (
        "RegionCounter",
        solutions.RegionCounter,
        False,
        DEMO_VIDEO,
        {"region": REGION, "model": MODEL, "show": SHOW},
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


@pytest.mark.skipif(
    (LINUX and checks.IS_PYTHON_3_11) or IS_RASPBERRYPI,
    reason="Disabled for testing due to --slow test errors after YOLOE PR.",
)
@pytest.mark.parametrize("name, solution_class, needs_frame_count, video, kwargs", SOLUTIONS)
def test_solution(name, solution_class, needs_frame_count, video, kwargs):
    """Test individual Ultralytics solution."""
    if video:
        safe_download(url=f"{ASSETS_URL}/{video}", dir=TMP)
    if name == "ParkingManager":
        safe_download(url=f"{ASSETS_URL}/{PARKING_AREAS_JSON}", dir=TMP)
        safe_download(url=f"{ASSETS_URL}/{PARKING_MODEL}", dir=TMP)
    elif name == "StreamlitInference":
        if checks.check_imshow():  # do not merge with elif above
            solution_class(**kwargs).inference()  # requires interactive GUI environment
        return

    process_video(
        solution=solution_class(**kwargs),
        video_path=str(TMP / video),
        needs_frame_count=needs_frame_count,
    )


@pytest.mark.slow
@pytest.mark.skipif(checks.IS_PYTHON_3_8, reason="Disabled due to unsupported CLIP dependencies.")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search():
    """Test similarity search solution."""
    from ultralytics import solutions

    searcher = solutions.VisualAISearch()
    _ = searcher("a dog sitting on a bench")  # Returns the results in format "- img name | similarity score"


def test_mouse_event_select_inside_box():
    """Tests that a left-click inside a box selects the object."""
    dc = solutions.DistanceCalculation()
    box, track_id = [100, 100, 200, 200], 1
    dc.boxes, dc.track_ids = [box], [track_id]
    dc.mouse_event_for_distance(event=cv2.EVENT_LBUTTONDOWN, x=150, y=150, flags=None, param=None)
    assert track_id in dc.selected_boxes


def test_mouse_event_click_outside_box():
    """Tests that a click outside a box does not select the object."""
    dc = solutions.DistanceCalculation()
    box, track_id = [100, 100, 200, 200], 1
    dc.boxes, dc.track_ids = [box], [track_id]
    dc.mouse_event_for_distance(event=cv2.EVENT_LBUTTONDOWN, x=250, y=250, flags=None, param=None)
    assert track_id not in dc.selected_boxes


def test_mouse_event_right_click_ignored():
    """Tests that right-click does not select any object."""
    dc = solutions.DistanceCalculation()
    box, track_id = [100, 100, 200, 200], 1
    dc.boxes, dc.track_ids = [box], [track_id]
    dc.mouse_event_for_distance(event=cv2.EVENT_RBUTTONDOWN, x=150, y=150, flags=None, param=None)
    assert dc.selected_boxes == {}


def test_distance_process_output():
    """Tests that the process() method computes distance correctly between selected objects."""
    dc = solutions.DistanceCalculation()
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    box1 = [100, 100, 200, 200]
    box2 = [300, 300, 400, 400]
    dc.boxes = [box1, box2]
    dc.track_ids = [1, 2]
    dc.selected_boxes = {1: box1, 2: box2}
    result = dc.process(dummy_image)
    assert isinstance(result, np.ndarray), "Output should be an image array"
    assert result.shape == dummy_image.shape, "Output image should match input size"


def test_count_objects_linear_region():
    """Test object counting across a linear region."""
    counter = solutions.ObjectCounter()
    counter.names = {0: 'person'}
    counter.region = [(100, 100), (200, 100)]  # Horizontal line
    counter.classwise_counts = {'person': {'IN': 0, 'OUT': 0}}
    prev_position = (90, 100)
    current_centroid = (110, 100)
    track_id = 1
    cls = 0
    counter.count_objects(current_centroid, track_id, prev_position, cls)
    assert counter.in_count == 1
    assert counter.classwise_counts['person']['IN'] == 1
    assert track_id in counter.counted_ids


def test_count_objects_polygonal_region():
    """Test object counting within a polygonal region."""
    counter = solutions.ObjectCounter()
    counter.names = {0: 'person'}
    counter.region = [(100, 100), (200, 100), (200, 200), (100, 200)]  # Square
    counter.classwise_counts = {'person': {'IN': 0, 'OUT': 0}}
    prev_position = (90, 90)
    current_centroid = (150, 150)
    track_id = 2
    cls = 0
    counter.count_objects(current_centroid, track_id, prev_position, cls)
    assert counter.in_count == 1
    assert counter.classwise_counts['person']['IN'] == 1
    assert track_id in counter.counted_ids

def test_display_counts():
    """Test that display_counts executes without errors."""
    counter = solutions.ObjectCounter()
    counter.classwise_counts = {'person': {'IN': 1, 'OUT': 0}}
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    counter.display_counts(dummy_image)
    # No assertion needed; test passes if no exceptions are raised
