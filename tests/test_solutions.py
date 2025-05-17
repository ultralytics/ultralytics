# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Tests Ultralytics Solutions: https://docs.ultralytics.com/solutions/,
# including every solution excluding DistanceCalculation and Security Alarm System.

import cv2
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
HORIZONTAL_LINE = [(10, 200), (540, 200)]  # for object counting
VERTICAL_LINE = [(200, 0), (200, 400)]  # for object counting

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
        {"region": HORIZONTAL_LINE, "model": MODEL, "show": SHOW},
    ),
    (
        "ObjectCounter",
        solutions.ObjectCounter,
        False,
        DEMO_VIDEO,
        {"region": VERTICAL_LINE, "model": MODEL, "show": SHOW},
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
        {"blur_ratio": 0.02, "model": MODEL, "show": SHOW},
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


def test_analytics_graph_not_supported():
    """Test for analytical graph not supported."""
    try:
        analytics = solutions.Analytics(analytics_type="test")  # 'test' is unsupported
        analytics.process(im0=None, frame_number=0)
        assert False, "Expected ModuleNotFoundError for unsupported chart type"
    except ModuleNotFoundError as e:
        assert "test chart is not supported" in str(e)


def test_area_chart_padding():
    """Test area chart updates with padding logic for coverage."""
    analytics = solutions.Analytics(analytics_type="area")
    analytics.update_graph(frame_number=1, count_dict={"car": 2}, plot="area")
    plot_im = analytics.update_graph(frame_number=2, count_dict={"car": 3, "person": 1}, plot="area")
    assert plot_im is not None


def test_update_invalid_argument():
    """Test update method with an invalid keyword argument."""
    obj = solutions.config.SolutionConfig()
    try:
        obj.update(invalid_key=123)
        assert False, "Expected ValueError for invalid update argument"
    except ValueError as e:
        assert "❌ invalid_key is not a valid solution argument" in str(e)


def test_plot_with_no_masks():
    """Test instance segmentation with no masks"""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    isegment = solutions.InstanceSegmentation(model="yolo11n-seg.pt")
    results = isegment(im0)
    assert results.plot_im is not None

def test_parking_json_none():
    """Test instance segmentation with no masks"""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    parkingmanager = solutions.ParkingManagement()
    with pytest.raises(ValueError, match="Json file path can not be empty"):
        parkingmanager(im0)


def test_left_click_selection():
    """Test distance calculation left click"""
    dc = solutions.DistanceCalculation()
    dc.boxes = [[10, 10, 50, 50]]
    dc.track_ids = [1]
    dc.mouse_event_for_distance(cv2.EVENT_LBUTTONDOWN, 30, 30, None, None)
    assert 1 in dc.selected_boxes


def test_right_click_reset():
    """Test distance calculation right click"""
    dc = solutions.DistanceCalculation()
    dc.selected_boxes = {1: [10, 10, 50, 50]}
    dc.left_mouse_count = 1
    dc.mouse_event_for_distance(cv2.EVENT_RBUTTONDOWN, 0, 0, None, None)
    assert dc.selected_boxes == {}
    assert dc.left_mouse_count == 0
