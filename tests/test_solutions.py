# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Tests Ultralytics Solutions: https://docs.ultralytics.com/solutions/,
# Includes all solutions except DistanceCalculation and the Security Alarm System.

import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tests import MODEL
from ultralytics import solutions
from ultralytics.utils import ASSETS_URL, IS_RASPBERRYPI, TORCH_VERSION, checks
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.torch_utils import TORCH_2_4

# Predefined argument values
SHOW = False
REGION = [(10, 200), (540, 200), (540, 180), (10, 180)]  # for object counting, speed estimation and queue management
HORIZONTAL_LINE = [(10, 200), (540, 200)]  # for object counting
VERTICAL_LINE = [(320, 0), (320, 400)]  # for object counting


def process_video(solution, video_path: str, needs_frame_count: bool = False):
    """Process video with solution, feeding frames and optional frame count to the solution instance."""
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


@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled for testing due to --slow test errors after YOLOE PR.")
@pytest.mark.parametrize(
    "name, solution_class, needs_frame_count, video_key, kwargs_update",
    [
        (
            "ObjectCounter",
            solutions.ObjectCounter,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounter",
            solutions.ObjectCounter,
            False,
            "demo_video",
            {"region": HORIZONTAL_LINE, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounterVertical",
            solutions.ObjectCounter,
            False,
            "vertical_video",
            {"region": VERTICAL_LINE, "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectCounterwithOBB",
            solutions.ObjectCounter,
            False,
            "demo_video",
            {"region": REGION, "model": "yolo26n-obb.pt", "show": SHOW},
        ),
        (
            "Heatmap",
            solutions.Heatmap,
            False,
            "demo_video",
            {"colormap": cv2.COLORMAP_PARULA, "model": MODEL, "show": SHOW, "region": None},
        ),
        (
            "HeatmapWithRegion",
            solutions.Heatmap,
            False,
            "demo_video",
            {"colormap": cv2.COLORMAP_PARULA, "region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "SpeedEstimator",
            solutions.SpeedEstimator,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "QueueManager",
            solutions.QueueManager,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        (
            "LineAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "line", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "PieAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "pie", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "BarAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "bar", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        (
            "AreaAnalytics",
            solutions.Analytics,
            True,
            "demo_video",
            {"analytics_type": "area", "model": MODEL, "show": SHOW, "figsize": (6.4, 3.2)},
        ),
        ("TrackZone", solutions.TrackZone, False, "demo_video", {"region": REGION, "model": MODEL, "show": SHOW}),
        (
            "ObjectCropper",
            solutions.ObjectCropper,
            False,
            "crop_video",
            {"temp_crop_dir": "cropped-detections", "model": MODEL, "show": SHOW},
        ),
        (
            "ObjectBlurrer",
            solutions.ObjectBlurrer,
            False,
            "demo_video",
            {"blur_ratio": 0.02, "model": MODEL, "show": SHOW},
        ),
        (
            "InstanceSegmentation",
            solutions.InstanceSegmentation,
            False,
            "demo_video",
            {"model": "yolo26n-seg.pt", "show": SHOW},
        ),
        ("VisionEye", solutions.VisionEye, False, "demo_video", {"model": MODEL, "show": SHOW}),
        (
            "RegionCounter",
            solutions.RegionCounter,
            False,
            "demo_video",
            {"region": REGION, "model": MODEL, "show": SHOW},
        ),
        ("AIGym", solutions.AIGym, False, "pose_video", {"kpts": [6, 8, 10], "show": SHOW}),
        (
            "ParkingManager",
            solutions.ParkingManagement,
            False,
            "parking_video",
            {"model": "parking_model", "show": SHOW, "json_file": "parking_areas"},
        ),
        (
            "StreamlitInference",
            solutions.Inference,
            False,
            None,  # streamlit application doesn't require video file
            {},  # streamlit application doesn't accept arguments
        ),
    ],
)
def test_solution(name, solution_class, needs_frame_count, video_key, kwargs_update, tmp_path, solution_assets):
    """Test individual Ultralytics solution with video processing and parameter validation."""
    # Get video path from persistent cache (no copying needed, read-only access)
    video_path = str(solution_assets(video_key)) if video_key else None

    # Update kwargs to use cached paths for parking manager
    kwargs = {}
    for key, value in kwargs_update.items():
        if key.startswith("temp_"):
            kwargs[key.replace("temp_", "")] = str(tmp_path / value)
        elif value == "parking_model":
            kwargs[key] = str(solution_assets("parking_model"))
        elif value == "parking_areas":
            kwargs[key] = str(solution_assets("parking_areas"))
        else:
            kwargs[key] = value

    if name == "StreamlitInference":
        if checks.check_imshow():  # do not merge with elif above
            solution_class(**kwargs).inference()  # requires interactive GUI environment
        return

    process_video(
        solution=solution_class(**kwargs),
        video_path=video_path,
        needs_frame_count=needs_frame_count,
    )


def test_left_click_selection():
    """Test distance calculation left click selection functionality."""
    dc = solutions.DistanceCalculation()
    dc.boxes, dc.track_ids = [[10, 10, 50, 50]], [1]
    dc.mouse_event_for_distance(cv2.EVENT_LBUTTONDOWN, 30, 30, None, None)
    assert 1 in dc.selected_boxes, f"Expected track_id 1 in selected_boxes, got {dc.selected_boxes}"


def test_right_click_reset():
    """Test distance calculation right click reset functionality."""
    dc = solutions.DistanceCalculation()
    dc.selected_boxes, dc.left_mouse_count = {1: [10, 10, 50, 50]}, 1
    dc.mouse_event_for_distance(cv2.EVENT_RBUTTONDOWN, 0, 0, None, None)
    assert not dc.selected_boxes, f"Expected empty selected_boxes after reset, got {dc.selected_boxes}"
    assert dc.left_mouse_count == 0, f"Expected left_mouse_count=0 after reset, got {dc.left_mouse_count}"


def test_parking_json_none():
    """Test that ParkingManagement handles missing JSON gracefully."""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    try:
        parkingmanager = solutions.ParkingManagement(json_path=None)
        parkingmanager(im0)
    except ValueError:
        pytest.skip("Skipping test due to missing JSON.")


def test_analytics_graph_not_supported():
    """Test that unsupported analytics type raises ValueError."""
    try:
        analytics = solutions.Analytics(analytics_type="test")  # 'test' is unsupported
        analytics.process(im0=np.zeros((640, 480, 3), dtype=np.uint8), frame_number=0)
        assert False, "Expected ValueError for unsupported chart type"
    except ValueError as e:
        assert "Unsupported analytics_type" in str(e), f"Expected 'Unsupported analytics_type' in error, got: {e}"


def test_area_chart_padding():
    """Test area chart graph update with dynamic class padding logic."""
    analytics = solutions.Analytics(analytics_type="area")
    analytics.update_graph(frame_number=1, count_dict={"car": 2}, plot="area")
    plot_im = analytics.update_graph(frame_number=2, count_dict={"car": 3, "person": 1}, plot="area")
    assert plot_im is not None, "Area chart plot returned None"


def test_config_update_method_with_invalid_argument():
    """Test that update() raises ValueError for invalid config keys."""
    obj = solutions.config.SolutionConfig()
    try:
        obj.update(invalid_key=123)
        assert False, "Expected ValueError for invalid update argument"
    except ValueError as e:
        assert "is not a valid solution argument" in str(e), f"Expected validation error message, got: {e}"


def test_plot_with_no_masks():
    """Test that instance segmentation handles cases with no masks."""
    im0 = np.zeros((640, 480, 3), dtype=np.uint8)
    isegment = solutions.InstanceSegmentation(model="yolo26n-seg.pt")
    results = isegment(im0)
    assert results.plot_im is not None, "Instance segmentation plot returned None"


def test_streamlit_handle_video_upload_creates_file():
    """Test Streamlit video upload logic saves file correctly."""
    import io

    fake_file = io.BytesIO(b"fake video content")
    fake_file.read = fake_file.getvalue
    if fake_file is not None:
        g = io.BytesIO(fake_file.read())
        with open("ultralytics.mp4", "wb") as out:
            out.write(g.read())
        output_path = "ultralytics.mp4"
    else:
        output_path = None
    assert output_path == "ultralytics.mp4", f"Expected output_path 'ultralytics.mp4', got {output_path}"
    assert os.path.exists("ultralytics.mp4"), "ultralytics.mp4 file not created"
    with open("ultralytics.mp4", "rb") as f:
        content = f.read()
        assert content == b"fake video content", f"File content mismatch: {content}"
    os.remove("ultralytics.mp4")


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search(tmp_path):
    """Test similarity search solution with sample images and text query."""
    safe_download(f"{ASSETS_URL}/4-imgs-similaritysearch.zip", dir=tmp_path)  # 4 dog images for testing in a zip file
    searcher = solutions.VisualAISearch(data=str(tmp_path / "4-imgs-similaritysearch"))
    _ = searcher("a dog sitting on a bench")  # Returns the results in format "- img name | similarity score"


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search_app_init():
    """Test SearchApp initializes with required attributes."""
    app = solutions.SearchApp(device="cpu")
    assert hasattr(app, "searcher")
    assert hasattr(app, "run")


@pytest.mark.skipif(not TORCH_2_4, reason=f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})")
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Disabled due to slow performance on Raspberry Pi.")
def test_similarity_search_complete(tmp_path):
    """Test VisualAISearch end-to-end with sample images and query."""
    from PIL import Image

    image_dir = tmp_path / "images"
    os.makedirs(image_dir, exist_ok=True)
    for i in range(2):
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img.save(image_dir / f"test_image_{i}.jpg")
    searcher = solutions.VisualAISearch(data=str(image_dir))
    results = searcher("a red and white object")
    assert results, "Similarity search returned empty results"


def test_distance_calculation_process_method():
    """Test DistanceCalculation.process() computes distance between selected boxes."""
    from ultralytics.solutions.solutions import SolutionResults

    dc = solutions.DistanceCalculation()
    dc.boxes, dc.track_ids, dc.clss, dc.confs = (
        [[100, 100, 200, 200], [300, 300, 400, 400]],
        [1, 2],
        [0, 0],
        [0.9, 0.95],
    )
    dc.selected_boxes = {1: dc.boxes[0], 2: dc.boxes[1]}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(dc, "extract_tracks"), patch.object(dc, "display_output"), patch("cv2.setMouseCallback"):
        result = dc.process(frame)
    assert isinstance(result, SolutionResults), f"Expected SolutionResults, got {type(result)}"
    assert result.total_tracks == 2, f"Expected 2 tracks, got {result.total_tracks}"
    assert result.pixels_distance > 0, f"Expected positive distance, got {result.pixels_distance}"


def test_object_crop_with_show_True():
    """Test ObjectCropper init with show=True to cover display warning."""
    solutions.ObjectCropper(show=True)


def test_display_output_method():
    """Test that display_output triggers imshow, waitKey, and destroyAllWindows when enabled."""
    counter = solutions.ObjectCounter(show=True)
    counter.env_check = True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("cv2.imshow") as mock_imshow, patch("cv2.waitKey", return_value=ord("q")) as mock_wait, patch(
        "cv2.destroyAllWindows"
    ) as mock_destroy:
        counter.display_output(frame)
        mock_imshow.assert_called_once()
        mock_wait.assert_called_once()
        mock_destroy.assert_called_once()
