# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import pytest

from ultralytics import YOLO, solutions
from ultralytics.utils.downloads import safe_download

MAJOR_SOLUTIONS_DEMO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
WORKOUTS_SOLUTION_DEMO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solution_ci_pose_demo.mp4"


@pytest.mark.slow
def test_major_solutions():
    """
    Test the object counting, heatmap generation, speed estimation, queue management, various analytics solutions, and
    DwellTimeAnalyzer.

    This test downloads demo videos, initializes different solution instances including the DwellTimeAnalyzer, processes
    each frame of the videos, and ensures that each solution operates without errors.
    """
    # Download the major solutions demo video
    safe_download(url=MAJOR_SOLUTIONS_DEMO)
    cap = cv2.VideoCapture("solutions_ci_demo.mp4")
    assert cap.isOpened(), "Error reading solutions_ci_demo.mp4 video file"

    # Define the region of interest
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

    # Initialize solution instances
    counter = solutions.ObjectCounter(region=region_points, model="yolo11n.pt", show=False)  # Test object counter
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False)  # Test heatmaps
    speed = solutions.SpeedEstimator(region=region_points, model="yolo11n.pt", show=False)  # Test speed estimator
    queue = solutions.QueueManager(region=region_points, model="yolo11n.pt", show=False)  # Test queue manager
    line_analytics = solutions.Analytics(analytics_type="line", model="yolo11n.pt", show=False)  # Line analytics
    pie_analytics = solutions.Analytics(analytics_type="pie", model="yolo11n.pt", show=False)  # Pie analytics
    bar_analytics = solutions.Analytics(analytics_type="bar", model="yolo11n.pt", show=False)  # Bar analytics
    area_analytics = solutions.Analytics(analytics_type="area", model="yolo11n.pt", show=False)  # Area analytics

    # Download the DwellTimeAnalyzer demo video
    dwell_cap = cv2.VideoCapture("solutions_ci_demo.mp4")
    assert dwell_cap.isOpened(), "Error reading dwell_time_analyzer_demo.mp4 video file"

    # Define configuration for DwellTimeAnalyzer
    dwell_config = {
        "fps": 30.0,
        "classes": [0, 1, 2],  # Example class indices to track
        "zones": {
            "Entrance": [(100, 200), (200, 200), (200, 300), (100, 300)],
            "Checkout": [(800, 200), (900, 200), (900, 300), (800, 300)],
        },
        "enable_funnel": True,
        "funnel_stages": ("Entrance", "Checkout"),
        "enable_avg_dwell": True,
        "detect_mode": "all_frames",
        "source": "solutions_ci_demo.mp4",
    }

    # Initialize DwellTimeAnalyzer
    dwell_time_analyzer = solutions.DwellTimeAnalyzer(**dwell_config)

    frame_count = 0  # Required for analytics

    # Process frames from the major solutions demo video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        original_im0 = im0.copy()

        # Process each solution
        _ = counter.count(original_im0.copy())
        _ = heatmap.generate_heatmap(original_im0.copy())
        _ = speed.estimate_speed(original_im0.copy())
        _ = queue.process_queue(original_im0.copy())
        _ = line_analytics.process_data(original_im0.copy(), frame_count)
        _ = pie_analytics.process_data(original_im0.copy(), frame_count)
        _ = bar_analytics.process_data(original_im0.copy(), frame_count)
        _ = area_analytics.process_data(original_im0.copy(), frame_count)

        frame_count += 1

    cap.release()

    # Reset frame count for DwellTimeAnalyzer
    frame_count = 0

    # Process frames from the DwellTimeAnalyzer demo video
    while dwell_cap.isOpened():
        success, im0 = dwell_cap.read()
        if not success:
            break

        # Process the frame with DwellTimeAnalyzer
        dwell_time_analyzer.count(im0.copy())

        # (Optional) Here you can add assertions or checks on `annotated_frame`
        # For example, verify that certain analytics are present
        # However, since the output is visual, automated checks might be limited

        frame_count += 1

    dwell_cap.release()


@pytest.mark.slow
def test_instance_segmentation():
    """
    Test the instance segmentation solution.

    This test initializes an instance segmentation model, processes each frame of a demo video, applies segmentation
    masks, and ensures that the segmentation operates without errors.
    """
    from ultralytics.utils.plotting import Annotator, colors

    # Initialize the instance segmentation model
    model = YOLO("yolo11n-seg.pt")
    names = model.names

    # Download the demo video if not already present
    safe_download(url=MAJOR_SOLUTIONS_DEMO)
    cap = cv2.VideoCapture("solutions_ci_demo.mp4")
    assert cap.isOpened(), "Error reading solutions_ci_demo.mp4 video file"

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                color = colors(int(cls), True)
                annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)])

        im0[:] = annotator.result()

    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_streamlit_predict():
    """
    Test the Streamlit live inference solution.

    This test initializes the Streamlit inference solution and ensures that it can start without errors.
    Note: This test might require a display environment to run Streamlit successfully.
    """
    solutions.inference()
