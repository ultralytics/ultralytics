# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import pytest

from tests import TMP
from ultralytics import YOLO, solutions
from ultralytics.utils import ASSETS_URL, WEIGHTS_DIR
from ultralytics.utils.downloads import safe_download

DEMO_VIDEO = "solutions_ci_demo.mp4"
POSE_VIDEO = "solution_ci_pose_demo.mp4"


@pytest.mark.slow
def test_major_solutions():
    """Test the object counting, heatmap, speed estimation, trackzone and queue management solution."""
    safe_download(url=f"{ASSETS_URL}/{DEMO_VIDEO}", dir=TMP)
    cap = cv2.VideoCapture(str(TMP / DEMO_VIDEO))
    assert cap.isOpened(), "Error reading video file"
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(region=region_points, model="yolo11n.pt", show=False)  # Test object counter
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False)  # Test heatmaps
    heatmap_count = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False, region=region_points
    )  # Test heatmaps with object counting
    speed = solutions.SpeedEstimator(region=region_points, model="yolo11n.pt", show=False)  # Test queue manager
    queue = solutions.QueueManager(region=region_points, model="yolo11n.pt", show=False)  # Test speed estimation
    line_analytics = solutions.Analytics(analytics_type="line", model="yolo11n.pt", show=False)  # line analytics
    pie_analytics = solutions.Analytics(analytics_type="pie", model="yolo11n.pt", show=False)  # line analytics
    bar_analytics = solutions.Analytics(analytics_type="bar", model="yolo11n.pt", show=False)  # line analytics
    area_analytics = solutions.Analytics(analytics_type="area", model="yolo11n.pt", show=False)  # line analytics
    trackzone = solutions.TrackZone(region=region_points, model="yolo11n.pt", show=False)  # Test trackzone
    frame_count = 0  # Required for analytics
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        frame_count += 1
        original_im0 = im0.copy()
        _ = counter.count(original_im0.copy())
        _ = heatmap.generate_heatmap(original_im0.copy())
        _ = heatmap_count.generate_heatmap(original_im0.copy())
        _ = speed.estimate_speed(original_im0.copy())
        _ = queue.process_queue(original_im0.copy())
        _ = line_analytics.process_data(original_im0.copy(), frame_count)
        _ = pie_analytics.process_data(original_im0.copy(), frame_count)
        _ = bar_analytics.process_data(original_im0.copy(), frame_count)
        _ = area_analytics.process_data(original_im0.copy(), frame_count)
        _ = trackzone.trackzone(original_im0.copy())
    cap.release()

    # Test workouts monitoring
    safe_download(url=f"{ASSETS_URL}/{POSE_VIDEO}", dir=TMP)
    cap = cv2.VideoCapture(str(TMP / POSE_VIDEO))
    assert cap.isOpened(), "Error reading video file"
    gym = solutions.AIGym(kpts=[5, 11, 13], show=False)
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        _ = gym.monitor(im0)
    cap.release()


@pytest.mark.slow
def test_instance_segmentation():
    """Test the instance segmentation solution."""
    from ultralytics.utils.plotting import Annotator, colors

    model = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
    names = model.names
    cap = cv2.VideoCapture(TMP / DEMO_VIDEO)
    assert cap.isOpened(), "Error reading video file"
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
    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_streamlit_predict():
    """Test streamlit predict live inference solution."""
    solutions.Inference().inference()
