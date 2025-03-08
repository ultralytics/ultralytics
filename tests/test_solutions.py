# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import pytest

from tests import TMP
from ultralytics import solutions
from ultralytics.utils import ASSETS_URL
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
    heatmapcounter = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False, region=region_points
    )  # Test heatmaps with object counting
    speed = solutions.SpeedEstimator(region=region_points, model="yolo11n.pt", show=False)  # Test queue manager
    queue = solutions.QueueManager(region=region_points, model="yolo11n.pt", show=False)  # Test speed estimation
    lineanalytics = solutions.Analytics(analytics_type="line", model="yolo11n.pt", show=False)  # line analytics
    pieanalytics = solutions.Analytics(analytics_type="pie", model="yolo11n.pt", show=False)  # line analytics
    baranalytics = solutions.Analytics(analytics_type="bar", model="yolo11n.pt", show=False)  # line analytics
    areaanalytics = solutions.Analytics(analytics_type="area", model="yolo11n.pt", show=False)  # line analytics
    trackzone = solutions.TrackZone(region=region_points, model="yolo11n.pt", show=False)  # trackzone
    objectcropper = solutions.ObjectCropper(model="yolo11n.pt", show=False)  # object cropping
    objectblurrer = solutions.ObjectBlurrer(blur_ratio=0.5, model="yolo11n.pt", show=False)  # Object blurring
    isegment = solutions.InstanceSegmentation(model="yolo11n-seg.pt", show=False)  # Instance segmentation
    visioneye = solutions.VisionEye(model="yolo11n.pt", show=False)  # Visioneye
    regioncounter = solutions.RegionCounter(region=region_points, model="yolo11n.pt", show=False)  # Region counter
    frame_count = 0  # Required for analytics
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        frame_count += 1
        original_im0 = im0.copy()
        _ = counter(original_im0.copy())
        _ = heatmap(original_im0.copy())
        _ = heatmapcounter(original_im0.copy())
        _ = speed(original_im0.copy())
        _ = queue(original_im0.copy())
        _ = lineanalytics(original_im0.copy(), frame_count)
        _ = pieanalytics(original_im0.copy(), frame_count)
        _ = baranalytics(original_im0.copy(), frame_count)
        _ = areaanalytics(original_im0.copy(), frame_count)
        _ = trackzone(original_im0.copy())
        _ = objectcropper(original_im0.copy())
        _ = isegment(original_im0.copy())
        _ = objectblurrer(original_im0.copy())
        _ = visioneye(original_im0.copy())
        _ = regioncounter(original_im0.copy())
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
        _ = gym(im0)
    cap.release()


@pytest.mark.slow
def test_streamlit_predict():
    """Test streamlit predict live inference solution."""
    solutions.Inference().inference()
