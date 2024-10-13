# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import pytest

from ultralytics import YOLO, solutions
from ultralytics.utils.downloads import safe_download

MAJOR_SOLUTIONS_DEMO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
WORKOUTS_SOLUTION_DEMO = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solution_ci_pose_demo.mp4"


@pytest.mark.slow
def test_major_solutions():
    """Test the object counting, heatmap, speed estimation and queue management solution."""
    safe_download(url=MAJOR_SOLUTIONS_DEMO)
    cap = cv2.VideoCapture("solutions_ci_demo.mp4")
    assert cap.isOpened(), "Error reading video file"
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(region=region_points, model="yolo11n.pt", show=False)
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, model="yolo11n.pt", show=False)
    speed = solutions.SpeedEstimator(region=region_points, model="yolo11n.pt", show=False)
    queue = solutions.QueueManager(region=region_points, model="yolo11n.pt", show=False)
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        original_im0 = im0.copy()
        _ = counter.count(original_im0.copy())
        _ = heatmap.generate_heatmap(original_im0.copy())
        _ = speed.estimate_speed(original_im0.copy())
        _ = queue.process_queue(original_im0.copy())
    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_aigym():
    """Test the workouts monitoring solution."""
    safe_download(url=WORKOUTS_SOLUTION_DEMO)
    cap = cv2.VideoCapture("solution_ci_pose_demo.mp4")
    assert cap.isOpened(), "Error reading video file"
    gym = solutions.AIGym(line_width=2, kpts=[5, 11, 13])
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        _ = gym.monitor(im0)
    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_instance_segmentation():
    """Test the instance segmentation solution."""
    from ultralytics.utils.plotting import Annotator, colors

    model = YOLO("yolo11n-seg.pt")
    names = model.names
    cap = cv2.VideoCapture("solutions_ci_demo.mp4")
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
    solutions.inference()
