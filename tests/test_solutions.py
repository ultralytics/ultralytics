# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import pytest

from ultralytics import solutions
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
    line_points = [(20, 400), (1080, 400)]
    polygon_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]
    queue_region = [(20, 340), (1080, 340), (1080, 480), (20, 480)]
    counter_reg = solutions.ObjectCounter(reg_pts=region_points, show=True, model="yolov8n.pt")
    counter_line = solutions.ObjectCounter(reg_pts=line_points, show=True, model="yolov8n.pt")
    counter_polygon = solutions.ObjectCounter(reg_pts=polygon_points, show=True, model="yolov8n.pt")
    counter_cls = solutions.ObjectCounter(reg_pts=line_points, show=True, model="yolov8n.pt", classes=0)
    queue = solutions.QueueManager(reg_pts=queue_region, show=True, model="yolov8n.pt")
    queue_cls = solutions.QueueManager(reg_pts=queue_region, show=True, model="yolov8n.pt", classes=0)
    speed = solutions.SpeedEstimator(reg_pts=line_points, show=True, model="yolov8n.pt")
    solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolov8n.pt", reg_pts=None)
    heatmap_line = solutions.Heatmap(reg_pts=line_points, colormap=cv2.COLORMAP_PARULA, show=True, model="yolov8n.pt")
    heatmap_polygon = solutions.Heatmap(
        reg_pts=polygon_points, colormap=cv2.COLORMAP_PARULA, show=True, model="yolov8n.pt"
    )
    heatmap_region = solutions.Heatmap(
        reg_pts=region_points, colormap=cv2.COLORMAP_PARULA, show=True, model="yolov8n.pt"
    )
    heatmap_cls = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolov8n.pt", classes=[0, 2])

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        original_im0 = im0.copy()
        _ = counter_reg.start_counting(original_im0.copy())
        _ = counter_line.start_counting(original_im0.copy())
        _ = counter_polygon.start_counting(original_im0.copy())
        _ = counter_cls.start_counting(original_im0.copy())
        _ = speed.estimate_speed(original_im0.copy())
        _ = queue.process_queue(original_im0.copy())
        _ = queue_cls.process_queue(original_im0.copy())
        _ = heatmap_line.generate_heatmap(original_im0.copy())
        _ = heatmap_polygon.generate_heatmap(original_im0.copy())
        _ = heatmap_region.generate_heatmap(original_im0.copy())
        _ = heatmap_cls.generate_heatmap(original_im0.copy())
    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_aigym():
    """Test the workouts monitoring solution."""
    safe_download(url=WORKOUTS_SOLUTION_DEMO)
    cap = cv2.VideoCapture("solution_ci_pose_demo.mp4")
    assert cap.isOpened(), "Error reading video file"
    gym_object = solutions.AIGym(pose_type="squat", show=True, kpts_to_check=[5, 11, 13], model="yolov8n-pose.pt")
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        _ = gym_object.start_counting(im0)
    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_instance_segmentation():
    """Test the instance segmentation solution."""
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors

    model = YOLO("yolov8n-seg.pt")
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
def test_analytics():
    """Test ultralytics analytics."""
    safe_download(url=MAJOR_SOLUTIONS_DEMO)
    import cv2
    from ultralytics import YOLO, solutions
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("solutions_ci_demo.mp4")
    assert cap.isOpened(), "Error reading video file"

    analytics_line = solutions.Analytics(type="line", show=True)
    analytics_multiple_lines = solutions.Analytics(type="line", show=True)
    analytics_pie = solutions.Analytics(type="pie",show=True)
    analytics_bar = solutions.Analytics(type="bar", show=True)
    analytics_area = solutions.Analytics(type="area", show=True)

    total_counts, frame_count = 0, 0
    data, labels, clswise_count = {}, [], {}

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1
            results = model.track(frame, persist=True, verbose=True)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    if model.names[int(cls)] not in labels:
                        labels.append(model.names[int(cls)])
                    if model.names[int(cls)] in data:
                        data[model.names[int(cls)]] += 1

                    if model.names[int(cls)] in clswise_count:
                        clswise_count[model.names[int(cls)]] += 1
                    else:
                        clswise_count[model.names[int(cls)]] = 1

                    total_counts += 1
            _ = analytics_line.update_line(frame_count, total_counts)
            _ = analytics_multiple_lines.update_line(frame_count, total_counts)
            _ = analytics_pie.update_pie(clswise_count)
            _ = analytics_bar.update_bar(clswise_count)
            _ = analytics_area.update_area(frame_count, clswise_count)
            clswise_count = {}
            data = {}
            total_counts = 0
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


@pytest.mark.slow
def test_streamlit_predict():
    """Test streamlit predict live inference solution."""
    solutions.inference()
