# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import pytest
from ultralytics import YOLO, solutions


def download_sample_videos():
    from ultralytics.utils.downloads import safe_download
    safe_download(url='https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4')
    safe_download(url='https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_pose_demo.mp4')


@pytest.mark.slow
def test_major_solutions():
    """Test the object counting, heatmap, speed estimation and queue management solution."""
    model = YOLO('yolov8n.pt')
    names = model.names
    cap = cv2.VideoCapture('solutions_ci_demo.mp4')
    assert cap.isOpened(), 'Error reading video file'
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

    counter = solutions.ObjectCounter(reg_pts=region_points, classes_names=names, view_img=False)
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, classes_names=names, view_img=False)
    speed = solutions.SpeedEstimator(reg_pts=region_points, names=names, view_img=False)
    queue = solutions.QueueManager(classes_names=names, reg_pts=region_points, view_img=False)

    temp_im0 = None
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        temp_im0 = im0
        tracks = model.track(im0, persist=True, show=False)
        im0 = counter.start_counting(im0, tracks)
        im0 = temp_im0
        im0 = heatmap.generate_heatmap(im0, tracks)
        im0 = temp_im0
        im0 = speed.estimate_speed(im0, tracks)
        im0 = temp_im0
        im0 = queue.process_queue(im0, tracks)
        del im0
    cap.release()
    cv2.destroyAllWindows()

