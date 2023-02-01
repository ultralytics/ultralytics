from ultralytics.yolo.engine.model import YOLO
from ultralytics.tracker import BYTETracker
from omegaconf import OmegaConf

cfg = OmegaConf.load("/home/laughing/codes/ultralytics-lq/ultralytics/tracker/cfg/bytetrack.yaml")


def on_predict_start(predictor):
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = BYTETracker(args=cfg, frame_rate=cfg.frame_rate)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_batch_end(predictor):
    # results -> List[batch_size]
    path, im, im0s, vid_cap, s = predictor.batch
    for i in range(predictor.dataset.bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        track_result = predictor.trackers[i].update(det)


def test_callback():
    model = YOLO("weights/yolov8n.pt")
    model.add_callback("on_predict_start", on_predict_start)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)
    results = model.predict(source="/home/laughing/Videos/test.mp4", show=False, stream=True)
    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bbox outputs
        # masks = result.masks  # Masks object for segmenation masks outputs


test_callback()
