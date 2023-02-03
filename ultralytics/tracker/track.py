from ultralytics.yolo.engine.model import YOLO
from ultralytics.tracker import BYTETracker, BOTSORT
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils import ROOT
from omegaconf import OmegaConf
import torch
import cv2

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
check_requirements('lap')  # for linear_assignment

def on_predict_start(predictor):
    tracker_type = getattr(predictor, "tracker_type", "botsort")
    assert tracker_type in ["bytetrack", "botsort"]
    trackers = []
    cfg = OmegaConf.load(ROOT / f"tracker/cfg/{tracker_type}.yaml")
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_batch_end(predictor):
    bs = predictor.dataset.bs 
    im0s = predictor.batch[2]
    im0s = im0s if isinstance(im0s, list) else [im0s]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks = predictor.trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))
        if predictor.results[i].masks is not None:
            idx = tracks[:, -1].tolist()
            predictor.results[i].masks = predictor.results[i].masks[idx]

def register_tracker(model):
    model.add_callback("on_predict_start", on_predict_start)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

def test_track():
    model = YOLO("weights/yolov8l.pt")
    model.add_callback("on_predict_start", on_predict_start)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)
    results = model.predict(source="/home/laughing/Videos/test.mp4", show=False, stream=True, half=True)
    for _, track_result, im0 in results:
        annotator = Annotator(im0, line_width=2)
        if track_result is not None:
            for track in track_result:
                xyxy = track.tlbr
                id = track.track_id
                c = int(track.cls)
                label = f'id:{id} {model.names[c]}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        cv2.imshow('p', annotator.result())
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    test_track()

