from ultralytics.tracker import BYTETracker, BOTSORT
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils import ROOT
from omegaconf import OmegaConf
import torch

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
