from ultralytics.yolo.engine.model import YOLO
from ultralytics.tracker import BYTETracker
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils import ROOT
from omegaconf import OmegaConf
import cv2


def on_predict_start(predictor):
    trackers = []
    cfg = OmegaConf.load(ROOT / "tracker/cfg/bytetrack.yaml")
    for _ in range(predictor.dataset.bs):
        tracker = BYTETracker(args=cfg, frame_rate=cfg.frame_rate)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_batch_end(predictor):
    bs = predictor.dataset.bs 
    track_results = [None] * bs
    im0s = predictor.batch[2]
    im0s = im0s if bs > 1 else [im0s]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        track_results[i] = predictor.trackers[i].update(det)
    predictor.results = zip(predictor.results, track_results, im0s)


def track():
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
    track()

