# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.cfg import TASKS
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .deep_oc_sort import DeepOCSORT
from .fast_tracker import FASTTracker
from .oc_sort import OCSORT
from .track_tracker import TRACKTRACK

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "tracktrack": TRACKTRACK,
    "fasttrack": FASTTracker,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
}


def on_predict_start(predictor: object) -> None:
    """Initialize trackers for object tracking during prediction.

    Existing trackers are reused only when ``predictor.args.persist`` is True, which ``Model.track()`` refreshes on
    every call so that a later ``persist=False`` resets the trackers.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor)
    """
    trackable = ("detect", "segment", "pose", "obb")  # tasks whose results carry boxes, in canonical order
    if (task := predictor.args.task) in TASKS and task not in trackable:  # unknown third-party tasks are left alone
        raise ValueError(f"❌ Task '{task}' doesn't support 'mode=track', valid tasks are {', '.join(trackable)}")

    if hasattr(predictor, "trackers") and getattr(predictor.args, "persist", False):
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))
    cfg.device = predictor.device  # run any ReID encoder on the predictor's device

    if cfg.tracker_type not in TRACKER_MAP:
        raise AssertionError(f"Only {sorted(TRACKER_MAP)} are supported for now, but got '{cfg.tracker_type}'")

    predictor.args.persist = False  # trackers are fresh, so let new videos reset them until persist is set again
    predictor._feats = None  # reset ReID pre-hook state
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if hasattr(predictor, "_orig_postprocess"):  # restore any raw-preds wrapper left by a prior TRACKTRACK run
        predictor.postprocess = predictor._orig_postprocess
        del predictor._orig_postprocess
    if cfg.tracker_type in {"botsort", "tracktrack", "deepocsort"} and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo26n-cls.pt"
        else:
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # non-stream modes reuse a single tracker
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # used to reset the tracker when switching videos

    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if hasattr(tracker_cls, "setup_predictor"):
        tracker_cls.setup_predictor(predictor)


def on_predict_postprocess_end(predictor: object) -> None:
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"

    tracker_cls = type(predictor.trackers[0])
    dets_del_list = (
        tracker_cls.compute_frame_extras(predictor) if hasattr(tracker_cls, "compute_frame_extras") else None
    )

    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not predictor.args.persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (src := result.obb if is_obb else result.boxes).cpu().numpy()
        kwargs = {"feats": getattr(result, "feats", None)}
        if dets_del_list is not None:
            kwargs["dets_del"] = dets_del_list[i]
        tracks = tracker.update(det, result.orig_img, **kwargs)
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1], device=src.data.device)}
        predictor.results[i].update(**update_args)


def register_tracker(model: object) -> None:
    """Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model)
    """
    model.add_callback("on_predict_start", on_predict_start)
    model.add_callback("on_predict_postprocess_end", on_predict_postprocess_end)
