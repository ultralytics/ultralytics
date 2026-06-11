# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML, IterableSimpleNamespace
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


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to reuse existing trackers if they are already attached.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if predictor.args.task in {"classify", "reid"}:
        raise ValueError(f"❌ {predictor.args.task} task doesn't support 'mode=track'")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in TRACKER_MAP:
        raise AssertionError(f"Only {sorted(TRACKER_MAP)} are supported for now, but got '{cfg.tracker_type}'")

    predictor._feats = None  # reset ReID pre-hook state
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if hasattr(predictor, "_orig_postprocess"):  # restore any raw-preds wrapper left by a prior TRACKTRACK run
        predictor.postprocess = predictor._orig_postprocess
        del predictor._orig_postprocess
    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if getattr(tracker_cls, "supports_native_reid", False) and cfg.with_reid and cfg.model == "auto":
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
        tracker = tracker_cls(args=cfg)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # non-stream modes reuse a single tracker
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # used to reset the tracker when switching videos

    if hasattr(tracker_cls, "setup_predictor"):
        tracker_cls.setup_predictor(predictor)


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
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
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        kwargs = {"feats": getattr(result, "feats", None)}
        if kwargs["feats"] is None and getattr(predictor, "_hook", None) is not None and len(det):
            if not getattr(predictor, "_feats_warned", False):
                predictor._feats_warned = True
                LOGGER.warning(
                    "Native ReID features were not extracted for this frame; tracker falls back to motion-only "
                    "association. Set the tracker YAML 'model' to a ReID weight (e.g. yolo26n-reid.pt) to use a "
                    "dedicated appearance encoder."
                )
        if dets_del_list is not None:
            kwargs["dets_del"] = dets_del_list[i]
        tracks = tracker.update(det, result.orig_img, **kwargs)
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
