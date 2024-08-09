# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default Deepsort tracker settings available at https://github.com/nwojke/deep_sort

from .deep_sort import DeepSort

__all__ = ["DeepSort", "build_tracker"]


def build_tracker(cfg, use_cuda):
    return DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda,
    )
