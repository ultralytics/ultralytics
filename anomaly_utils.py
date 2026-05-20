"""Utility functions for YOLOAnomaly / AnomalyDINO training-free anomaly detection.

Shared helpers for MVTec data loading, memory-bank model construction, heatmap
visualization, and metrics export.  Import from scripts as::

    from anomaly_utils import get_mvtec_yolo_data, build_ad_model
"""

import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np




# Default fused-heatmap bank config (P3 raw backbone features).
FUSED_HEATMAP_ARGS: dict[str, Any] = dict(
    feature_mode="fused_heatmap",
    return_heatmap=True,
    active_layers=[],
    fused_use_pre_clshead=True,
    fused_layers=[0],
)






def save_heatmap_overlay(
    test_img: str,
    heatmap: Any,
    save_path: str,
    alpha: float = 0.45,
) -> bool:
    """Overlay a normalised heatmap on *test_img* and save to *save_path*.

    Args:
        test_img: Path to the original BGR image.
        heatmap: 2-D or 3-D (1×H×W) float tensor or array in [0, 1].
        save_path: Output path (parent dirs are created automatically).
        alpha: Heatmap blend weight; 0 = original image only.

    Returns:
        True on success, False if the image or heatmap could not be read.
    """
    if heatmap is None:
        return False
    hm = heatmap.detach().cpu().float().numpy() if hasattr(heatmap, "detach") else np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm.squeeze()
    if hm.ndim != 2:
        return False
    img = cv2.imread(test_img, cv2.IMREAD_COLOR)
    if img is None:
        return False
    hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
    hm = np.clip(hm, 0.0, 1.0)
    hm_u8 = (hm * 255.0).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1.0 - alpha, hm_color, alpha, 0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    return True


def save_metrics_to_csv(metrics: dict, save_path: str = "./runs/temp/metrics.csv") -> None:
    """Append *metrics* dict as one row to a CSV file, creating it if needed.

    List values are serialised as ``0|1|2`` to avoid comma-inside-cell issues
    when the file is opened in Excel.

    Args:
        metrics: Flat dict of metric name → value.
        save_path: Destination CSV path (parent dirs are created automatically).
    """
    def _fmt(v: Any) -> Any:
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, (list, tuple)):
            return "|".join(str(x) for x in v)
        return v

    row = {k: _fmt(v) for k, v in metrics.items()}
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    file_exists = os.path.isfile(save_path)
    # utf-8-sig writes BOM on new files so Excel on macOS opens them correctly;
    # append mode uses plain utf-8 (BOM must only appear at the file start).
    encoding = "utf-8" if file_exists else "utf-8"
    with open(save_path, mode="a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved metrics to {save_path}")


BUILD_ARGS_KEYS= [ "mode", # "anomaly" or "heatmap"
                  "accumulate_thresh", # float in [0,1], lower = more support features, higher = more selective
                    "score_filter_kernel", # int, odd, >=1, size of maxpool kernel for support feature filtering (default 1 = no filtering)
                    "calibration_interval", # int, number of support images between memory bank calibrations (default 3)
                    "auto_temperature", # bool, whether to automatically set the EM temperature based on the support feature distribution (default True)
                    "active_layers", # list of ints, which FPN layers to use for anomaly scoring (default [1,2] = P4 and P5)
                    "fused_use_pre_clshead", # bool, whether the fused heatmap head should use the pre-CLS-HEAD features (True) or raw backbone features (False, default)
                    "fused_layers", # list of ints, which FPN layers to use for the fused heatmap memory bank (default [0] = P3 only)

    ]
DEFAULT_BUILD_ARGS=dict(
    model="anomaly",
    score_filter_kernel=3, # 
    auto_temperature=True, # 
    calibration_interval=3, # set 3 get imporvement
    accumulate_thresh=0.3, # float in [0,1], lower = more support features, higher = more selective
    K=15,# number of nearest neighbors to use in anoamly scoring calculation
    em_iters=1,
    

    )


def get_arguments(category: str = "screw") -> tuple[dict, dict]:
    """Return default (model_arg, anomaly_arg) inference dicts.

    Args:
        category: MVTec category name (unused currently, reserved for future
            per-category tuning).

    Returns:
        Tuple of (model_arg, anomaly_arg) dicts.
    """
    model_arg = dict(conf=0.1, iou=0.25, max_det=1000, imgsz=640, single_cls=True, rect=False)
    anomaly_arg = dict(
        
        mode="anomaly",
        score_filter_kernel=1,
        active_layers=[1, 2],

        # for bank building and calibration
        auto_temperature=True,
        em_iters=5,
        accumulate_thresh=0.3,
        calibration_interval=3,
        calibration_target_score=0.2,
        
        # ------- for infer 
        ad_conf=0.4,
        ad_max_det=10,
        # ------------ for heatmpa 
        return_heatmap=True,
        feature_mode="fused_heatmap", 
		fused_use_pre_clshead=True, 
        fused_layers=[0]
    )
    return model_arg, anomaly_arg


def build_ad_model(
    base_weight: str,
    data_config: dict,
    model_arg: dict,
    anomaly_arg: dict,
    category: str = "screw",
    replace_model: bool = False,
):
    """Build or load a cached YOLOAnomaly model with a frozen memory bank.

    Both per-level (P4/P5) and fused (P3) banks are always built in a single
    ``load_support_set()`` call — no separate fused-bank flag needed.

    The model is saved to ``./runs/temp/{category}_{base_weight_stem}_anomaly_model.pt``
    and reloaded from there on subsequent calls (unless *replace_model* is True).

    Args:
        base_weight: Path to a YOLO / YOLOE base checkpoint.
        data_config: Output of :func:`get_mvtec_yolo_data`.
        model_arg: Inference kwargs (imgsz, conf, …).
        anomaly_arg: Anomaly head kwargs passed to ``set_anomaly_args``.
        category: Category name used in the cached model filename.
        replace_model: Force rebuild even if a cached model exists.

    Returns:
        Configured :class:`~ultralytics.models.yolo.model.YOLOAnomaly` instance.
    """
    from ultralytics.models.yolo.model import YOLOAnomaly  # avoid circular import

    base_weight_name = Path(base_weight).stem
    saved_model_path = f"./runs/temp/{category}_{base_weight_name}_anomaly_model.pt"

    if not os.path.exists(saved_model_path) or replace_model:
        os.makedirs("./runs/temp", exist_ok=True)
        model = YOLOAnomaly(base_weight)
        model.setup(names=["anomaly"])
        model.set_anomaly_args(**anomaly_arg)
        support_images = collect_images(data_config["train_im_dir"])
        model.load_support_set(support_images, imgsz=model_arg["imgsz"])
        model.save(saved_model_path)
        print(f"Saved anomaly model to: {saved_model_path}")
    else:
        model = YOLOAnomaly(saved_model_path)
        print(f"Loaded anomaly model from {saved_model_path}, is_configured={model.is_configured}")
    return model


def eval_anomaly_category(
    model,
    data: dict,
    build_args: dict,
    model_arg: dict,
) -> dict:
    """Run anomaly + heatmap val on one category and return a metrics summary dict.

    Performs three val passes:
    1. Anomaly mode (per-level bbox → mAP + fused heatmap → AUROC).
    2. Heatmap mode with configured ``ad_conf`` (connected-component bbox → mAP + AUROC).
    3. Heatmap mode with pixel-optimal F1 threshold (TEMP diagnostic).

    Args:
        model: Configured YOLOAnomaly with frozen memory bank.
        data: Output of :func:`get_mvtec_yolo_data`.
        build_args: Dict that was passed to ``set_anomaly_args`` at build time
            (used to restore ``ad_conf`` for the heatmap pass).
        model_arg: Inference kwargs (imgsz, conf, …).

    Returns:
        Dict with keys: map10/25/50, image_auroc, pixel_auroc (anomaly mode),
        map10/25/50_hm, image_auroc_hm, pixel_auroc_hm (heatmap mode),
        opt_thresh, map10/25/50_opt (opt-F1 heatmap pass).
    """

    yaml_path = data["data_yaml"]
    val_kw = dict(plots=False, batch=1, **model_arg,)

    def _get(m, attr):
        return getattr(m, attr, float("nan"))

    # ── 1. anomaly mode ───────────────────────────────────────────────────
    model.set_anomaly_args(feature_mode="per_level", active_layers=[1, 2], mode="anomaly")
    model.val(data=yaml_path, **val_kw)
    ma = model.metrics
    results = dict(
        map10=_get(ma, "map10"), map25=_get(ma, "map25"), map50=_get(ma, "map50"),
        image_auroc=_get(ma, "image_auroc"), pixel_auroc=_get(ma, "pixel_auroc"),
    )

    # ── 2. heatmap mode (configured ad_conf) ─────────────────────────────
    model.set_anomaly_args(feature_mode="fused_heatmap", active_layers=[])
    model.val(data=yaml_path, **val_kw)
    mh = model.metrics
    results.update(dict(
        map10_hm=_get(mh, "map10"), map25_hm=_get(mh, "map25"), map50_hm=_get(mh, "map50"),
        image_auroc_hm=_get(mh, "image_auroc"), pixel_auroc_hm=_get(mh, "pixel_auroc"),
    ))

    # ── TEMP: heatmap mode with opt-F1 threshold ──────────────────────────
    opt_thresh = _get(mh, "opt_f1_threshold")
    results["opt_thresh"] = opt_thresh
    if opt_thresh == opt_thresh:  # not nan
        model.set_anomaly_args(feature_mode="fused_heatmap", active_layers=[], ad_conf=opt_thresh)
        model.val(data=yaml_path, **val_kw)
        mo = model.metrics
        results.update(dict(
            map10_opt=_get(mo, "map10"), map25_opt=_get(mo, "map25"), map50_opt=_get(mo, "map50"),
        ))
    else:
        results.update(dict(map10_opt=float("nan"), map25_opt=float("nan"), map50_opt=float("nan")))
    # ─────────────────────────────────────────────────────────────────────

    return results


def print_eval_summary(category: str, r: dict) -> None:
    """Print a formatted summary of :func:`eval_anomaly_category` results."""
    w = 70
    print("\n" + "=" * w)
    print(f"SUMMARY  [{category}]")
    print("=" * w)
    print(f"  anomaly           mAP10 : {r['map10']:.4f}")
    print(f"  anomaly           mAP25 : {r['map25']:.4f}")
    print(f"  anomaly           mAP50 : {r['map50']:.4f}")
    print(f"  anomaly      image_auroc: {r['image_auroc']:.4f}")
    print(f"  anomaly      pixel_auroc: {r['pixel_auroc']:.4f}")
    print(f"  heatmap           mAP10 : {r['map10_hm']:.4f}")
    print(f"  heatmap           mAP25 : {r['map25_hm']:.4f}")
    print(f"  heatmap           mAP50 : {r['map50_hm']:.4f}")
    print(f"  heatmap      image_auroc: {r['image_auroc_hm']:.4f}")
    print(f"  heatmap      pixel_auroc: {r['pixel_auroc_hm']:.4f}")
    # ── TEMP ──────────────────────────────────────────────────────────────
    t = r["opt_thresh"]
    print(f"  heatmap(opt-F1 {t:.4f})  mAP10 : {r['map10_opt']:.4f}")
    print(f"  heatmap(opt-F1 {t:.4f})  mAP25 : {r['map25_opt']:.4f}")
    print(f"  heatmap(opt-F1 {t:.4f})  mAP50 : {r['map50_opt']:.4f}")
    # ─────────────────────────────────────────────────────────────────────


def _draw_boxes(img_bgr: np.ndarray, boxes_obj, title: str) -> np.ndarray:
    """Draw bbox + score on a BGR image and stamp a title in the top-left."""
    out = img_bgr.copy()
    n = len(boxes_obj) if boxes_obj is not None else 0
    if n > 0:
        xyxy = boxes_obj.xyxy.cpu().numpy().astype(int)
        scores = boxes_obj.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, scores):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f"{score:.2f}", (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, f"{title}  n={n}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out


def _heatmap_overlay(img_bgr: np.ndarray, heatmap, alpha: float = 0.45,
                     title: str = "heatmap") -> np.ndarray:
    """Return a BGR overlay of `heatmap` on `img_bgr` with a stamped title."""
    if heatmap is None:
        out = img_bgr.copy()
        cv2.putText(out, f"{title}  (none)", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return out
    hm = heatmap.detach().cpu().float().numpy() if hasattr(heatmap, "detach") \
        else np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm.squeeze()
    hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
    hm = np.clip(hm, 0.0, 1.0)
    hm_u8 = (hm * 255.0).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (img_bgr.shape[1], img_bgr.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1.0 - alpha, hm_color, alpha, 0)
    cv2.putText(out, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out


def visualize_heatmap_predict(
    model,
    data: dict,
    category: str,
    model_arg: dict,
    out_dir: str = "runs/viz_heatmap",
    max_imgs: int = 10,
) -> None:
    """Save a 3-panel comparison per anomalous test image.

    Panels (left → right):
        1. anomaly-mode bbox detection (per-level scoring head).
        2. heatmap-mode overlay (fused_heatmap).
        3. heatmap-mode bbox (boxes derived from the heatmap).

    Args:
        model: YOLOAnomaly with frozen bank.  Mutated between modes during the call;
            the final mode left set is ``fused_heatmap``.
        data: Output of :func:`get_mvtec_yolo_data`.
        category: Category name (used for the output subdirectory).
        model_arg: Inference kwargs (imgsz, conf, iou, …).
        out_dir: Root directory for output images.
        max_imgs: Maximum number of anomalous images to process.
    """
    anomaly_imgs = data["test_anomaly_im_list"][:max_imgs]
    out_path = Path(out_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    pred_kw = dict(imgsz=model_arg["imgsz"], conf=model_arg["conf"],
                   iou=model_arg["iou"], verbose=False)

    print(f"\n[{category}] 3-panel viz → {out_path}  ({len(anomaly_imgs)} images)")
    for img_path in anomaly_imgs:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  skip (unreadable): {img_path}")
            continue
        stem   = Path(img_path).stem
        parent = Path(img_path).parent.name

        # ── Panel 1: anomaly-mode bbox ──────────────────────────────────────
        model.set_anomaly_args(feature_mode="per_level", active_layers=[1, 2])
        res_ad = model.predict(img_path, **pred_kw)[0]
        panel_ad = _draw_boxes(img_bgr, res_ad.boxes, "anomaly bbox")

        # ── Panels 2 & 3: heatmap-mode overlay + heatmap-derived bbox ───────
        model.set_anomaly_args(feature_mode="fused_heatmap", active_layers=[])
        res_hm = model.predict(img_path, **pred_kw)[0]
        panel_hm   = _heatmap_overlay(img_bgr, res_hm.heatmap, title="heatmap")
        panel_hmbb = _draw_boxes(img_bgr, res_hm.boxes, "heatmap bbox")

        triptych = np.hstack([panel_ad, panel_hm, panel_hmbb])
        out_file = out_path / f"{parent}_{stem}_triptych.jpg"
        cv2.imwrite(str(out_file), triptych)
        print(f"  {parent}/{stem}: ad={len(res_ad.boxes) if res_ad.boxes is not None else 0}  "
              f"hm_box={len(res_hm.boxes) if res_hm.boxes is not None else 0}")
