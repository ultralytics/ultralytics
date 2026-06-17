from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# =============================================================================
# SELECT DEFAULT BENCHMARK HERE: "m5", "m5_new", "m5_coreml", "m5_onnx_coreml", "xeon", "xeon_new", "t4",
# "t4_deimv2_xl_obj365_analysis", "t4_deim_backbone_map", "t4_yolo26", "rf_compare", "t4_all_rf",
# "yolo27detr_compare", "yolo27detr_compare2", "t4_new",
# "jetson-agx-thor-gpu", "jetson-agx-thor-cpu", "jetson-agx-orin-gpu", "jetson-agx-orin-cpu",
# "jetson-orin-nano-super-gpu", or "jetson-orin-nano-super-cpu"
# =============================================================================
BENCHMARK = "t4"

# Default metric for Y axis.
DEFAULT_METRIC = "ap"
METRIC_LABELS = {
    "ap": "mAP50-95 (COCO)",
    "ap50": "AP50 (COCO)",
    "ap75": "AP75 (COCO)",
    "ap_small": "AP_small (COCO)",
    "ap_medium": "AP_medium (COCO)",
    "ap_large": "AP_large (COCO)",
}
METRIC_TITLE_TOKENS = {
    "ap": "mAP",
    "ap50": "AP50",
    "ap75": "AP75",
    "ap_small": "AP_small",
    "ap_medium": "AP_medium",
    "ap_large": "AP_large",
}


# =============================================================================
# BENCHMARK DATA
# =============================================================================
# Format: (size_label, latency_ms, ap_value[, latency_err_ms])
#         ap_value can be a float (legacy: AP/mAP50-95 only)
#         or a metric dict with keys like:
#         {"ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"}

DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP = [
    # T4 TensorRT v10.11, rtdetr_best_op17_nosim_norope_*_fp32attn_debug_fp16.engine.
    # The 640 point was exported before imgsz was added to artifact names.
    # Labels use decoder layer count; export_eval_idx=eidx means l{eidx + 1}.
    # ("480/l2/q100", 7.7, {"ap": 55.8, "ap50": 73.5, "ap75": 60.7, "ap_small": 35.5, "ap_medium": 61.5, "ap_large": 75.2}, 0.1),
    ("512/l3", 8.8, {"ap": 57.5, "ap50": 75.0, "ap75": 62.4, "ap_small": 37.8, "ap_medium": 63.1, "ap_large": 76.4}, 0.2),
    ("576/l4", 11.1, {"ap": 58.8, "ap50": 76.5, "ap75": 64.0, "ap_small": 40.6, "ap_medium": 64.1, "ap_large": 76.9}, 0.2),
    ("640/l6", 13.7, {"ap": 59.8, "ap50": 77.7, "ap75": 65.3, "ap_small": 43.3, "ap_medium": 64.8, "ap_large": 77.4}, 0.3),
    ("704/l6", 18.1, {"ap": 60.3, "ap50": 77.9, "ap75": 66.0, "ap_small": 43.9, "ap_medium": 65.2, "ap_large": 77.7}, 0.4),
    ("800/l6", 24.1, {"ap": 60.9, "ap50": 78.3, "ap75": 66.5, "ap_small": 45.3, "ap_medium": 65.3, "ap_large": 77.2}, 1.0),
]

DEIM_DINOV3B_OBJ365_IMGSZ_SWEEP = [
    # T4 TensorRT v10.11, deim_dinov3b_yolodecayp07_32epc_obj2coco24_fp32.
    # COCO-val metrics converted from fractions to percentage points.
    # Labels use decoder layer count; export_eval_idx=eidx means l{eidx + 1}.
    ("512/l3", 17.1, {"ap": 59.5, "ap50": 77.2, "ap75": 64.5, "ap_small": 41.1, "ap_medium": 65.3, "ap_large": 77.7}, 0.6),
    ("576/l4", 20.8, {"ap": 60.6, "ap50": 78.4, "ap75": 66.0, "ap_small": 43.2, "ap_medium": 66.0, "ap_large": 78.3}, 0.9),
    ("640/l4", 24.6, {"ap": 61.4, "ap50": 79.0, "ap75": 67.2, "ap_small": 45.0, "ap_medium": 66.5, "ap_large": 78.6}, 0.8),
    ("640/l6", 25.5, {"ap": 61.4, "ap50": 79.1, "ap75": 67.2, "ap_small": 45.3, "ap_medium": 66.5, "ap_large": 78.7}, 0.8),
    ("704/l6", 34.8, {"ap": 61.9, "ap50": 79.5, "ap75": 67.4, "ap_small": 46.0, "ap_medium": 66.9, "ap_large": 78.4}, 1.2),
    ("800/l6", 45.0, {"ap": 62.3, "ap50": 79.9, "ap75": 67.9, "ap_small": 47.2, "ap_medium": 66.8, "ap_large": 78.2}, 1.7),
]

RF_DETR_OBJ365_TOPK_IMGSZ = [
    # Our benchmark measurements, not paper-reported values. Labels include model scale and profiled ONNX image size.
    ("n/384", 2.7, {"ap": 48.4, "ap50": 67.5, "ap75": 51.7, "ap_small": 25.3, "ap_medium": 53.6, "ap_large": 71.0}, 0.0),
    ("s/512", 3.9, {"ap": 53.0, "ap50": 72.0, "ap75": 57.1, "ap_small": 31.8, "ap_medium": 58.4, "ap_large": 73.1}, 0.1),
    ("m/576", 5.0, {"ap": 54.7, "ap50": 73.6, "ap75": 59.1, "ap_small": 35.9, "ap_medium": 59.8, "ap_large": 73.7}, 0.0),
    ("l/704", 7.9, {"ap": 56.5, "ap50": 75.1, "ap75": 61.2, "ap_small": 39.0, "ap_medium": 61.0, "ap_large": 74.0}, 0.1),
    ("x/700", 16.2, {"ap": 58.6, "ap50": 77.5, "ap75": 64.0, "ap_small": 40.8, "ap_medium": 64.3, "ap_large": 76.3}, 0.8),
    ("xxl/880", 25.6, {"ap": 60.1, "ap50": 78.5, "ap75": 65.8, "ap_small": 43.7, "ap_medium": 65.1, "ap_large": 76.3}, 1.3),
]

DFINE_DINOV3SPLUS_OBJ365_IMGSZ640 = [
    # T4 TensorRT v10.11, imgsz640 fp16 engine. The paired ~887 ms values are PT latency.
    ("l4", 14.4, {"ap": 59.5, "ap50": 77.1, "ap75": 65.0, "ap_small": 42.2, "ap_medium": 64.6, "ap_large": 77.1}),
    ("no-fp32", 15.0, {"ap": 59.5, "ap50": 77.3, "ap75": 64.9, "ap_small": 41.8, "ap_medium": 64.5, "ap_large": 77.2}),
    ("fp32", 14.9, {"ap": 59.7, "ap50": 77.4, "ap75": 65.1, "ap_small": 42.2, "ap_medium": 64.7, "ap_large": 77.2}),
    ("imgsz800", 25.3, {"ap": 60.7, "ap50": 78.4, "ap75": 66.4, "ap_small": 44.6, "ap_medium": 65.1, "ap_large": 76.9}),
]

DFINE_DINOV3S_OBJ365_IMGSZ_SWEEP = [
    # T4 TensorRT v10.11, rtdetr_dfine_dinov3s_obj365_op17_nosim_norope_*_fp32attn_fp16.engine.
    # AP metrics are Objects365+COCO COCO-val results, converted from fractions to percentage points.
    # The paired ~416-785 ms values are PT latency; these plots use the TensorRT latency column.
    ("640/l6", 11.2, {"ap": 58.0, "ap50": 75.5, "ap75": 63.3, "ap_small": 40.8, "ap_medium": 63.0, "ap_large": 76.3}, 0.2),
    ("640/l4", 10.7, {"ap": 57.8, "ap50": 75.3, "ap75": 63.1, "ap_small": 40.3, "ap_medium": 62.7, "ap_large": 76.3}, 0.2),
    ("576/l4", 9.4, {"ap": 56.7, "ap50": 74.4, "ap75": 61.7, "ap_small": 38.1, "ap_medium": 62.1, "ap_large": 75.3}, 0.2),
    ("512/l3", 7.3, {"ap": 55.0, "ap50": 72.6, "ap75": 59.5, "ap_small": 35.2, "ap_medium": 60.5, "ap_large": 74.9}, 0.1),
]

DEIM_EUPE_CONVNEXT_COCO_IMGSZ640 = [
    # T4 TensorRT v10.11, rtdetr_deim_eupe_convnext{tiny,small}_*_op17_nosim_norope_imgsz640*_fp32attn_debug_fp16.engine.
    # COCO-val metrics converted from fractions to percentage points.
    ("T/l4", 10.2, {"ap": 56.3, "ap50": 73.4, "ap75": 61.3, "ap_small": 38.3, "ap_medium": 61.0, "ap_large": 73.8}, 0.3),
    ("T/l6", 10.9, {"ap": 56.7, "ap50": 74.1, "ap75": 61.8, "ap_small": 38.7, "ap_medium": 61.4, "ap_large": 74.0}, 0.3),
    ("S/l6", 17.0, {"ap": 59.3, "ap50": 76.5, "ap75": 64.8, "ap_small": 41.6, "ap_medium": 64.6, "ap_large": 77.0}, 0.6),
]

YOLO26L_RTDETR_OBJ365_IMGSZ_SWEEP = [
    # T4 TensorRT v10.11, rtdetr_yolo26*_rtdetr_obj365_op17_nosim_norope_*_nofp32attn_fp16.engine.
    # Labels use decoder layer count; export_eval_idx=eidx means l{eidx + 1}.
    ("n/480/l3e", 1.8, {"ap": 41.1, "ap50": 57.4, "ap75": 44.5, "ap_small": 21.7, "ap_medium": 44.5, "ap_large": 58.8}),
    # ("n/480/l2", 1.8, {"ap": 38.6, "ap50": 54.0, "ap75": 41.6, "ap_small": 19.0, "ap_medium": 42.2, "ap_large": 55.8}, 0.0),
    ("s/512/l3", 2.7, {"ap": 48.4, "ap50": 65.3, "ap75": 52.5, "ap_small": 29.5, "ap_medium": 53.0, "ap_large": 66.2}, 0.0),
    ("m/512/l3", 4.0, {"ap": 51.7, "ap50": 69.0, "ap75": 56.5, "ap_small": 34.3, "ap_medium": 56.7, "ap_large": 68.3}, 0.0),
    # ("n/640/l6", 3.4, {"ap": 46.4, "ap50": 63.3, "ap75": 50.2, "ap_small": 28.9, "ap_medium": 49.8, "ap_large": 63.2}, 0.0),
    # ("s/640/l6", 4.4, {"ap": 51.6, "ap50": 68.9, "ap75": 56.2, "ap_small": 35.0, "ap_medium": 55.8, "ap_large": 67.6}, 0.1),
    # ("l/384/l2", 3.8, {"ap": 47.2, "ap50": 68.3, "ap75": 52.1, "ap_small": 28.5, "ap_medium": 51.6, "ap_large": 65.5}, 0.1),
    # ("l/480/l3", 5.0, {"ap": 54.1, "ap50": 72.0, "ap75": 58.8, "ap_small": 37.6, "ap_medium": 58.9, "ap_large": 71.0}, 0.1),
    ("l/512/l3", 5.2, {"ap": 54.9, "ap50": 72.4, "ap75": 59.9, "ap_small": 37.6, "ap_medium": 59.9, "ap_large": 71.5}, 0.0),
    # ("l/576/l4", 6.7, {"ap": 55.9, "ap50": 73.3, "ap75": 60.8, "ap_small": 38.7, "ap_medium": 60.4, "ap_large": 71.8}, 0.1),
    ("l/640/l4", 7.1, {"ap": 56.7, "ap50": 74.0, "ap75": 62.1, "ap_small": 41.3, "ap_medium": 61.1, "ap_large": 71.4}, 0.1),
]

DEIM_DINOV3B_OBJ365_IMGSZ_SWEEP_YOLO27 = [
    # T4 TensorRT v10.11, deim_dinov3b_yolodecayp07_32epc_obj2coco24_fp32.
    # COCO-val metrics converted from fractions to percentage points.
    # Labels use decoder layer count; export_eval_idx=eidx means l{eidx + 1}.
    ("640/l4", 24.6, {"ap": 61.4, "ap50": 79.0, "ap75": 67.2, "ap_small": 45.0, "ap_medium": 66.5, "ap_large": 78.6}, 0.8),
    # ("640/l6", 25.5, {"ap": 61.4, "ap50": 79.1, "ap75": 67.2, "ap_small": 45.3, "ap_medium": 66.5, "ap_large": 78.7}, 0.8),
]

DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP_YOLO27 = [
    # T4 TensorRT v10.11, rtdetr_best_op17_nosim_norope_*_fp32attn_debug_fp16.engine.
    # The 640 point was exported before imgsz was added to artifact names.
    # Labels use decoder layer count; export_eval_idx=eidx means l{eidx + 1}.
    # ("480/l2/q100", 7.7, {"ap": 55.8, "ap50": 73.5, "ap75": 60.7, "ap_small": 35.5, "ap_medium": 61.5, "ap_large": 75.2}, 0.1),
    # ("512/l3", 8.8, {"ap": 57.5, "ap50": 75.0, "ap75": 62.4, "ap_small": 37.8, "ap_medium": 63.1, "ap_large": 76.4}, 0.2),
    ("576/l4", 11.1, {"ap": 58.8, "ap50": 76.5, "ap75": 64.0, "ap_small": 40.6, "ap_medium": 64.1, "ap_large": 76.9}, 0.2),
    ("640/l6", 13.7, {"ap": 59.8, "ap50": 77.7, "ap75": 65.3, "ap_small": 43.3, "ap_medium": 64.8, "ap_large": 77.4}, 0.3),
    # ("704/l6", 18.1, {"ap": 60.3, "ap50": 77.9, "ap75": 66.0, "ap_small": 43.9, "ap_medium": 65.2, "ap_large": 77.7}, 0.4),
    # ("800/l6", 24.1, {"ap": 60.9, "ap50": 78.3, "ap75": 66.5, "ap_small": 45.3, "ap_medium": 65.3, "ap_large": 77.2}, 1.0),
]

YOLO27DETR_OBJ365_IMGSZ_SWEEP = [
    *[(f"L/{point[0].removeprefix('l/')}", *point[1:]) for point in YOLO26L_RTDETR_OBJ365_IMGSZ_SWEEP],
    *[(f"X/{point[0]}", *point[1:]) for point in DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP_YOLO27],
    *[(f"XXL/{point[0]}", *point[1:]) for point in DEIM_DINOV3B_OBJ365_IMGSZ_SWEEP_YOLO27],
]

BENCHMARKS = {
    "m5": {
        "title": "Object Detection Models: Latency vs mAP (Apple M5 CPU, ONNX)",
        "models": {
            "YOLO26": [
                ("n", 21.0, 40.9),
                ("s", 61.7, 48.6),
                ("m", 168.6, 53.1),
                ("l", 219.1, 55.0),
                ("x", 439.3, 57.5),
            ],
            "RF-DETR": [
                ("n", 79.6, 48.4),
                ("s", 145.5, 53.0),
                ("m", 192.0, 54.7),
                ("l", 307.2, 56.5),
                ("x", 688.3, 58.6),
                ("xxl", 956.2, 60.1),
            ],
            "LW-DETR": [
                ("t", 73.2, 42.9),
                ("s", 110.9, 48.1),
                ("m", 226.7, 52.6),
                ("l", 353.3, 56.1),
                ("x", 765.1, 58.3),
            ],
            "DEIM D-FINE": [
                ("n", 31.4, 43.0),
                ("s", 78.7, 49.0),
                ("m", 158.4, 52.7),
                ("l", 244.5, 54.7),
                ("x", 486.8, 56.5),
            ],
            "DEIM RT-DETRv2": [
                ("r18", 157.8, 49.0),
                ("r34", 233.6, 50.9),
                ("r50m", 254.3, 53.2),
                ("r50", 335.4, 54.3),
                ("r101", 584.6, 55.5),
            ],
            "DEIMv2": [
                ("pico", 22.4, 38.5),
                ("n", 30.5, 43.0),
                ("s", 157.2, 50.9),
                ("m", 240.9, 53.0),
                ("l", 386.2, 56.0),
                ("x", 525.0, 57.8),
            ],
        },
    },
    "m5_new": {
        "title": "Object Detection Models: Latency vs mAP (Apple M5 CPU, ONNX)",
        "models": {
            "YOLO26 (E2E)": [
                ("n", 20.9, 40.1),
                ("s", 61.2, 47.8),
                ("m", 172.0, 52.5),
                ("l", 219.4, 54.4),
                ("x", 439.6, 56.9),
            ],
            "YOLO26 (NMS)": [
                ("n", 21.2, 40.9),
                ("s", 61.8, 48.6),
                ("m", 172.8, 53.1),
                ("l", 220.8, 55.0),
                ("x", 441.5, 57.5),
            ],
            "RF-DETR (TopK)": [
                ("n", 80.3, 48.4),
                ("s", 146.8, 53.0),
                ("m", 193.0, 54.7),
                ("l", 307.2, 56.5),
                ("x", 667.1, 58.6),
                ("xxl", 927.3, 60.1),
            ],
        },
    },
    "m5_onnx_coreml": {
        "title": "Object Detection Models: Latency vs mAP (Apple M5, ONNX via CoreML EP)",
        "models": {
            "YOLO26 (E2E)": [
                ("n", 6.0, 40.1),
                ("s", 10.5, 47.8),
                ("m", 18.1, 52.5),
                ("l", 21.6, 54.4),
                ("x", 37.1, 56.9),
            ],
            "YOLO26 (NMS)": [
                ("n", 7.5, 40.9),
                ("s", 11.7, 48.6),
                ("m", 19.3, 53.1),
                ("l", 22.9, 55.0),
                ("x", 38.2, 57.5),
            ],
            "RF-DETR (TopK)": [
                ("n", 69.5, 48.4),
                ("s", 128.1, 53.0),
                ("m", 172.8, 54.7),
                ("l", 287.4, 56.5),
                ("x", 491.7, 58.6),
                ("xxl", 606.1, 60.1),
            ],
        },
    },
    # Ultralytics ProfileModels TRT-path methodology applied to CoreML mlpackages:
    # warmup=30 (3 rounds x 10), min_time=60s -> num_runs ~ 5000 per model,
    # sigma=2 max_iters=3 clip, ComputeUnit.ALL (default). Entries: (size, mean_ms, {ap}, std_ms).
    "m5_coreml": {
        "title": "Object Detection Models: Latency vs mAP (Apple M5, native CoreML .mlpackage, ProfileModels TRT-path: warmup=30, num_runs~5000, sigma=2 iters=3)",
        "models": {
            "YOLO26 (E2E)": [
                ("l", 10.35, {"ap": 54.4}, 0.06),
                ("x", 21.20, {"ap": 56.9}, 0.16),
            ],
            "YOLO26 (NMS)": [
                ("l", 10.32, {"ap": 55.0}, 0.06),
                ("x", 21.12, {"ap": 57.5}, 0.08),
            ],
            "YOLO26_RTDETR (obj365)": [
                ("l", 16.88, {"ap": 56.7}, 0.23),
                ("x", 28.01, {"ap": 58.4}, 0.29),
            ],
            "DEIM-DINOv3SPlus (obj365)": [
                ("xl/l6", 115.97, {"ap": 59.9}, 2.64),
            ],
        },
    },
    "xeon": {
        "title": "Object Detection Models: Latency vs mAP (Intel Xeon CPU @ 2.00GHz, ONNX)",
        "models": {
            "YOLO26 (E2E)": [
                ("n", 38.1, 40.1),
                ("s", 84.8, 47.8),
                ("m", 218.5, 52.5),
                ("l", 279.5, 54.4),
                ("x", 575.5, 56.9),
            ],
            "YOLO26 (NMS)": [
                ("n", 41.0, 40.9),
                ("s", 100.2, 48.6),
                ("m", 261.6, 53.1),
                ("l", 335.5, 55.0),
                ("x", 623.0, 57.5),
            ],
            "YOLO26_RTDETR": [
                ("n", 55.5, 41.2),
                ("ns", 89.1, 47.4),
                ("s", 200.4, 49.5),
                ("m", 338.1, 53.5),
                ("l", 411.8, 55.2),
                ("x", 664.9, 56.6),
            ],
            "RF-DETR (TopK)": [
                ("n", 114.3, 48.4),
                ("s", 203.3, 53.0),
                ("m", 266.1, 54.7),
                ("l", 410.5, 56.5),
                ("x", 931.1, 58.6),
                ("xxl", 1304.8, 60.1),
            ],
        },
    },
    "t4_reported": {
        "title": "Object Detection Models: Latency vs mAP (Tesla T4 GPU, TensorRT, Reported)",
        "models": {
            "YOLO26 (E2E)": [
                ("n", 1.7, {"ap": 40.1, "ap50": 55.6, "ap75": 43.5, "ap_small": 19.7, "ap_medium": 44.0, "ap_large": 58.4}),
                ("s", 2.5, {"ap": 47.8, "ap50": 64.6, "ap75": 52.2, "ap_small": 29.1, "ap_medium": 52.5, "ap_large": 64.3}),
                ("m", 4.7, {"ap": 52.5, "ap50": 69.8, "ap75": 57.2, "ap_small": 36.2, "ap_medium": 56.9, "ap_large": 68.5}),
                ("l", 6.2, {"ap": 54.4, "ap50": 71.5, "ap75": 59.4, "ap_small": 37.8, "ap_medium": 58.6, "ap_large": 70.3}),
                ("x", 11.8, {"ap": 56.9, "ap50": 74.1, "ap75": 62.1, "ap_small": 41.3, "ap_medium": 61.2, "ap_large": 72.7}),
            ],
            "YOLO26 (NMS)": [
                ("n", 1.7, {"ap": 40.9, "ap50": 56.8, "ap75": 44.3, "ap_small": 21.1, "ap_medium": 44.8, "ap_large": 59.1}),
                ("s", 2.5, {"ap": 48.6, "ap50": 65.8, "ap75": 52.8, "ap_small": 29.5, "ap_medium": 53.2, "ap_large": 65.8}),
                ("m", 4.7, {"ap": 53.1, "ap50": 70.7, "ap75": 57.7, "ap_small": 36.7, "ap_medium": 57.8, "ap_large": 68.9}),
                ("l", 6.2, {"ap": 55.0, "ap50": 72.5, "ap75": 60.0, "ap_small": 38.4, "ap_medium": 59.5, "ap_large": 71.1}),
                ("x", 11.8, {"ap": 57.5, "ap50": 75.0, "ap75": 62.7, "ap_small": 41.8, "ap_medium": 62.1, "ap_large": 73.3}),
            ],
            "DEIMv2": [
                ("pico", 1.7, {"ap": 38.5}),
                ("n", 2.0, {"ap": 43.0}),
                ("s", 5.78, {"ap": 50.9, "ap50": 68.3, "ap75": 55.1, "ap_small": 31.4, "ap_medium": 55.3, "ap_large": 70.3}),
                ("m", 8.80, {"ap": 53.0, "ap50": 70.2, "ap75": 57.6, "ap_small": 34.2, "ap_medium": 57.4, "ap_large": 71.5}),
                ("l", 10.47, {"ap": 56.0, "ap50": 73.4, "ap75": 60.9, "ap_small": 37.5, "ap_medium": 60.8, "ap_large": 75.2}),
                ("x", 13.75, {"ap": 57.8, "ap50": 75.4, "ap75": 63.2, "ap_small": 39.2, "ap_medium": 62.9, "ap_large": 75.9}),
            ],
            "RF-DETR (obj365)": [
                ("n", 2.3, {"ap": 48.4, "ap50": 67.5, "ap75": 51.7, "ap_small": 25.3, "ap_medium": 53.6, "ap_large": 71.0}),
                ("s", 3.5, {"ap": 53.0, "ap50": 72.0, "ap75": 57.1, "ap_small": 31.8, "ap_medium": 58.4, "ap_large": 73.1}),
                ("m", 4.4, {"ap": 54.7, "ap50": 73.6, "ap75": 59.1, "ap_small": 35.9, "ap_medium": 59.8, "ap_large": 73.7}),
                ("l", 6.8, {"ap": 56.5, "ap50": 75.1, "ap75": 61.2, "ap_small": 39.0, "ap_medium": 61.0, "ap_large": 74.0}),
                ("x", 11.5, {"ap": 58.6, "ap50": 77.5, "ap75": 64.0, "ap_small": 40.8, "ap_medium": 64.3, "ap_large": 76.3}),
                ("xxl", 17.2, {"ap": 60.1, "ap50": 78.5, "ap75": 65.8, "ap_small": 43.7, "ap_medium": 65.1, "ap_large": 76.3}),
            ],
            "RF-DETR (obj365, ECDet reported)": [
                # RF-DETR obj365 results as reported in ECDet paper (arXiv 2603.18739), TRT v10.6
                ("s", 3.65, {"ap": 52.9, "ap50": 71.9, "ap75": 57.0, "ap_small": 32.0, "ap_medium": 58.3, "ap_large": 73.0}),
                ("m", 4.62, {"ap": 54.7, "ap50": 73.5, "ap75": 59.2, "ap_small": 36.1, "ap_medium": 59.7, "ap_large": 73.8}),
                ("l", 7.38, {"ap": 56.5, "ap50": 75.1, "ap75": 61.3, "ap_small": 39.0, "ap_medium": 61.0, "ap_large": 73.9}),
                ("x", 14.79, {"ap": 58.6, "ap50": 77.4, "ap75": 63.8, "ap_small": 40.3, "ap_medium": 63.9, "ap_large": 76.2}),
            ],
            "LW-DETR (obj365)": [
                # LW-DETR obj365 results as reported in ECDet paper (arXiv 2603.18739), TRT v10.6
                ("s", 3.09, {"ap": 48.0, "ap50": 66.9, "ap75": 51.7, "ap_small": 26.8, "ap_medium": 52.5, "ap_large": 65.5}),
                ("m", 5.27, {"ap": 52.6, "ap50": 69.9, "ap75": 56.7, "ap_small": 32.6, "ap_medium": 57.7, "ap_large": 70.7}),
                ("l", 8.25, {"ap": 56.1, "ap50": 74.6, "ap75": 60.9, "ap_small": 37.2, "ap_medium": 60.4, "ap_large": 73.0}),
                ("x", 16.06, {"ap": 58.3, "ap50": 76.9, "ap75": 63.3, "ap_small": 40.9, "ap_medium": 63.3, "ap_large": 74.8}),
            ],
            "RT-DETRv4": [
                # RT-DETRv4 COCO-only results from ECDet paper (arXiv 2603.18739), TRT v10.6
                ("s", 3.60, {"ap": 49.7, "ap50": 66.8, "ap75": 54.1, "ap_small": 30.2, "ap_medium": 53.6, "ap_large": 66.9}),
                ("m", 5.66, {"ap": 53.5, "ap50": 71.1, "ap75": 58.1, "ap_small": 34.9, "ap_medium": 57.7, "ap_large": 72.1}),
                ("l", 8.10, {"ap": 55.4, "ap50": 73.0, "ap75": 60.3, "ap_small": 37.1, "ap_medium": 60.1, "ap_large": 72.9}),
                ("x", 12.90, {"ap": 57.0, "ap50": 74.6, "ap75": 62.1, "ap_small": 39.5, "ap_medium": 61.9, "ap_large": 74.8}),
            ],
            "ECDet": [
                # ECDet COCO-only results from arXiv 2603.18739, TRT v10.6
                ("s", 5.41, {"ap": 51.7, "ap50": 69.4, "ap75": 55.8, "ap_small": 32.3, "ap_medium": 56.4, "ap_large": 70.5}),
                ("m", 7.98, {"ap": 54.3, "ap50": 72.2, "ap75": 58.7, "ap_small": 35.9, "ap_medium": 59.1, "ap_large": 72.7}),
                ("l", 10.49, {"ap": 57.0, "ap50": 75.1, "ap75": 61.7, "ap_small": 38.7, "ap_medium": 62.5, "ap_large": 75.0}),
                ("x", 12.70, {"ap": 57.9, "ap50": 76.0, "ap75": 62.9, "ap_small": 38.7, "ap_medium": 63.4, "ap_large": 76.1}),
            ],
            "DEIMv1 D-FINE": [
                # DEIMv1 paper: Table 11 (nano/s/m) + Table 1 (l/x), applied to D-FINE backbone
                ("n", 2.12, {"ap": 43.0, "ap50": 60.4, "ap75": 46.2, "ap_small": 24.5, "ap_medium": 47.1, "ap_large": 62.1}),
                ("s", 3.49, {"ap": 49.0, "ap50": 65.9, "ap75": 53.1, "ap_small": 30.4, "ap_medium": 52.6, "ap_large": 65.7}),
                ("m", 5.55, {"ap": 52.7, "ap50": 70.0, "ap75": 57.3, "ap_small": 35.3, "ap_medium": 56.7, "ap_large": 69.5}),
                ("l", 8.07, {"ap": 54.7, "ap50": 72.4, "ap75": 59.4, "ap_small": 36.9, "ap_medium": 59.6, "ap_large": 71.8}),
                ("x", 12.89, {"ap": 56.5, "ap50": 74.0, "ap75": 61.5, "ap_small": 38.8, "ap_medium": 61.4, "ap_large": 74.2}),
            ],
            "DEIMv1 RT-DETRv2": [
                # DEIMv1 paper: Table 11 (s/m/m*) + Table 2 (l=R50, x=R101)
                ("s", 4.59, {"ap": 49.0, "ap50": 66.1, "ap75": 53.3, "ap_small": 32.6, "ap_medium": 52.5, "ap_large": 64.1}),
                ("m", 6.40, {"ap": 50.9, "ap50": 68.6, "ap75": 55.2, "ap_small": 34.3, "ap_medium": 54.4, "ap_large": 67.1}),
                ("m*", 6.90, {"ap": 53.2, "ap50": 71.2, "ap75": 57.8, "ap_small": 35.3, "ap_medium": 57.6, "ap_large": 70.2}),
                ("l", 9.29, {"ap": 54.3, "ap50": 72.3, "ap75": 58.8, "ap_small": 37.5, "ap_medium": 58.7, "ap_large": 70.8}),
                ("x", 13.88, {"ap": 55.5, "ap50": 73.5, "ap75": 60.3, "ap_small": 37.9, "ap_medium": 59.9, "ap_large": 73.0}),
            ],
            "RT-DETRv2": [
                # NOTE: Sub-metrics for S and M are wrong in DEIM papers (copy-paste from L and X rows).
                # ECDet paper (arXiv 2603.18739) reports correct values:
                #   S: ap75=52.1 ap_s=30.2 ap_m=51.5 ap_l=63.9 (DEIM papers copied L's sub-metrics)
                #   M: ap75=54.1 ap_s=32.0 ap_m=53.2 ap_l=66.5 (DEIM papers copied X's sub-metrics)
                # m* (R50vd_m) is absent from ECDet; values here are from DEIM papers only.
                ("s", 4.59, {"ap": 48.1, "ap50": 65.1, "ap75": 52.1, "ap_small": 30.2, "ap_medium": 51.5, "ap_large": 63.9}),
                ("m", 6.40, {"ap": 49.9, "ap50": 67.5, "ap75": 54.1, "ap_small": 32.0, "ap_medium": 53.2, "ap_large": 66.5}),
                ("m*", 6.90, {"ap": 51.9, "ap50": 69.9, "ap75": 56.5, "ap_small": 33.5, "ap_medium": 56.8, "ap_large": 69.2}),
                ("l", 9.29, {"ap": 53.4, "ap50": 71.6, "ap75": 57.4, "ap_small": 36.1, "ap_medium": 57.9, "ap_large": 70.8}),
                ("x", 13.88, {"ap": 54.3, "ap50": 72.8, "ap75": 58.8, "ap_small": 35.8, "ap_medium": 58.8, "ap_large": 72.1}),
            ],
            "D-FINE": [
                # D-FINE README COCO-only results; latency from DEIMv1 D-FINE reported values (same arch)
                ("n", 2.12, {"ap": 42.8, "ap50": 60.3, "ap75": 45.5, "ap_small": 22.9, "ap_medium": 46.8, "ap_large": 62.1}),
                ("s", 3.49, {"ap": 48.5, "ap50": 65.6, "ap75": 52.6, "ap_small": 29.1, "ap_medium": 52.2, "ap_large": 65.4}),
                ("m", 5.62, {"ap": 52.3, "ap50": 69.8, "ap75": 56.4, "ap_small": 33.2, "ap_medium": 56.5, "ap_large": 70.2}),
                ("l", 8.07, {"ap": 54.0, "ap50": 71.6, "ap75": 58.4, "ap_small": 36.5, "ap_medium": 58.0, "ap_large": 71.9}),
                ("x", 12.89, {"ap": 55.8, "ap50": 73.7, "ap75": 60.2, "ap_small": 37.3, "ap_medium": 60.5, "ap_large": 73.4}),
            ],
            "D-FINE (obj365)": [
                # D-FINE README Objects365+COCO results; latency from DEIMv1 D-FINE reported values
                ("s", 3.49, {"ap": 50.7, "ap50": 67.6, "ap75": 55.1, "ap_small": 32.7, "ap_medium": 54.6, "ap_large": 66.5}),
                ("m", 5.62, {"ap": 55.1, "ap50": 72.6, "ap75": 59.7, "ap_small": 37.9, "ap_medium": 59.4, "ap_large": 71.7}),
                ("l", 8.07, {"ap": 57.3, "ap50": 74.9, "ap75": 62.3, "ap_small": 40.6, "ap_medium": 61.5, "ap_large": 73.7}),
                ("x", 12.89, {"ap": 59.3, "ap50": 76.8, "ap75": 64.6, "ap_small": 42.3, "ap_medium": 64.2, "ap_large": 76.4}),
            ],
            "RT-DETR": [
                # RT-DETR v1 paper Table 2 (arXiv 2304.08069): COCO-only training
                # Sub-metrics from HuggingFace model card (PekingU/rtdetr_*)
                # Latency from main table FPS: R18=217→4.61ms, R50=108→9.26ms, R101=74→13.51ms
                ("s", 4.61, {"ap": 46.5, "ap50": 63.8, "ap75": 50.4, "ap_small": 28.4, "ap_medium": 49.8, "ap_large": 63.0}),
                ("l", 9.26, {"ap": 53.1, "ap50": 71.3, "ap75": 57.7, "ap_small": 34.8, "ap_medium": 58.0, "ap_large": 70.0}),
                ("x", 13.51, {"ap": 54.3, "ap50": 72.7, "ap75": 58.6, "ap_small": 36.0, "ap_medium": 58.8, "ap_large": 72.1}),
            ],
            "RT-DETR (obj365)": [
                # RT-DETR v1 paper Table C (arXiv 2304.08069): obj365 pretrain → COCO finetune
                # Latency from main table FPS: R18=217→4.61ms, R50=108→9.26ms, R101=74→13.51ms
                ("s", 4.61, {"ap": 49.2, "ap50": 66.6, "ap75": 53.5, "ap_small": 33.2, "ap_medium": 52.3, "ap_large": 64.8}),
                ("l", 9.26, {"ap": 55.3, "ap50": 73.4, "ap75": 60.1, "ap_small": 37.9, "ap_medium": 59.9, "ap_large": 71.8}),
                ("x", 13.51, {"ap": 56.2, "ap50": 74.6, "ap75": 61.3, "ap_small": 38.3, "ap_medium": 60.5, "ap_large": 73.5}),
            ],
        },
    },
    "t4": {
        "title": "Object Detection Models: Latency vs mAP (Tesla T4 GPU, TensorRT v10.11)",
        "models": {
            "YOLO26 (E2E)": [
                ("n", 1.8, {"ap": 40.1, "ap50": 55.6, "ap75": 43.5, "ap_small": 19.7, "ap_medium": 44.0, "ap_large": 58.4}, 0.0),
                ("s", 2.6, {"ap": 47.8, "ap50": 64.6, "ap75": 52.2, "ap_small": 29.1, "ap_medium": 52.5, "ap_large": 64.3}, 0.0),
                ("m", 4.8, {"ap": 52.5, "ap50": 69.8, "ap75": 57.2, "ap_small": 36.2, "ap_medium": 56.9, "ap_large": 68.5}, 0.1),
                ("l", 6.4, {"ap": 54.4, "ap50": 71.5, "ap75": 59.4, "ap_small": 37.8, "ap_medium": 58.6, "ap_large": 70.3}, 0.1),
                ("x", 11.7, {"ap": 56.9, "ap50": 74.1, "ap75": 62.1, "ap_small": 41.3, "ap_medium": 61.2, "ap_large": 72.7}, 0.3),
            ],
            "YOLO26 (NMS)": [
                ("n", 1.9, {"ap": 40.9, "ap50": 56.8, "ap75": 44.3, "ap_small": 21.1, "ap_medium": 44.8, "ap_large": 59.1}),
                ("s", 2.7, {"ap": 48.6, "ap50": 65.8, "ap75": 52.8, "ap_small": 29.5, "ap_medium": 53.2, "ap_large": 65.8}),
                ("m", 5.0, {"ap": 53.1, "ap50": 70.7, "ap75": 57.7, "ap_small": 36.7, "ap_medium": 57.8, "ap_large": 68.9}),
                ("l", 6.6, {"ap": 55.0, "ap50": 72.5, "ap75": 60.0, "ap_small": 38.4, "ap_medium": 59.5, "ap_large": 71.1}),
                ("x", 12.3, {"ap": 57.5, "ap50": 75.0, "ap75": 62.7, "ap_small": 41.8, "ap_medium": 62.1, "ap_large": 73.3}),
            ],
            "YOLO26_RTDETR": [
                ("n", 1.8, {"ap": 41.1, "ap50": 57.4, "ap75": 44.5, "ap_small": 21.7, "ap_medium": 44.5, "ap_large": 58.8}),
                ("ns", 2.5, {"ap": 47.7, "ap50": 65.1, "ap75": 51.5, "ap_small": 29.6, "ap_medium": 52.0, "ap_large": 64.0}),
                ("s", 4.4, {"ap": 51.0, "ap50": 68.4, "ap75": 55.6, "ap_small": 34.3, "ap_medium": 54.7, "ap_large": 66.9}),
                ("m", 6.5, {"ap": 54.0, "ap50": 71.5, "ap75": 58.5, "ap_small": 38.3, "ap_medium": 57.9, "ap_large": 68.8}),
                ("l", 8.2, {"ap": 55.3, "ap50": 73.0, "ap75": 60.2, "ap_small": 39.6, "ap_medium": 59.3, "ap_large": 70.7}),
                ("x", 13.5, {"ap": 56.5, "ap50": 74.0, "ap75": 61.6, "ap_small": 41.1, "ap_medium": 60.8, "ap_large": 71.5}),
            ],
            "RF-DETR (obj365, TopK)": RF_DETR_OBJ365_TOPK_IMGSZ,
            "YOLO26_RTDETR (obj365)": [
                ("l4", 7.1, {"ap": 56.5, "ap50": 74.1, "ap75": 61.6, "ap_small": 41.3, "ap_medium": 61.0, "ap_large": 70.9}, 0.1),
                ("l", 8.1, {"ap": 56.7, "ap50": 74.3, "ap75": 61.8, "ap_small": 41.7, "ap_medium": 61.1, "ap_large": 71.0}, 0.1),
                ("x", 13.2, {"ap": 58.4, "ap50": 75.8, "ap75": 64.0, "ap_small": 43.7, "ap_medium": 62.8, "ap_large": 73.9}, 0.3),
            ],
            "YOLO26L-RTDETR (obj365)": YOLO26L_RTDETR_OBJ365_IMGSZ_SWEEP,
            "YOLO26_Dfine (obj365)": [
                ("xl", 13.1, {"ap": 58.5, "ap50": 75.6, "ap75": 63.9, "ap_small": 43.8, "ap_medium": 62.7, "ap_large": 74.0}, 0.3),
            ],
            "D-FINE-DINOv3SPlus (obj365)": DFINE_DINOV3SPLUS_OBJ365_IMGSZ640,
            "D-FINE-DINOv3S (obj365)": DFINE_DINOV3S_OBJ365_IMGSZ_SWEEP,
            "ConvNeXt D-FINE": [
                ("t", 11.8, {"ap": 55.1, "ap50": 72.6, "ap75": 60.0, "ap_small": 37.6, "ap_medium": 59.6, "ap_large": 72.8}, 0.2),
                ("s", 15.1, {"ap": 56.9, "ap50": 74.5, "ap75": 62.1, "ap_small": 40.4, "ap_medium": 61.9, "ap_large": 74.0}, 0.5),
            ],
            "DEIM-EUPE ConvNeXt": DEIM_EUPE_CONVNEXT_COCO_IMGSZ640,
            "LW-DETR (obj365)": [
                # LW-DETR obj365 results as reported in ECDet paper (arXiv 2603.18739), TRT v10.6
                ("n", 2.0, {"ap": 42.6}),
                ("s", 2.9, {"ap": 48.0, "ap50": 66.9, "ap75": 51.7, "ap_small": 26.8, "ap_medium": 52.5, "ap_large": 65.5}),
                ("m", 5.1, {"ap": 52.6, "ap50": 69.9, "ap75": 56.7, "ap_small": 32.6, "ap_medium": 57.7, "ap_large": 70.7}),
                ("l", 8.5, {"ap": 56.1, "ap50": 74.6, "ap75": 60.9, "ap_small": 37.2, "ap_medium": 60.4, "ap_large": 73.0}),
                ("x", 18.4, {"ap": 58.3, "ap50": 76.9, "ap75": 63.3, "ap_small": 40.9, "ap_medium": 63.3, "ap_large": 74.8}),
            ],
            "DEIMv2 (Ultralytics)": [
                ("l", 10.7, {"ap": 56.2, "ap50": 73.5, "ap75": 61.2, "ap_small": 37.1, "ap_medium": 61.3, "ap_large": 74.9}),
                ("xl", 14.6, {"ap": 58.0, "ap50": 75.3, "ap75": 63.2, "ap_small": 39.6, "ap_medium": 63.3, "ap_large": 76.3}),
                ("xxl_v2", 25.0, {"ap": 59.4, "ap50": 76.7, "ap75": 64.9, "ap_small": 41.3, "ap_medium": 65.1, "ap_large": 76.8}, 0.8),
                ("xxl", 32.6, {"ap": 59.8, "ap50": 77.1, "ap75": 65.3, "ap_small": 42.8, "ap_medium": 65.5, "ap_large": 77.1}),
            ],
            "DEIM-DINOv3SPlus (obj365)": DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP,
            "DEIM-DINOv3B (obj365)": DEIM_DINOV3B_OBJ365_IMGSZ_SWEEP,
            # "DINOv3-RTDETR": [
            #     ("s", 4.3, {"ap": 50.3, "ap50": 69.0, "ap75": 54.4, "ap_small": 27.8, "ap_medium": 55.8, "ap_large": 72.5}),
            # ],
            # "DINOv3-RTDETR (obj365)": [
            #     ("s", 4.3, {"ap": 52.3, "ap50": 71.1, "ap75": 56.7, "ap_small": 33.6, "ap_medium": 57.6, "ap_large": 70.0}),
            # ],
            # "DINOv3-STA-RTDETR": [
            #     ("l3", 9.9, {"ap": 54.3, "ap50": 72.8, "ap75": 58.9, "ap_small": 35.1, "ap_medium": 59.7, "ap_large": 73.0}, 0.1),
            #     ("l6", 10.8, {"ap": 55.0, "ap50": 73.7, "ap75": 59.6, "ap_small": 36.3, "ap_medium": 60.4, "ap_large": 74.3}, 0.1),
            # ],
            # "DINOv3-STA-RTDETR (obj365)": [
            #     # NOTE: AP metrics are provided as fractional values in logs; converted here to percentage points.
            #     # Latency is kept equal to the current l6 entry until a dedicated obj365 latency measurement is available.
            #     ("l6", 10.8, {"ap": 56.8, "ap50": 75.2, "ap75": 61.8, "ap_small": 39.9, "ap_medium": 61.4, "ap_large": 74.5}, 0.1),
            # ],
            "DEIMv2": [
                ("pico", 1.7, {"ap": 38.5}),
                ("n", 2.0, {"ap": 43.0}),
                ("s", 4.6, {"ap": 50.9, "ap50": 68.3, "ap75": 55.1, "ap_small": 31.4, "ap_medium": 55.3, "ap_large": 70.3}),
                ("m", 7.4, {"ap": 53.0, "ap50": 70.2, "ap75": 57.6, "ap_small": 34.2, "ap_medium": 57.4, "ap_large": 71.5}),
                ("l", 9.7, {"ap": 56.0, "ap50": 73.4, "ap75": 60.9, "ap_small": 37.5, "ap_medium": 60.8, "ap_large": 75.2}),
                ("x", 14.0, {"ap": 57.8, "ap50": 75.4, "ap75": 63.2, "ap_small": 39.2, "ap_medium": 62.9, "ap_large": 75.9}),
            ],
            "DEIMv1 D-FINE": [
                # DEIMv1 paper: Table 11 (nano/s/m) + Table 1 (l/x), applied to D-FINE backbone
                ("n", 2.0, {"ap": 43.0, "ap50": 60.4, "ap75": 46.2, "ap_small": 24.5, "ap_medium": 47.1, "ap_large": 62.1}),
                ("s", 3.7, {"ap": 49.0, "ap50": 65.9, "ap75": 53.1, "ap_small": 30.4, "ap_medium": 52.6, "ap_large": 65.7}),
                ("m", 5.6, {"ap": 52.7, "ap50": 70.0, "ap75": 57.3, "ap_small": 35.3, "ap_medium": 56.7, "ap_large": 69.5}),
                ("l", 8.0, {"ap": 54.7, "ap50": 72.4, "ap75": 59.4, "ap_small": 36.9, "ap_medium": 59.6, "ap_large": 71.8}),
                ("x", 13.6, {"ap": 56.5, "ap50": 74.0, "ap75": 61.5, "ap_small": 38.8, "ap_medium": 61.4, "ap_large": 74.2}),
            ],
            "RT-DETRv2": [
                # NOTE: Sub-metrics for S and M are wrong in DEIM papers (copy-paste from L and X rows).
                # ECDet paper (arXiv 2603.18739) reports correct values:
                #   S: ap75=52.1 ap_s=30.2 ap_m=51.5 ap_l=63.9 (DEIM papers copied L's sub-metrics)
                #   M: ap75=54.1 ap_s=32.0 ap_m=53.2 ap_l=66.5 (DEIM papers copied X's sub-metrics)
                # m* (R50vd_m) is absent from ECDet; values here are from DEIM papers only.
                ("s", 4.0, {"ap": 48.1, "ap50": 65.1, "ap75": 52.1, "ap_small": 30.2, "ap_medium": 51.5, "ap_large": 63.9}),
                ("m", 5.6, {"ap": 49.9, "ap50": 67.5, "ap75": 54.1, "ap_small": 32.0, "ap_medium": 53.2, "ap_large": 66.5}),
                ("m*", 6.6, {"ap": 51.9, "ap50": 69.9, "ap75": 56.5, "ap_small": 33.5, "ap_medium": 56.8, "ap_large": 69.2}),
                ("l", 8.4, {"ap": 53.4, "ap50": 71.6, "ap75": 57.4, "ap_small": 36.1, "ap_medium": 57.9, "ap_large": 70.8}),
                ("x", 13.6, {"ap": 54.3, "ap50": 72.8, "ap75": 58.8, "ap_small": 35.8, "ap_medium": 58.8, "ap_large": 72.1}),
            ],
            "DEIMv1 RT-DETRv2": [
                # DEIMv1 paper: Table 11 (s/m/m*) + Table 2 (l=R50, x=R101)
                ("s", 4.1, {"ap": 49.0, "ap50": 66.1, "ap75": 53.3, "ap_small": 32.6, "ap_medium": 52.5, "ap_large": 64.1}),
                ("m", 5.7, {"ap": 50.9, "ap50": 68.6, "ap75": 55.2, "ap_small": 34.3, "ap_medium": 54.4, "ap_large": 67.1}),
                ("m*", 6.7, {"ap": 53.2, "ap50": 71.2, "ap75": 57.8, "ap_small": 35.3, "ap_medium": 57.6, "ap_large": 70.2}),
                ("l", 8.6, {"ap": 54.3, "ap50": 72.3, "ap75": 58.8, "ap_small": 37.5, "ap_medium": 58.7, "ap_large": 70.8}),
                ("x", 13.8, {"ap": 55.5, "ap50": 73.5, "ap75": 60.3, "ap_small": 37.9, "ap_medium": 59.9, "ap_large": 73.0}),
            ],
            # "D-FINE": [
            #     # D-FINE official training logs (COCO-only); latency from DEIMv1 D-FINE t4 measurements
            #     ("n", 2.0, {"ap": 42.8, "ap50": 60.3, "ap75": 45.5, "ap_small": 22.9, "ap_medium": 46.8, "ap_large": 62.1}),
            #     ("s", 3.7, {"ap": 48.5, "ap50": 65.6, "ap75": 52.6, "ap_small": 29.1, "ap_medium": 52.2, "ap_large": 65.4}),
            #     ("m", 5.6, {"ap": 52.3, "ap50": 69.8, "ap75": 56.4, "ap_small": 33.2, "ap_medium": 56.5, "ap_large": 70.2}),
            #     ("l", 8.0, {"ap": 54.0, "ap50": 71.6, "ap75": 58.4, "ap_small": 36.5, "ap_medium": 58.0, "ap_large": 71.9}),
            #     ("x", 13.6, {"ap": 55.8, "ap50": 73.7, "ap75": 60.2, "ap_small": 37.3, "ap_medium": 60.5, "ap_large": 73.4}),
            # ],
            "D-FINE (obj365)": [
                # D-FINE official training logs (Objects365+COCO); latency from DEIMv1 D-FINE t4 measurements
                ("s", 3.7, {"ap": 50.7, "ap50": 67.6, "ap75": 55.1, "ap_small": 32.7, "ap_medium": 54.6, "ap_large": 66.5}),
                ("m", 5.6, {"ap": 55.1, "ap50": 72.6, "ap75": 59.7, "ap_small": 37.9, "ap_medium": 59.4, "ap_large": 71.7}),
                ("l", 8.0, {"ap": 57.3, "ap50": 74.9, "ap75": 62.3, "ap_small": 40.6, "ap_medium": 61.5, "ap_large": 73.7}),
                ("x", 13.6, {"ap": 59.3, "ap50": 76.8, "ap75": 64.6, "ap_small": 42.3, "ap_medium": 64.2, "ap_large": 76.4}),
            ],

        },
    },
    "rf_compare": {
        "title": "Object Detection Models: Latency vs mAP (Tesla T4 GPU, TensorRT v10.11)",
        "models": {
            "RF-DETR (obj365, TopK)": RF_DETR_OBJ365_TOPK_IMGSZ,
            "RT-DETR (Ultralytics)": [
                ("l", 8.0, {"ap": 52.7, "ap50": 71.2, "ap75": 57.1, "ap_small": 34.4, "ap_medium": 57.3, "ap_large": 70.8}, 0.0),
                ("x", 13.6, {"ap": 54.4, "ap50": 72.6, "ap75": 59.1, "ap_small": 35.3, "ap_medium": 59.3, "ap_large": 72.2}, 0.4),
            ],
            "DEIM-DINOv3SPlus (obj365)": DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP,
            "DEIM-DINOv3B (obj365)": DEIM_DINOV3B_OBJ365_IMGSZ_SWEEP,
            "DEIM-DINOv3SPlus Light (obj365)": [
                # rtdetr_deim_dinov3sp_light_obj365_op17_nosim_norope_imgsz640*_fp32attn_debug_fp16.
                # DINOv3SPlus backbone with a light neck; AP metrics converted from fractions to percentage points.
                ("light/l4", 11.8, {"ap": 58.2, "ap50": 75.9, "ap75": 63.4, "ap_small": 41.2, "ap_medium": 63.3, "ap_large": 76.1}, 0.2),
                ("light/l6", 12.3, {"ap": 58.6, "ap50": 76.6, "ap75": 63.7, "ap_small": 41.5, "ap_medium": 63.8, "ap_large": 76.6}, 0.2),
            ],
            "DEIMv2 (Ultralytics)": [
                # DINOv3S neckless COCO study. In labels, n = neckless and ln = LayerNorm.
                ("s-n/l4", 7.2, {"ap": 52.8, "ap50": 70.4, "ap75": 57.6, "ap_small": 32.4, "ap_medium": 58.7, "ap_large": 73.1}, 0.1),
                # ("s-ln-n/l4", 7.2, {"ap": 53.0, "ap50": 70.7, "ap75": 57.7, "ap_small": 31.9, "ap_medium": 58.8, "ap_large": 72.9}, 0.1),
                ("s-n/l6", 7.5, {"ap": 53.1, "ap50": 70.8, "ap75": 57.8, "ap_small": 32.3, "ap_medium": 59.0, "ap_large": 73.1}, 0.1),
                # ("s-ln-n/l6", 7.6, {"ap": 53.4, "ap50": 71.1, "ap75": 58.0, "ap_small": 32.1, "ap_medium": 59.1, "ap_large": 73.8}, 0.2),
                ("sp-n/l6", 9.6, {"ap": 54.7, "ap50": 72.7, "ap75": 59.7, "ap_small": 34.6, "ap_medium": 60.6, "ap_large": 73.9}, 0.2),
                ("l", 10.7, {"ap": 56.2, "ap50": 73.5, "ap75": 61.2, "ap_small": 37.1, "ap_medium": 61.3, "ap_large": 74.9}),
                ("xl", 14.6, {"ap": 58.0, "ap50": 75.3, "ap75": 63.2, "ap_small": 39.6, "ap_medium": 63.3, "ap_large": 76.3}),
                ("xxl_v2", 25.0, {"ap": 59.4, "ap50": 76.7, "ap75": 64.9, "ap_small": 41.3, "ap_medium": 65.1, "ap_large": 76.8}, 0.8),
                ("xxl", 32.6, {"ap": 59.8, "ap50": 77.1, "ap75": 65.3, "ap_small": 42.8, "ap_medium": 65.5, "ap_large": 77.1}),
            ],
            "YOLO26_RTDETR (obj365)": [
                ("l4", 7.1, {"ap": 56.5, "ap50": 74.1, "ap75": 61.6, "ap_small": 41.3, "ap_medium": 61.0, "ap_large": 70.9}, 0.1),
                ("l", 8.1, {"ap": 56.7, "ap50": 74.3, "ap75": 61.8, "ap_small": 41.7, "ap_medium": 61.1, "ap_large": 71.0}, 0.1),
                ("x", 13.2, {"ap": 58.4, "ap50": 75.8, "ap75": 64.0, "ap_small": 43.7, "ap_medium": 62.8, "ap_large": 73.9}, 0.3),
            ],
            "YOLO26L-RTDETR (obj365)": YOLO26L_RTDETR_OBJ365_IMGSZ_SWEEP,
            "YOLO26_Dfine (obj365)": [
                ("xl", 13.1, {"ap": 58.5, "ap50": 75.6, "ap75": 63.9, "ap_small": 43.8, "ap_medium": 62.7, "ap_large": 74.0}, 0.3),
            ],
            "D-FINE-DINOv3SPlus (obj365)": DFINE_DINOV3SPLUS_OBJ365_IMGSZ640,
            "D-FINE-DINOv3S (obj365)": DFINE_DINOV3S_OBJ365_IMGSZ_SWEEP,
            "ConvNeXt D-FINE": [
                ("t", 11.8, {"ap": 55.1, "ap50": 72.6, "ap75": 60.0, "ap_small": 37.6, "ap_medium": 59.6, "ap_large": 72.8}, 0.2),
                ("s", 15.1, {"ap": 56.9, "ap50": 74.5, "ap75": 62.1, "ap_small": 40.4, "ap_medium": 61.9, "ap_large": 74.0}, 0.5),
            ],
            "DEIM-EUPE ConvNeXt": DEIM_EUPE_CONVNEXT_COCO_IMGSZ640,

            "YOLO26_RTDETR": [
                ("l", 8.2, {"ap": 55.3, "ap50": 73.0, "ap75": 60.2, "ap_small": 39.6, "ap_medium": 59.3, "ap_large": 70.7}),
            ],
            "YOLO26 (NMS)": [
                ("n", 1.9, {"ap": 40.9, "ap50": 56.8, "ap75": 44.3, "ap_small": 21.1, "ap_medium": 44.8, "ap_large": 59.1}),
                ("s", 2.7, {"ap": 48.6, "ap50": 65.8, "ap75": 52.8, "ap_small": 29.5, "ap_medium": 53.2, "ap_large": 65.8}),
                ("m", 5.0, {"ap": 53.1, "ap50": 70.7, "ap75": 57.7, "ap_small": 36.7, "ap_medium": 57.8, "ap_large": 68.9}),
                ("l", 6.6, {"ap": 55.0, "ap50": 72.5, "ap75": 60.0, "ap_small": 38.4, "ap_medium": 59.5, "ap_large": 71.1}),
                ("x", 12.3, {"ap": 57.5, "ap50": 75.0, "ap75": 62.7, "ap_small": 41.8, "ap_medium": 62.1, "ap_large": 73.3}),
            ],
        },
    },
    "t4_all_rf": {
        "title": "RF-DETR Obj365: All Sources vs DEIM-DINOv3SPlus (Tesla T4 GPU, TensorRT)",
        "models": {
            "RF-DETR (obj365, reported)": [
                # Original RF-DETR repo reported values (TRT v10.something); matches t4_reported entry.
                ("n", 2.3, {"ap": 48.4, "ap50": 67.5, "ap75": 51.7, "ap_small": 25.3, "ap_medium": 53.6, "ap_large": 71.0}),
                ("s", 3.5, {"ap": 53.0, "ap50": 72.0, "ap75": 57.1, "ap_small": 31.8, "ap_medium": 58.4, "ap_large": 73.1}),
                ("m", 4.4, {"ap": 54.7, "ap50": 73.6, "ap75": 59.1, "ap_small": 35.9, "ap_medium": 59.8, "ap_large": 73.7}),
                ("l", 6.8, {"ap": 56.5, "ap50": 75.1, "ap75": 61.2, "ap_small": 39.0, "ap_medium": 61.0, "ap_large": 74.0}),
                ("x", 11.5, {"ap": 58.6, "ap50": 77.5, "ap75": 64.0, "ap_small": 40.8, "ap_medium": 64.3, "ap_large": 76.3}),
                ("xxl", 17.2, {"ap": 60.1, "ap50": 78.5, "ap75": 65.8, "ap_small": 43.7, "ap_medium": 65.1, "ap_large": 76.3}),
            ],
            "RF-DETR (obj365, ECDet reported)": [
                # RF-DETR obj365 results as reported in ECDet paper (arXiv 2603.18739), TRT v10.6.
                ("s", 3.65, {"ap": 52.9, "ap50": 71.9, "ap75": 57.0, "ap_small": 32.0, "ap_medium": 58.3, "ap_large": 73.0}),
                ("m", 4.62, {"ap": 54.7, "ap50": 73.5, "ap75": 59.2, "ap_small": 36.1, "ap_medium": 59.7, "ap_large": 73.8}),
                ("l", 7.38, {"ap": 56.5, "ap50": 75.1, "ap75": 61.3, "ap_small": 39.0, "ap_medium": 61.0, "ap_large": 73.9}),
                ("x", 14.79, {"ap": 58.6, "ap50": 77.4, "ap75": 63.8, "ap_small": 40.3, "ap_medium": 63.9, "ap_large": 76.2}),
            ],
            "RF-DETR (obj365, our measurements)": RF_DETR_OBJ365_TOPK_IMGSZ,
            "DEIM-DINOv3SPlus (obj365)": DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP,
            # "YOLO26L-RTDETR (obj365)": YOLO26L_RTDETR_OBJ365_IMGSZ_SWEEP,
        },
    },
    "yolo27detr_compare": {
        "title": "YOLO27-DETR vs RF-DETR Obj365: Latency vs mAP (Tesla T4 GPU, TensorRT v10.11)",
        "models": {
            "YOLO27-DETR (obj365)": YOLO27DETR_OBJ365_IMGSZ_SWEEP,
            "RF-DETR (obj365, our benchmark)": RF_DETR_OBJ365_TOPK_IMGSZ,
        },
    },
    "yolo27detr_compare2": {
        "title": "YOLO27-DETR vs RF-DETR vs D-FINE Obj365: Latency vs mAP (Tesla T4 GPU, TensorRT v10.11)",
        "models": {
            "YOLO27-DETR (obj365)": YOLO27DETR_OBJ365_IMGSZ_SWEEP,
            "RF-DETR (obj365, our benchmark)": RF_DETR_OBJ365_TOPK_IMGSZ,
            "D-FINE (obj365)": [
                # D-FINE official training logs (Objects365+COCO); latency from DEIMv1 D-FINE t4 measurements.
                ("s", 3.7, {"ap": 50.7, "ap50": 67.6, "ap75": 55.1, "ap_small": 32.7, "ap_medium": 54.6, "ap_large": 66.5}),
                ("m", 5.6, {"ap": 55.1, "ap50": 72.6, "ap75": 59.7, "ap_small": 37.9, "ap_medium": 59.4, "ap_large": 71.7}),
                ("l", 8.0, {"ap": 57.3, "ap50": 74.9, "ap75": 62.3, "ap_small": 40.6, "ap_medium": 61.5, "ap_large": 73.7}),
                ("x", 13.6, {"ap": 59.3, "ap50": 76.8, "ap75": 64.6, "ap_small": 42.3, "ap_medium": 64.2, "ap_large": 76.4}),
            ],
        },
    },
    "t4_deimv2_xl_obj365_analysis": {
        "title": "DEIMv2-XL obj365: Export Variant Analysis (Tesla T4 GPU, TensorRT v10.11)",
        "models": {
            "Decoder Layers": [
                # Default model uses 6 decoder layers. export_eval_idx=eidx means eidx+1 layers are kept.
                ("2L", 13.5, {"ap": 58.4}),
                ("3L", 13.8, {"ap": 58.8}),
                ("4L", 13.9, {"ap": 59.2}),
                ("6L", 14.6, {"ap": 59.4}),
            ],
            "Queries @ 4L": [
                ("Q50", 13.1, {"ap": 57.7}),
                ("Q100", 13.5, {"ap": 58.7}),
                ("Q300", 13.9, {"ap": 59.2}),
            ],
            "Image Size / Export Variants": DEIM_DINOV3SPLUS_OBJ365_IMGSZ_SWEEP,
        },
    },
    "t4_deim_backbone_map": {
        "title": "DINOv3/ConvNeXt-S + DEIM: Latency vs mAP (Tesla T4 GPU, TensorRT v10.11)",
        "legend": {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)},
        "models": {
            "DINOv3 Small + DEIM": [
                ("s", 13.9, {"ap": 57.8, "ap50": 75.0, "ap75": 63.0, "ap_small": 39.2, "ap_medium": 63.0, "ap_large": 75.8}),
            ],
            "ConvNeXt-S + DEIM": [
                ("s", 15.6, {"ap": 57.0, "ap50": 74.5, "ap75": 62.1, "ap_small": 40.3, "ap_medium": 61.9, "ap_large": 74.2}),
            ],
        },
    },
    "t4_yolo26": {
        "title": "Object Detection Models: Latency vs mAP (Tesla T4 GPU, TensorRT — subset matching m5_coreml)",
        "models": {
            "YOLO26 (E2E)": [
                ("l", 6.4, {"ap": 54.4}, 0.1),
                ("x", 11.7, {"ap": 56.9}, 0.3),
            ],
            "YOLO26 (NMS)": [
                ("l", 6.6, {"ap": 55.0}),
                ("x", 12.3, {"ap": 57.5}),
            ],
            "YOLO26_RTDETR (obj365)": [
                ("l", 8.1, {"ap": 56.7}, 0.1),
                ("x", 13.2, {"ap": 58.4}, 0.3),
            ],
            "DEIM-DINOv3SPlus (obj365)": [
                ("xl/l6", 13.7, {"ap": 59.8}, 0.3),
            ],
        },
    },
}

# Marker and label offset config for each model
MODEL_STYLES = {
    "YOLO26": ("o", 8),
    "YOLO26 (E2E)": ("o", 8),
    "YOLO26 (NMS)": ("o", -12),
    "YOLO26-reported": ("o", -12),
    "YOLO26_RTDETR": ("^", -12),
    "YOLO26_RTDETR (obj365)": ("^", 8),
    "YOLO26L-RTDETR (obj365)": ("<", 8),
    "YOLO27-DETR (obj365)": ("*", 14),
    "YOLO26_Dfine (obj365)": ("D", -12),
    "D-FINE-DINOv3SPlus (obj365)": ("X", -16),
    "D-FINE-DINOv3S (obj365)": ("P", 10),
    "DINOv3-RTDETR": ("X", 8),
    "DINOv3-RTDETR (obj365)": ("X", -12),
    "DINOv3-STA-RTDETR": ("X", -12),
    "DINOv3-STA-RTDETR (obj365)": ("X", 8),
    "RF-DETR (obj365)": ("s", -12),
    "RF-DETR (obj365, reported)": ("s", -12),
    "RF-DETR (obj365, TopK)": ("s", -12),
    "RF-DETR (obj365, our measurements)": ("s", -12),
    "RF-DETR (obj365, our benchmark)": ("s", -12),
    "RF-DETR (obj365, ECDet reported)": ("s", 8),
    "LW-DETR (obj365)": ("^", 8),
    "RT-DETRv4": ("^", -12),
    "ECDet": ("h", 8),
    "DEIM D-FINE": ("D", -12),
    "DEIM RT-DETRv2": ("v", 8),
    "DEIMv2": ("p", -12),
    "DEIMv1 D-FINE": ("D", 8),
    "DEIMv1 RT-DETRv2": ("v", 8),
    "RT-DETRv2 (paper)": ("v", -12),
    "D-FINE": ("D", -12),
    "D-FINE (obj365)": ("D", 8),
    "RT-DETR": ("v", -12),
    "RT-DETR (obj365)": ("v", 8),
    "RT-DETR (Ultralytics)": ("v", -12),
    "ConvNeXt D-FINE": ("h", 8),
    "DEIM-EUPE ConvNeXt": ("h", -12),
    "DEIMv2 (Ultralytics)": ("p", 8),
    "DEIM-DINOv3SPlus (obj365)": ("*", 14),
    "DEIM-DINOv3B (obj365)": ("*", -12),
    "DEIM-DINOv3SPlus Light (obj365)": ("*", -12),
    "Decoder Layers": ("o", 8),
    "Queries @ 4L": ("s", -12),
    "Image Size / Export Variants": ("*", 14),
    "DINOv3 Small + DEIM": ("X", 8),
    "ConvNeXt-S + DEIM": ("D", -12),
}


def get_metric_value(y_value, metric):
    if isinstance(y_value, dict):
        return y_value.get(metric)
    return y_value if metric == "ap" else None


def plot_series(ax, points, label, color, marker, label_offset, metric):
    parsed_points = []
    for point in points:
        if len(point) == 3:
            size, x_value, y_value = point
            x_error = 0.0
        elif len(point) == 4:
            size, x_value, y_value, x_error = point
        else:
            raise ValueError(
                f"Point '{point}' must have 3 or 4 values: (size, latency, metric_value[, latency_error])."
            )

        metric_value = get_metric_value(y_value, metric)
        if metric_value is None:
            continue
        parsed_points.append((size, x_value, metric_value, x_error))

    if not parsed_points:
        return False

    xs = [point[1] for point in parsed_points]
    ys = [point[2] for point in parsed_points]
    x_errors = [point[3] for point in parsed_points]

    if any(x_errors):
        ax.errorbar(
            xs,
            ys,
            xerr=x_errors,
            label=label,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=7,
            capsize=3,
        )
    else:
        ax.plot(
            xs,
            ys,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=7,
        )

    for size, x_value, y_value, _ in parsed_points:
        ax.annotate(
            size,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, label_offset),
            ha="center",
            fontsize=9,
            color=color,
        )
    return True


def build_plot(output_path: Path, show: bool, metric: str, benchmark_name: str) -> None:
    benchmark_data = BENCHMARKS[benchmark_name]
    models = benchmark_data["models"]
    title = benchmark_data["title"]
    legend_kwargs = benchmark_data.get("legend", {"loc": "lower right"})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

    palette = plt.get_cmap("tab20")
    # palette = plt.get_cmap("tab10")
    plotted = False
    for i, (model_name, data) in enumerate(models.items()):
        marker, label_offset = MODEL_STYLES.get(model_name, ("o", 8))
        plotted |= plot_series(ax, data, model_name, palette(i), marker, label_offset, metric)

    if not plotted:
        raise ValueError(f"No '{metric}' values available for benchmark '{benchmark_name}'.")

    title_metric = METRIC_TITLE_TOKENS[metric]
    ax.set_title(title if metric == "ap" else title.replace("mAP", title_metric))
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(frameon=True, fontsize=9, **legend_kwargs)
    ax.margins(x=0.05, y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot object detection model latency vs mAP benchmarks."
    )
    parser.add_argument(
        "--metric",
        choices=list(METRIC_LABELS.keys()),
        default=DEFAULT_METRIC,
        help="Y-axis metric to plot. AP keeps existing mAP50-95 values.",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS.keys()),
        default=BENCHMARK,
        help="Benchmark dataset to plot.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(__file__).with_name(
        f"benchmark_plot_{args.benchmark}.png"
        if args.metric == "ap"
        else f"benchmark_plot_{args.benchmark}_{args.metric}.png"
    )
    build_plot(output_path, args.show, args.metric, args.benchmark)


if __name__ == "__main__":
    main()
