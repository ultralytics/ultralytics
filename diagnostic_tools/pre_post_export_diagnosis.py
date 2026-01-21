"""
YOLO Pre/Post Export Diagnostic Tool.

Compares detection outputs before (PyTorch) and after (ONNX Runtime) export.
Reports pairing IoU, confidence drift, and unmatched detections.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ultralytics import YOLO


def iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def extract_detections(results):
    dets = []
    for b in results.boxes.data.cpu().numpy():
        dets.append(
            {
                "box": b[:4].tolist(),
                "score": float(b[4]),
                "cls": int(b[5]),
            }
        )
    return dets


def run_pytorch(weights, source):
    model = YOLO(weights)
    res = model(source)[0]
    return extract_detections(res)


def run_onnx(weights, source, do_export):
    onnx_path = Path(weights).with_suffix(".onnx")

    if do_export or not onnx_path.exists():
        YOLO(weights).export(format="onnx", opset=20)

    model = YOLO(onnx_path, task="detect")
    res = model(source)[0]
    return extract_detections(res)


def pair_and_report(pre, post):
    matched = []
    matched_pre_idx = set()
    matched_post_idx = set()

    for i, p in enumerate(pre):
        best_iou = 0.0
        best_j = None

        for j, q in enumerate(post):
            if j in matched_post_idx or p["cls"] != q["cls"]:
                continue

            v = iou(p["box"], q["box"])
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j is not None and best_iou > 0.5:
            matched_pre_idx.add(i)
            matched_post_idx.add(best_j)
            matched.append(
                {
                    "cls": p["cls"],
                    "iou": best_iou,
                    "score_diff": abs(p["score"] - post[best_j]["score"]),
                }
            )

    unmatched_pre = [p for i, p in enumerate(pre) if i not in matched_pre_idx]
    unmatched_post = [q for j, q in enumerate(post) if j not in matched_post_idx]

    return matched, unmatched_pre, unmatched_post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--source", default="bus.jpg")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)

    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    print("=" * 38)
    print("YOLO Pre/Post Export Diagnostic Report")
    print("=" * 38)

    pre = run_pytorch(weights, source)
    post = run_onnx(weights, source, args.export)

    matched, un_pre, un_post = pair_and_report(pre, post)

    print("\nSummary:")
    print(f"Matched detections : {len(matched)}")
    print(f"Unmatched PRE      : {len(un_pre)}")
    print(f"Unmatched POST     : {len(un_post)}")

    if matched:
        ious = [m["iou"] for m in matched]
        diffs = [m["score_diff"] for m in matched]

        print("\nDrift:")
        print(f"Mean IoU           : {np.mean(ious):.3f}")
        print(f"Mean score diff    : {np.mean(diffs):.4f}")
        print(f"Max score diff     : {np.max(diffs):.4f}")

        print("\nPer-detection pairing:")
        for m in matched:
            print(f"CLS {m['cls']} | IoU {m['iou']:.3f} | Score Δ {m['score_diff']:.4f}")

    if un_pre:
        print("\nUnmatched PRE detections:")
        for d in un_pre:
            print(d)

    if un_post:
        print("\nUnmatched POST detections:")
        for d in un_post:
            print(d)


if __name__ == "__main__":
    main()
