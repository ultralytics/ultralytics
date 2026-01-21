"""
YOLO Pre/Post Export Diagnostic Tool

Compares YOLO detection behavior before and after ONNX export.
Focuses on numerical drift and missing detections.
"""

import argparse
import contextlib
import io
from ultralytics import YOLO


def compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def extract_detections(result):
    boxes = result.boxes
    detections = []

    for i in range(len(boxes)):
        detections.append({
            "box": boxes.xyxy[i].tolist(),
            "score": float(boxes.conf[i]),
            "cls": int(boxes.cls[i]),
        })

    return detections


def pair_detections(pre_dets, post_dets, iou_thresh=0.5):
    matches = []
    used_post = set()

    for pre in pre_dets:
        best_iou = 0.0
        best_idx = None

        for j, post in enumerate(post_dets):
            if j in used_post:
                continue
            if pre["cls"] != post["cls"]:
                continue

            iou = compute_iou(pre["box"], post["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx is not None and best_iou >= iou_thresh:
            matches.append({
                "cls": pre["cls"],
                "iou": best_iou,
                "score_diff": abs(pre["score"] - post_dets[best_idx]["score"]),
                "pre": pre,
                "post": post_dets[best_idx],
            })
            used_post.add(best_idx)

    unmatched_pre = [d for d in pre_dets if d not in [m["pre"] for m in matches]]
    unmatched_post = [d for i, d in enumerate(post_dets) if i not in used_post]

    return matches, unmatched_pre, unmatched_post


def compute_metrics(matches):
    box_diffs = []
    score_diffs = []

    for m in matches:
        for a, b in zip(m["pre"]["box"], m["post"]["box"]):
            box_diffs.append(abs(a - b))
        score_diffs.append(m["score_diff"])

    return {
        "box_mean": sum(box_diffs) / len(box_diffs) if box_diffs else 0.0,
        "box_max": max(box_diffs) if box_diffs else 0.0,
        "score_mean": sum(score_diffs) / len(score_diffs) if score_diffs else 0.0,
        "score_max": max(score_diffs) if score_diffs else 0.0,
    }


def run_pre_inference():
    model = YOLO("yolov8n.pt")
    with contextlib.redirect_stdout(io.StringIO()):
        results = model("bus.jpg", verbose=False)
    return extract_detections(results[0])


def run_post_inference():
    model = YOLO("yolov8n.pt")
    with contextlib.redirect_stdout(io.StringIO()):
        onnx_path = model.export(format="onnx", imgsz=640, simplify=False, opset=20)
        onnx_model = YOLO(onnx_path, task="detect")
        results = onnx_model("bus.jpg", verbose=False)
    return extract_detections(results[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO pre/post export diagnostic tool")
    parser.add_argument(
        "--verbose-detections",
        action="store_true",
        help="Print raw PRE and POST detections",
    )
    args = parser.parse_args()

    print("======================================")
    print("YOLO Pre/Post Export Diagnostic Report")
    print("======================================\n")

    pre = run_pre_inference()
    post = run_post_inference()

    matches, unmatched_pre, unmatched_post = pair_detections(pre, post)
    metrics = compute_metrics(matches)

    print("Paired detection comparison (CLS | IoU | Δscore):")
    for m in matches:
        print(f"CLS {m['cls']} | IoU: {m['iou']:.3f} | Score Δ: {m['score_diff']:.4f}")

    print("\nSummary:")
    print(f"Matched detections     : {len(matches)}")
    print(f"Unmatched PRE          : {len(unmatched_pre)}")
    print(f"Unmatched POST         : {len(unmatched_post)}\n")

    print("Box drift (pixels):")
    print(f"  Mean absolute diff   : {metrics['box_mean']:.3f}")
    print(f"  Max absolute diff    : {metrics['box_max']:.3f}\n")

    print("Confidence drift:")
    print(f"  Mean score diff      : {metrics['score_mean']:.4f}")
    print(f"  Max score diff       : {metrics['score_max']:.4f}\n")

    verdict = "PASS"
    if metrics["score_max"] > 0.15 or unmatched_pre or unmatched_post:
        verdict = "WARN"

    print(f"Verification verdict   : {verdict}\n")

    if unmatched_pre:
        print("Unmatched PRE detections:")
        for d in unmatched_pre:
            print(d)

    if unmatched_post:
        print("\nUnmatched POST detections:")
        for d in unmatched_post:
            print(d)

    if args.verbose_detections:
        print("\n--------------------------------------")
        print("Raw PRE detections:")
        for d in pre:
            print(d)
        print("\nRaw POST detections:")
        for d in post:
            print(d)
