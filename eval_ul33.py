"""Evaluate UL33 prediction JSONs with native or image-size-calibrated area buckets.

By default, Tiny, Small, Medium, and Large use native bounding-box pixel areas. Pass
``--ref-size 640`` to scale each area by
``(640 / max(image_width, image_height)) ** 2`` before assigning its size bucket.
F1 is size-agnostic and uses the Ultralytics mean-class score at the best confidence
threshold and 0.5 IoU. Overall mAP uses Ultralytics DetMetrics matching, while size
mAP uses faster-coco-eval, matching the phase-2 training evaluator.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from faster_coco_eval import COCO, COCOeval_faster

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DETECTION_AREA_RANGES
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou

DEFAULT_RUN_ROOT = Path("/data/shared-datasets/fatih-runs/classify/yolo-next-encoder")
SIZE_LABELS = {"tiny": "T", "small": "S", "medium": "M", "large": "L"}
MIN_TINY_OBJECTS = 5
MIN_CALIBRATED_OBJECTS = 10
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def build_ground_truth(
    data_yaml: Path, split: str, reference_size: int | None
) -> tuple[COCO, dict[str, int], dict[str, int]]:
    """Build COCO ground truth and filename IDs for one UL33 validation split."""
    data = check_det_dataset(str(data_yaml), autodownload=False, split=split)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "cache": False, "workers": 0})
    dataset = build_yolo_dataset(cfg, data[split], 16, data, mode="val", stride=32)
    annotations, images = [], []
    file_ids = {Path(path).name: image_id for image_id, path in enumerate(dataset.im_files, 1)}
    for path, label in zip(dataset.im_files, dataset.labels):
        image_id = file_ids[Path(path).name]
        height, width = label["shape"]
        images.append({"id": image_id, "file_name": Path(path).name, "height": height, "width": width})
        boxes = ops.xywh2ltwh(label["bboxes"]) * (width, height, width, height)
        area_scale = (reference_size / max(width, height)) ** 2 if reference_size else 1.0
        for cls, box in zip(label["cls"].reshape(-1), boxes):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": box.tolist(),
                    "area": float(box[2] * box[3]) * area_scale,
                    "iscrowd": 0,
                }
            )
    return (
        COCO(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": int(i) + 1, "name": name} for i, name in data["names"].items()],
            }
        ),
        file_ids,
        {
            size: sum(low <= annotation["area"] < high for annotation in annotations)
            for size, (low, high) in DETECTION_AREA_RANGES.items()
        },
    )


def evaluate_predictions(
    ground_truth: COCO,
    detections: list[dict],
    reference_size: int | None,
    max_det: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate predictions across the UL33 detection area ranges."""
    predictions = ground_truth.loadRes(detections)
    if reference_size:
        for annotation in predictions.anns.values():
            image = ground_truth.imgs[annotation["image_id"]]
            annotation["area"] *= (reference_size / max(image["width"], image["height"])) ** 2
    evaluator = COCOeval_faster(
        ground_truth,
        predictions,
        iouType="bbox",
        ranges=DETECTION_AREA_RANGES,
        lvis_style=False,
        print_function=lambda *args, **kwargs: None,
    )
    evaluator.params.imgIds = ground_truth.getImgIds()
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ap50 = {}
    for size in ("all", *DETECTION_AREA_RANGES):
        precision = evaluator.eval["precision"][
            np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0],
            :,
            :,
            evaluator.params.areaRngLbl.index(size),
            -1,
        ]
        ap50[size] = float(precision[precision > -1].mean()) if np.any(precision > -1) else -1.0
    return ap50, {size: float(evaluator.stats_as_dict[f"AP_{size}"]) for size in ("all", *DETECTION_AREA_RANGES)}


def calculate_overall_metrics(ground_truth: COCO, detections: list[dict], max_det: int) -> dict[str, float]:
    """Calculate overall Ultralytics metrics from saved detections."""
    detections_by_image = defaultdict(list)
    for detection in detections:
        detections_by_image[detection["image_id"]].append(detection)

    correct, confidence, predicted_classes, target_classes = [], [], [], []
    for image_id in ground_truth.getImgIds():
        targets = ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=[image_id]))
        predictions = sorted(detections_by_image[image_id], key=lambda item: item["score"], reverse=True)[:max_det]
        target_classes.extend(target["category_id"] for target in targets)
        matches = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
        if targets and predictions:
            iou = box_iou(
                ops.ltwh2xyxy(torch.tensor([target["bbox"] for target in targets])),
                ops.ltwh2xyxy(torch.tensor([prediction["bbox"] for prediction in predictions])),
            ).numpy()
            target_cls = np.array([target["category_id"] for target in targets])
            prediction_cls = np.array([prediction["category_id"] for prediction in predictions])
            iou *= target_cls[:, None] == prediction_cls
            for i, threshold in enumerate(IOU_THRESHOLDS):
                target_idx, prediction_idx = np.nonzero(iou >= threshold)
                if target_idx.size:
                    pairs = np.column_stack((target_idx, prediction_idx, iou[target_idx, prediction_idx]))
                    pairs = pairs[pairs[:, 2].argsort()[::-1]]
                    pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
                    pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
                    matches[pairs[:, 1].astype(int), i] = True
        correct.append(matches)
        confidence.extend(prediction["score"] for prediction in predictions)
        predicted_classes.extend(prediction["category_id"] for prediction in predictions)

    _, _, precision, recall, _, ap, *_ = ap_per_class(
        np.concatenate(correct),
        np.array(confidence),
        np.array(predicted_classes),
        np.array(target_classes),
    )
    precision, recall = float(precision.mean()), float(recall.mean())
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50-95": float(ap.mean()),
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_dataset(
    dataset: str,
    run_dirs: tuple[Path, ...],
    data_root: Path | None,
    reference_size: int | None,
    max_det: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Evaluate one dataset for every requested run."""
    run_args = yaml.safe_load((run_dirs[0] / dataset / "args.yaml").read_text())
    data_yaml = data_root / dataset / "data.yaml" if data_root else Path(run_args["data"])
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth, file_ids, size_objects = build_ground_truth(
            data_yaml, run_args.get("split", "val"), reference_size
        )
    results = {}
    for run_dir in run_dirs:
        detections = json.loads((run_dir / dataset / "predictions.json").read_text())
        for detection in detections:
            detection["image_id"] = file_ids[detection["file_name"]]
        with contextlib.redirect_stdout(io.StringIO()):
            ap50, ap5095 = evaluate_predictions(ground_truth, detections, reference_size, max_det)
        results[run_dir.name] = {
            **calculate_overall_metrics(ground_truth, detections, max_det),
            **{f"mAP50-{SIZE_LABELS[size]}": ap50[size] for size in DETECTION_AREA_RANGES},
            **{f"mAP50-95-{SIZE_LABELS[size]}": ap5095[size] for size in DETECTION_AREA_RANGES},
        }
        for size, count in size_objects.items():
            minimum = MIN_CALIBRATED_OBJECTS if reference_size else MIN_TINY_OBJECTS if size == "tiny" else 1
            if count < minimum:
                results[run_dir.name].pop(f"mAP50-{SIZE_LABELS[size]}")
                results[run_dir.name].pop(f"mAP50-95-{SIZE_LABELS[size]}")
    return dataset, results


def main() -> None:
    """Parse arguments, evaluate prediction JSONs, and print per-dataset and macro metrics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Absolute run directories or names under --run-root")
    parser.add_argument("--data-root", type=Path, help="Optional root containing <dataset>/data.yaml")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300, help="COCO maxDets cap (default: 300)")
    parser.add_argument(
        "--ref-size",
        type=int,
        help="Normalize areas by (SIZE / longest image side)^2 before assigning size buckets",
    )
    args = parser.parse_args()
    run_dirs = tuple(Path(run) if Path(run).is_absolute() else args.run_root / run for run in args.runs)
    datasets = sorted(path.parent.name for path in run_dirs[0].glob("*/predictions.json"))
    if not datasets:
        raise FileNotFoundError(f"No prediction JSONs found under {run_dirs[0]}")

    values = {run_dir.name: defaultdict(list) for run_dir in run_dirs}
    dataset_scores = {run_dir.name: {} for run_dir in run_dirs}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_dataset, dataset, run_dirs, args.data_root, args.ref_size, args.max_det): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset, results = future.result()
            for run, metrics in results.items():
                dataset_scores[run][dataset] = {
                    metric: round(100 * value, 4) for metric, value in metrics.items() if value >= 0
                }
                for metric, value in metrics.items():
                    if value >= 0:
                        values[run][metric].append(value)

    output = {
        "runs": [
            {
                "run": run,
                "dataset_count": len(datasets),
                "datasets": dict(sorted(dataset_scores[run].items())),
                "metrics": {metric: round(100 * np.mean(items), 4) for metric, items in metrics.items()},
                "valid_dataset_counts": {metric: len(items) for metric, items in metrics.items()},
            }
            for run, metrics in values.items()
        ]
    }
    if args.ref_size:
        output["size_calibration"] = {
            "formula": "area * (reference_size / max(image_width, image_height))^2",
            "min_bucket_objects": MIN_CALIBRATED_OBJECTS,
            "reference_size": args.ref_size,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
