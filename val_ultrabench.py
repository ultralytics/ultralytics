# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Score overall mAP/F1 and size-aware AP from Ultra Benchmark prediction JSONs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import YAML, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ap_per_class, box_iou

check_requirements("faster-coco-eval>=1.7.0")
from faster_coco_eval import COCO, COCOeval_faster  # noqa: E402, RUF100

DATASETS = Path(__file__).with_name("ul37_platform_uris.txt")
REFERENCE_SIZE = 640
MIN_BUCKET_OBJECTS = 10
SIZE_RANGES = {"tiny": [0, 10**2], "small": [0, 32**2], "medium": [32**2, 96**2], "large": [96**2, 1e5**2]}
SIZE_LABELS = {"tiny": "T", "small": "S", "medium": "M", "large": "L"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
FRACTION = [1000, 1.0, 0.0]


def build_ground_truth(
    data_yaml: str | Path, reference_size: int | None
) -> tuple[COCO, dict[str, int], dict[str, int]]:
    """Build COCO ground truth for one validation split."""
    data = check_det_dataset(data_yaml, autodownload=False, split="val")
    cfg = get_cfg(overrides={"task": "detect", "imgsz": REFERENCE_SIZE, "cache": False, "workers": 0})
    dataset = build_yolo_dataset(cfg, data["val"], 16, data, mode="val", stride=32)
    annotations, images = [], []
    file_ids = {Path(path).name: image_id for image_id, path in enumerate(dataset.im_files)}
    if len(file_ids) != len(dataset.im_files):
        raise ValueError(f"Duplicate validation filenames in {data_yaml}")
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
            for size, (low, high) in SIZE_RANGES.items()
        },
    )


def evaluate_predictions(
    ground_truth: COCO, detections: list[dict], max_det: int, reference_size: int | None
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate size-aware AP50 and AP50-95 from saved detections."""
    predictions = (
        ground_truth.loadRes(detections)
        if detections
        else COCO(
            {
                "images": ground_truth.dataset["images"],
                "annotations": [],
                "categories": ground_truth.dataset["categories"],
            }
        )
    )
    if reference_size:
        for annotation in predictions.anns.values():
            image = ground_truth.imgs[annotation["image_id"]]
            annotation["area"] *= (reference_size / max(image["width"], image["height"])) ** 2
    evaluator = COCOeval_faster(
        ground_truth,
        predictions,
        iouType="bbox",
        ranges=SIZE_RANGES,
        lvis_style=False,
        print_function=lambda *args, **kwargs: None,
    )
    evaluator.params.imgIds = ground_truth.getImgIds()
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ap50 = {}
    for size in SIZE_RANGES:
        precision = evaluator.eval["precision"][
            np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0],
            :,
            :,
            evaluator.params.areaRngLbl.index(size),
            -1,
        ]
        ap50[size] = float(precision[precision > -1].mean()) if np.any(precision > -1) else -1.0
    return ap50, {size: float(evaluator.stats_as_dict[f"AP_{size}"]) for size in SIZE_RANGES}


def calculate_overall_metrics(ground_truth: COCO, detections: list[dict], max_det: int) -> dict[str, float]:
    """Calculate overall Ultralytics mAP and F1 from saved detections."""
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
    dataset: str, data_yaml: Path, run_dir: Path, max_det: int, reference_size: int | None
) -> tuple[str, dict[str, float]]:
    """Evaluate one dataset from its prediction artifact."""
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth, file_ids, size_objects = build_ground_truth(data_yaml, reference_size)
    detections = []
    for detection in json.loads((run_dir / dataset / "predictions.json").read_text()):
        if (image_id := file_ids.get(detection["file_name"])) is not None:
            detection["image_id"] = image_id
            detections.append(detection)
    with contextlib.redirect_stdout(io.StringIO()):
        ap50, ap5095 = evaluate_predictions(ground_truth, detections, max_det, reference_size)
    metrics = {
        **calculate_overall_metrics(ground_truth, detections, max_det),
        **{f"mAP50-{SIZE_LABELS[size]}": ap50[size] for size in SIZE_RANGES},
        **{f"mAP50-95-{SIZE_LABELS[size]}": ap5095[size] for size in SIZE_RANGES},
    }
    for size, count in size_objects.items():
        if count < MIN_BUCKET_OBJECTS:
            metrics.pop(f"mAP50-{SIZE_LABELS[size]}")
            metrics.pop(f"mAP50-95-{SIZE_LABELS[size]}")
    return dataset, metrics


def evaluate_run(run_dir: Path, imgsz: int = REFERENCE_SIZE, reference_size: int | None = REFERENCE_SIZE) -> dict:
    """Validate and aggregate one completed Ultra Benchmark run."""
    datasets = {
        Path(uri).name: uri
        for raw in DATASETS.read_text().splitlines()
        if (uri := raw.strip()) and not uri.startswith("#")
    }
    if len(datasets) != 37:
        raise ValueError(f"Ultra Benchmark requires 37 unique Platform datasets, found {len(datasets)}")
    predictions = {path.parent.name for path in run_dir.glob("*/predictions.json")}
    if predictions != datasets.keys():
        raise ValueError(
            f"Ultra Benchmark predictions differ: missing={sorted(datasets.keys() - predictions)}, "
            f"extra={sorted(predictions - datasets.keys())}"
        )
    data_yamls, max_dets = {}, {}
    for dataset, uri in datasets.items():
        saved = YAML.load(run_dir / dataset / "args.yaml")
        data_yaml = Path(saved["data"]) if saved.get("data") else None
        if (
            saved.get("platform_uri") != uri
            or saved.get("imgsz") != imgsz
            or type(saved.get("max_det")) is not int
            or data_yaml is None
            or not data_yaml.is_file()
        ):
            raise ValueError(
                f"Invalid local dataset provenance for {dataset}: {saved.get('platform_uri')}, "
                f"imgsz={saved.get('imgsz')}, {data_yaml}"
            )
        data_yamls[dataset] = data_yaml
        max_dets[dataset] = saved["max_det"]

    rows = {}
    values = defaultdict(list)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {
            executor.submit(
                evaluate_dataset, dataset, data_yamls[dataset], run_dir, max_dets[dataset], reference_size
            ): dataset
            for dataset in datasets
        }
        for future in as_completed(futures):
            dataset, metrics = future.result()
            rows[dataset] = {metric: round(100 * value, 4) for metric, value in metrics.items() if value >= 0}
            for metric, value in metrics.items():
                if value >= 0:
                    values[metric].append(value)

    output = {
        "suite": "ul37",
        "dataset_count": len(rows),
        "datasets": dict(sorted(rows.items())),
        "MACRO": {metric: round(100 * np.mean(items), 4) for metric, items in values.items()},
        "valid_dataset_counts": {metric: len(items) for metric, items in values.items()},
        "evaluator": {
            "metric_source": "predictions.json",
            "training_imgsz": imgsz,
            "max_det": dict(sorted(max_dets.items())),
            "reference_size": reference_size,
            "area_scale": f"({reference_size} / max(image_width, image_height)) ** 2" if reference_size else "1.0",
            "min_bucket_objects": MIN_BUCKET_OBJECTS,
            "unit": "percent",
        },
    }
    destination = run_dir / ("val_ultrabench.json" if reference_size else "val_ultrabench_native.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(destination)
    return output


def main() -> None:
    """Evaluate one completed Ultra Benchmark run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--imgsz", type=int, default=REFERENCE_SIZE)
    parser.add_argument("--native-size-bins", action="store_true")
    args = parser.parse_args()
    evaluate_run(args.run_dir, args.imgsz, None if args.native_size_bins else REFERENCE_SIZE)


if __name__ == "__main__":
    main()
