#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Benchmark script for Stereo 3D Detection inference speed (FPS) and accuracy (AP3D) measurement.

This script measures inference speed and accuracy metrics for stereo 3D detection models
on the KITTI dataset. It supports multiple configurations for comprehensive benchmarking.

T051: Inference speed (FPS) measurement
T052: AP3D accuracy measurement

Success Criteria Validation:
- SC-001: AP3D ≥30% (ResNet-18, geometric construction only)
- SC-002: AP3D ≥32% (ResNet-18, full pipeline)
- SC-003: AP3D ≥40% (DLA-34)
- SC-004: ≥30 FPS (ResNet-18 backbone)
- SC-005: ≥20 FPS (DLA-34 backbone)

Usage:
    # Basic FPS benchmark
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml --mode fps

    # Accuracy benchmark (AP3D)
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml --mode accuracy

    # Accuracy benchmark with target validation (SC-001)
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml \\
        --mode accuracy --target-ap3d 0.30

    # Combined benchmark (FPS + AP3D)
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml --mode both

    # Batch size sweep for throughput analysis
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml --batch-sweep

    # Full success criteria validation
    python scripts/benchmark_stereo3ddet.py --model /path/to/model.pt --data /path/to/kitti.yaml \\
        --mode both --target-fps 30 --target-ap3d 0.30

Examples:
    # ResNet-18 benchmark - SC-001 validation (target: ≥30% AP3D, ≥30 FPS)
    python scripts/benchmark_stereo3ddet.py --model runs/stereo3ddet/train/weights/best.pt \\
        --data ultralytics/cfg/datasets/kitti-stereo.yaml --mode both --target-fps 30 --target-ap3d 0.30

    # DLA-34 benchmark - SC-003/SC-005 validation (target: ≥40% AP3D, ≥20 FPS)
    python scripts/benchmark_stereo3ddet.py --model runs/stereo3ddet/train_dla34/weights/best.pt \\
        --data ultralytics/cfg/datasets/kitti-stereo.yaml --mode both --target-fps 20 --target-ap3d 0.40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model_path: str
    data_yaml: str
    mode: str = "fps"  # fps, accuracy, both
    batch_size: int = 1
    num_warmup: int = 10
    num_iterations: int = 100
    num_runs: int = 3
    device: str = "auto"  # auto, cuda, cpu
    half: bool = True  # Use FP16 for faster inference
    imgsz: int = 384
    conf_threshold: float = 0.25
    max_samples: int | None = None
    target_fps: float | None = None  # Expected FPS threshold for pass/fail (SC-004/SC-005)
    target_ap3d: float | None = None  # Expected AP3D@0.7 threshold for pass/fail (SC-001/SC-002/SC-003)
    output_dir: Path = Path("runs/benchmark")


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    # Timing metrics
    total_images: int = 0
    total_time_s: float = 0.0
    fps_mean: float = 0.0
    fps_std: float = 0.0
    fps_min: float = 0.0
    fps_max: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_p99: float = 0.0

    # Accuracy metrics (optional)
    ap3d_50: float | None = None
    ap3d_70: float | None = None
    maps3d_50: float | None = None
    maps3d_70: float | None = None

    # Per-class AP3D metrics (T052: detailed accuracy breakdown)
    ap3d_50_per_class: dict[str, float] | None = None  # {class_name: ap3d_50}
    ap3d_70_per_class: dict[str, float] | None = None  # {class_name: ap3d_70}

    # Precision and Recall at different IoU thresholds
    precision_50: float | None = None
    precision_70: float | None = None
    recall_50: float | None = None
    recall_70: float | None = None

    # Detection counts
    total_predictions: int = 0
    total_ground_truth: int = 0
    true_positives_50: int = 0
    true_positives_70: int = 0

    # System info
    device: str = ""
    dtype: str = ""
    batch_size: int = 1
    model_name: str = ""
    backbone_type: str = ""

    # Pass/fail status
    target_fps: float | None = None
    target_ap3d: float | None = None
    fps_passed: bool | None = None
    ap3d_passed: bool | None = None
    passed: bool | None = None  # Overall pass (both FPS and AP3D if applicable)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "timing": {
                "total_images": self.total_images,
                "total_time_s": round(self.total_time_s, 3),
                "fps": {
                    "mean": round(self.fps_mean, 2),
                    "std": round(self.fps_std, 2),
                    "min": round(self.fps_min, 2),
                    "max": round(self.fps_max, 2),
                },
                "latency_ms": {
                    "mean": round(self.latency_ms_mean, 2),
                    "p50": round(self.latency_ms_p50, 2),
                    "p95": round(self.latency_ms_p95, 2),
                    "p99": round(self.latency_ms_p99, 2),
                },
            },
            "accuracy": {
                "ap3d_50": round(self.ap3d_50, 4) if self.ap3d_50 is not None else None,
                "ap3d_70": round(self.ap3d_70, 4) if self.ap3d_70 is not None else None,
                "maps3d_50": round(self.maps3d_50, 4) if self.maps3d_50 is not None else None,
                "maps3d_70": round(self.maps3d_70, 4) if self.maps3d_70 is not None else None,
                "per_class": {
                    "ap3d_50": {k: round(v, 4) for k, v in (self.ap3d_50_per_class or {}).items()},
                    "ap3d_70": {k: round(v, 4) for k, v in (self.ap3d_70_per_class or {}).items()},
                },
                "precision": {
                    "iou_50": round(self.precision_50, 4) if self.precision_50 is not None else None,
                    "iou_70": round(self.precision_70, 4) if self.precision_70 is not None else None,
                },
                "recall": {
                    "iou_50": round(self.recall_50, 4) if self.recall_50 is not None else None,
                    "iou_70": round(self.recall_70, 4) if self.recall_70 is not None else None,
                },
                "detection_counts": {
                    "total_predictions": self.total_predictions,
                    "total_ground_truth": self.total_ground_truth,
                    "true_positives_50": self.true_positives_50,
                    "true_positives_70": self.true_positives_70,
                },
            },
            "system": {
                "device": self.device,
                "dtype": self.dtype,
                "batch_size": self.batch_size,
                "model_name": self.model_name,
                "backbone_type": self.backbone_type,
            },
            "validation": {
                "target_fps": self.target_fps,
                "target_ap3d": self.target_ap3d,
                "fps_passed": self.fps_passed,
                "ap3d_passed": self.ap3d_passed,
                "passed": self.passed,
            },
        }


def get_device(device_arg: str) -> torch.device:
    """Determine the device to use for benchmarking.

    Args:
        device_arg: Device argument ('auto', 'cuda', 'cpu', or specific like 'cuda:0').

    Returns:
        torch.device for benchmarking.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            LOGGER.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    return torch.device(device_arg)


def detect_backbone_type(model: Any) -> str:
    """Detect the backbone type from the model.

    Args:
        model: Loaded YOLO model.

    Returns:
        Backbone type string (e.g., 'resnet18', 'dla34', 'unknown').
    """
    try:
        # Try to get backbone type from model config
        if hasattr(model, "model") and hasattr(model.model, "backbone_type"):
            return model.model.backbone_type
        if hasattr(model, "model") and hasattr(model.model, "cfg"):
            cfg = model.model.cfg
            if isinstance(cfg, dict) and "backbone" in cfg:
                return cfg["backbone"]
        # Default fallback
        return "unknown"
    except Exception:
        return "unknown"


def run_fps_benchmark(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: BenchmarkConfig,
) -> BenchmarkResults:
    """Run FPS benchmark on the model.

    Args:
        model: Loaded model for inference.
        dataloader: DataLoader providing stereo image batches.
        device: Device to run inference on.
        config: Benchmark configuration.

    Returns:
        BenchmarkResults with timing metrics.
    """
    results = BenchmarkResults()
    results.device = str(device)
    results.dtype = "float16" if config.half else "float32"
    results.batch_size = config.batch_size
    results.model_name = Path(config.model_path).stem
    results.backbone_type = detect_backbone_type(model)
    results.target_fps = config.target_fps

    # Get the actual model for inference
    if hasattr(model, "model"):
        inference_model = model.model
    else:
        inference_model = model

    inference_model.eval()
    inference_model.to(device)

    if config.half and device.type == "cuda":
        inference_model.half()

    latencies_ms: list[float] = []
    total_images = 0

    # Warmup phase
    LOGGER.info(f"Running {config.num_warmup} warmup iterations...")
    warmup_count = 0
    for batch in dataloader:
        if warmup_count >= config.num_warmup:
            break
        img = batch.get("img")
        if img is None:
            continue
        img = img.to(device)
        if config.half and device.type == "cuda":
            img = img.half()
        else:
            img = img.float()
        img = img / 255.0 if img.max() > 1.0 else img
        with torch.no_grad():
            _ = inference_model(img)
        warmup_count += 1

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark phase
    LOGGER.info(f"Running {config.num_iterations} benchmark iterations...")
    iteration_count = 0

    for batch in dataloader:
        if iteration_count >= config.num_iterations:
            break

        img = batch.get("img")
        if img is None:
            continue

        img = img.to(device)
        if config.half and device.type == "cuda":
            img = img.half()
        else:
            img = img.float()
        img = img / 255.0 if img.max() > 1.0 else img

        # Time the inference
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            _ = inference_model(img)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies_ms.append(latency_ms)
        total_images += img.shape[0]
        iteration_count += 1

    # Calculate statistics
    if latencies_ms:
        results.total_images = total_images
        results.total_time_s = sum(latencies_ms) / 1000

        # FPS calculation (images per second)
        fps_per_batch = [1000 / lat for lat in latencies_ms]
        fps_per_image = [fps * config.batch_size for fps in fps_per_batch]

        results.fps_mean = statistics.mean(fps_per_image)
        results.fps_std = statistics.stdev(fps_per_image) if len(fps_per_image) > 1 else 0.0
        results.fps_min = min(fps_per_image)
        results.fps_max = max(fps_per_image)

        # Latency statistics
        results.latency_ms_mean = statistics.mean(latencies_ms)
        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)
        results.latency_ms_p50 = sorted_latencies[int(n * 0.50)]
        results.latency_ms_p95 = sorted_latencies[int(n * 0.95)]
        results.latency_ms_p99 = sorted_latencies[int(n * 0.99)]

        # Pass/fail check for FPS (SC-004/SC-005)
        results.target_fps = config.target_fps
        if config.target_fps is not None:
            results.fps_passed = results.fps_mean >= config.target_fps
            results.passed = results.fps_passed

    return results


def run_accuracy_benchmark(
    model: Any,
    config: BenchmarkConfig,
) -> BenchmarkResults:
    """Run accuracy benchmark (AP3D) on the model.

    T052: Comprehensive AP3D accuracy measurement with per-class breakdown,
    precision/recall metrics, and success criteria validation.

    Args:
        model: Loaded YOLO model.
        config: Benchmark configuration.

    Returns:
        BenchmarkResults with detailed accuracy metrics.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
    from ultralytics.utils import YAML

    results = BenchmarkResults()
    results.model_name = Path(config.model_path).stem
    results.backbone_type = detect_backbone_type(model)
    results.target_ap3d = config.target_ap3d

    device = get_device(config.device)
    results.device = str(device)
    results.dtype = "float16" if config.half else "float32"
    results.batch_size = config.batch_size

    # Create validator
    val_kwargs = {
        "data": config.data_yaml,
        "task": "stereo3ddet",
        "imgsz": config.imgsz,
        "batch": config.batch_size,
        "device": str(device),
        "half": config.half,
        "conf": config.conf_threshold,
        "plots": False,
        "verbose": True,  # Enable verbose for detailed metrics
    }

    args = get_cfg(overrides=val_kwargs)
    validator = Stereo3DDetValidator(args=args, _callbacks=model.callbacks)

    if config.max_samples is not None:
        validator.args.max_samples = config.max_samples

    # Run validation
    LOGGER.info("Running AP3D accuracy benchmark...")
    LOGGER.info(f"  Model: {results.model_name}")
    LOGGER.info(f"  Backbone: {results.backbone_type}")
    LOGGER.info(f"  Device: {results.device}")
    LOGGER.info(f"  Precision: {results.dtype}")
    if config.target_ap3d is not None:
        LOGGER.info(f"  Target AP3D@0.7: {config.target_ap3d:.1%}")

    start_time = time.perf_counter()
    metrics = validator(model=model.model)
    end_time = time.perf_counter()

    results.total_time_s = end_time - start_time
    results.total_images = validator.seen

    # Extract comprehensive accuracy metrics
    if hasattr(validator, "metrics"):
        metrics_obj = validator.metrics

        # Mean AP3D (aggregate)
        results.maps3d_50 = getattr(metrics_obj, "maps3d_50", None)
        results.maps3d_70 = getattr(metrics_obj, "maps3d_70", None)

        # Per-class AP3D breakdown (T052)
        if hasattr(metrics_obj, "ap3d_50") and metrics_obj.ap3d_50:
            results.ap3d_50_per_class = dict(metrics_obj.ap3d_50)
            ap_values = list(metrics_obj.ap3d_50.values())
            results.ap3d_50 = statistics.mean(ap_values) if ap_values else 0.0

        if hasattr(metrics_obj, "ap3d_70") and metrics_obj.ap3d_70:
            results.ap3d_70_per_class = dict(metrics_obj.ap3d_70)
            ap_values = list(metrics_obj.ap3d_70.values())
            results.ap3d_70 = statistics.mean(ap_values) if ap_values else 0.0

        # Precision and Recall extraction (T052)
        if hasattr(metrics_obj, "precision") and metrics_obj.precision:
            precision_dict = metrics_obj.precision
            # Extract mean precision at IoU 0.5 and 0.7
            if isinstance(precision_dict, dict):
                for iou_thresh, class_precisions in precision_dict.items():
                    if isinstance(class_precisions, dict):
                        mean_prec = statistics.mean(class_precisions.values()) if class_precisions else 0.0
                        if abs(iou_thresh - 0.5) < 0.01:
                            results.precision_50 = mean_prec
                        elif abs(iou_thresh - 0.7) < 0.01:
                            results.precision_70 = mean_prec

        if hasattr(metrics_obj, "recall") and metrics_obj.recall:
            recall_dict = metrics_obj.recall
            if isinstance(recall_dict, dict):
                for iou_thresh, class_recalls in recall_dict.items():
                    if isinstance(class_recalls, dict):
                        mean_recall = statistics.mean(class_recalls.values()) if class_recalls else 0.0
                        if abs(iou_thresh - 0.5) < 0.01:
                            results.recall_50 = mean_recall
                        elif abs(iou_thresh - 0.7) < 0.01:
                            results.recall_70 = mean_recall

        # Detection counts from accumulated stats (T052)
        if hasattr(metrics_obj, "stats") and metrics_obj.stats:
            total_preds = 0
            total_gt = 0
            total_tp_50 = 0
            total_tp_70 = 0

            for stat in metrics_obj.stats:
                if "tp" in stat and len(stat["tp"]) > 0:
                    tp_array = stat["tp"]
                    total_preds += len(tp_array)
                    if tp_array.shape[1] >= 1:  # IoU 0.5
                        total_tp_50 += int(np.sum(tp_array[:, 0]))
                    if tp_array.shape[1] >= 2:  # IoU 0.7
                        total_tp_70 += int(np.sum(tp_array[:, 1]))
                if "target_cls" in stat:
                    total_gt += len(stat["target_cls"])

            results.total_predictions = total_preds
            results.total_ground_truth = total_gt
            results.true_positives_50 = total_tp_50
            results.true_positives_70 = total_tp_70

    # AP3D pass/fail validation against target (SC-001, SC-002, SC-003)
    if config.target_ap3d is not None and results.maps3d_70 is not None:
        results.ap3d_passed = results.maps3d_70 >= config.target_ap3d
        # Overall passed = AP3D passed (FPS pass status set separately)
        results.passed = results.ap3d_passed

    return results


def run_batch_sweep(
    model: Any,
    config: BenchmarkConfig,
) -> list[BenchmarkResults]:
    """Run FPS benchmark across different batch sizes.

    Args:
        model: Loaded YOLO model.
        config: Benchmark configuration.

    Returns:
        List of BenchmarkResults for each batch size.
    """
    from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
    from ultralytics.utils import YAML

    device = get_device(config.device)
    batch_sizes = [1, 2, 4, 8, 16]
    results_list = []

    # Load dataset config
    data_cfg = YAML.load(config.data_yaml)
    dataset_root = Path(data_cfg.get("path", ".")).resolve()
    val_split = data_cfg.get("val_split", "val")

    # Create dataset
    dataset = Stereo3DDetAdapterDataset(
        root=str(dataset_root),
        split=val_split,
        imgsz=config.imgsz,
        max_samples=config.max_samples,
    )

    for batch_size in batch_sizes:
        LOGGER.info(f"Running batch size {batch_size}...")

        try:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=device.type == "cuda",
                collate_fn=dataset.collate_fn,
                drop_last=True,
            )

            sweep_config = BenchmarkConfig(
                model_path=config.model_path,
                data_yaml=config.data_yaml,
                batch_size=batch_size,
                num_warmup=config.num_warmup,
                num_iterations=min(config.num_iterations, len(dataloader)),
                device=config.device,
                half=config.half,
                imgsz=config.imgsz,
            )

            result = run_fps_benchmark(model, dataloader, device, sweep_config)
            results_list.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                LOGGER.warning(f"OOM at batch size {batch_size}, stopping sweep")
                break
            raise

    return results_list


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results in a formatted table.

    T052: Enhanced output with per-class metrics and validation status.

    Args:
        results: BenchmarkResults to print.
    """
    print("\n" + "=" * 70)
    print("STEREO 3D DETECTION BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Model:':<20} {results.model_name}")
    print(f"{'Backbone:':<20} {results.backbone_type}")
    print(f"{'Device:':<20} {results.device}")
    print(f"{'Precision:':<20} {results.dtype}")
    print(f"{'Batch Size:':<20} {results.batch_size}")

    if results.fps_mean > 0:
        print("\n" + "-" * 50)
        print("TIMING METRICS")
        print("-" * 50)
        print(f"{'Total Images:':<25} {results.total_images}")
        print(f"{'Total Time:':<25} {results.total_time_s:.2f}s")
        print(f"{'FPS (mean ± std):':<25} {results.fps_mean:.2f} ± {results.fps_std:.2f}")
        print(f"{'FPS (min/max):':<25} {results.fps_min:.2f} / {results.fps_max:.2f}")
        print(f"{'Latency (mean):':<25} {results.latency_ms_mean:.2f} ms")
        print(f"{'Latency (P50):':<25} {results.latency_ms_p50:.2f} ms")
        print(f"{'Latency (P95):':<25} {results.latency_ms_p95:.2f} ms")
        print(f"{'Latency (P99):':<25} {results.latency_ms_p99:.2f} ms")

    if results.ap3d_50 is not None or results.maps3d_50 is not None:
        print("\n" + "-" * 50)
        print("ACCURACY METRICS (AP3D)")
        print("-" * 50)

        # Aggregate metrics
        if results.maps3d_50 is not None:
            print(f"{'mAP3D@0.5 (mean):':<25} {results.maps3d_50:.4f} ({results.maps3d_50*100:.2f}%)")
        if results.maps3d_70 is not None:
            print(f"{'mAP3D@0.7 (mean):':<25} {results.maps3d_70:.4f} ({results.maps3d_70*100:.2f}%)")

        # Per-class breakdown (T052)
        if results.ap3d_70_per_class:
            print(f"\n{'Per-Class AP3D@0.7:'}")
            for class_name, ap in sorted(results.ap3d_70_per_class.items()):
                print(f"  {class_name:<18} {ap:.4f} ({ap*100:.2f}%)")

        if results.ap3d_50_per_class:
            print(f"\n{'Per-Class AP3D@0.5:'}")
            for class_name, ap in sorted(results.ap3d_50_per_class.items()):
                print(f"  {class_name:<18} {ap:.4f} ({ap*100:.2f}%)")

        # Precision and Recall (T052)
        print("\n" + "-" * 50)
        print("PRECISION & RECALL")
        print("-" * 50)
        if results.precision_70 is not None:
            print(f"{'Precision@0.7:':<25} {results.precision_70:.4f}")
        if results.recall_70 is not None:
            print(f"{'Recall@0.7:':<25} {results.recall_70:.4f}")
        if results.precision_50 is not None:
            print(f"{'Precision@0.5:':<25} {results.precision_50:.4f}")
        if results.recall_50 is not None:
            print(f"{'Recall@0.5:':<25} {results.recall_50:.4f}")

        # Detection counts (T052)
        if results.total_predictions > 0 or results.total_ground_truth > 0:
            print("\n" + "-" * 50)
            print("DETECTION COUNTS")
            print("-" * 50)
            print(f"{'Total Predictions:':<25} {results.total_predictions}")
            print(f"{'Total Ground Truth:':<25} {results.total_ground_truth}")
            print(f"{'True Positives@0.7:':<25} {results.true_positives_70}")
            print(f"{'True Positives@0.5:':<25} {results.true_positives_50}")

    # Success Criteria Validation (T052)
    if results.target_fps is not None or results.target_ap3d is not None:
        print("\n" + "-" * 50)
        print("SUCCESS CRITERIA VALIDATION")
        print("-" * 50)

        # FPS validation (SC-004, SC-005)
        if results.target_fps is not None:
            fps_status = "✓ PASS" if results.fps_passed else "✗ FAIL"
            print(f"{'FPS Target:':<25} {results.target_fps:.1f} (SC-004/SC-005)")
            print(f"{'FPS Achieved:':<25} {results.fps_mean:.2f}")
            print(f"{'FPS Status:':<25} {fps_status}")

        # AP3D validation (SC-001, SC-002, SC-003)
        if results.target_ap3d is not None:
            ap3d_status = "✓ PASS" if results.ap3d_passed else "✗ FAIL"
            print(f"\n{'AP3D@0.7 Target:':<25} {results.target_ap3d:.1%} (SC-001/SC-002/SC-003)")
            achieved_ap3d = results.maps3d_70 if results.maps3d_70 is not None else 0.0
            print(f"{'AP3D@0.7 Achieved:':<25} {achieved_ap3d:.4f} ({achieved_ap3d*100:.2f}%)")
            print(f"{'AP3D Status:':<25} {ap3d_status}")

        # Overall status
        if results.passed is not None:
            overall_status = "✓ PASS" if results.passed else "✗ FAIL"
            print(f"\n{'OVERALL:':<25} {overall_status}")

    print("\n" + "=" * 70 + "\n")


def print_batch_sweep_results(results_list: list[BenchmarkResults]) -> None:
    """Print batch sweep results in a formatted table.

    Args:
        results_list: List of BenchmarkResults from batch sweep.
    """
    print("\n" + "=" * 70)
    print("BATCH SIZE SWEEP RESULTS")
    print("=" * 70)
    print(f"\n{'Batch':<8} {'FPS':<12} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-" * 50)

    for r in results_list:
        throughput = r.fps_mean * r.batch_size if r.fps_mean > 0 else 0
        print(f"{r.batch_size:<8} {r.fps_mean:<12.2f} {r.latency_ms_mean:<15.2f} {throughput:<15.2f}")

    print("=" * 70 + "\n")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark Stereo 3D Detection inference speed and accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights file (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fps",
        choices=["fps", "accuracy", "both"],
        help="Benchmark mode: fps, accuracy, or both",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--batch-sweep",
        action="store_true",
        help="Run batch size sweep to find optimal throughput",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs for averaging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 for inference (faster on GPU)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 (use FP32)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=384,
        help="Input image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples for benchmarking (default: all)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target FPS for pass/fail validation (SC-004: 30 for ResNet-18, SC-005: 20 for DLA-34)",
    )
    parser.add_argument(
        "--target-ap3d",
        type=float,
        default=None,
        help="Target AP3D@0.7 for pass/fail validation (SC-001: 0.30, SC-002: 0.32, SC-003: 0.40)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmark",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        LOGGER.error(f"Model file not found: {args.model}")
        return 1
    if not Path(args.data).exists():
        LOGGER.error(f"Dataset YAML not found: {args.data}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        data_yaml=args.data,
        mode=args.mode,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        num_runs=args.runs,
        device=args.device,
        half=args.half and not args.no_half,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        max_samples=args.max_samples,
        target_fps=args.target_fps,
        target_ap3d=args.target_ap3d,
        output_dir=output_dir,
    )

    # Load model
    LOGGER.info(f"Loading model from {args.model}...")
    model = YOLO(args.model, task="stereo3ddet")

    # Run appropriate benchmark
    try:
        if args.batch_sweep:
            # Batch size sweep
            results_list = run_batch_sweep(model, config)
            print_batch_sweep_results(results_list)

            # Save results
            sweep_data = [r.to_dict() for r in results_list]
            results_path = output_dir / "batch_sweep_results.json"
            with open(results_path, "w") as f:
                json.dump(sweep_data, f, indent=2)
            LOGGER.info(f"Saved batch sweep results to {results_path}")

        elif args.mode == "fps" or args.mode == "both":
            # FPS benchmark
            from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
            from ultralytics.utils import YAML

            device = get_device(config.device)

            # Load dataset
            data_cfg = YAML.load(config.data_yaml)
            dataset_root = Path(data_cfg.get("path", ".")).resolve()
            val_split = data_cfg.get("val_split", "val")

            dataset = Stereo3DDetAdapterDataset(
                root=str(dataset_root),
                split=val_split,
                imgsz=config.imgsz,
                max_samples=config.max_samples,
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=device.type == "cuda",
                collate_fn=dataset.collate_fn,
                drop_last=False,
            )

            # Run multiple times and average
            all_results = []
            for run_idx in range(config.num_runs):
                LOGGER.info(f"FPS benchmark run {run_idx + 1}/{config.num_runs}...")
                result = run_fps_benchmark(model, dataloader, device, config)
                all_results.append(result)

            # Aggregate results
            if all_results:
                final_result = all_results[0]
                if len(all_results) > 1:
                    fps_values = [r.fps_mean for r in all_results]
                    final_result.fps_mean = statistics.mean(fps_values)
                    final_result.fps_std = statistics.stdev(fps_values)
                    final_result.fps_min = min(r.fps_min for r in all_results)
                    final_result.fps_max = max(r.fps_max for r in all_results)

                    latency_means = [r.latency_ms_mean for r in all_results]
                    final_result.latency_ms_mean = statistics.mean(latency_means)

                # Update FPS pass/fail with aggregated results (SC-004/SC-005)
                final_result.target_fps = config.target_fps
                if config.target_fps is not None:
                    final_result.fps_passed = final_result.fps_mean >= config.target_fps

                # Run accuracy if mode is "both" (T052)
                if args.mode == "both":
                    LOGGER.info("Running accuracy benchmark...")
                    accuracy_result = run_accuracy_benchmark(model, config)
                    # Copy all accuracy metrics to final result
                    final_result.ap3d_50 = accuracy_result.ap3d_50
                    final_result.ap3d_70 = accuracy_result.ap3d_70
                    final_result.maps3d_50 = accuracy_result.maps3d_50
                    final_result.maps3d_70 = accuracy_result.maps3d_70
                    final_result.ap3d_50_per_class = accuracy_result.ap3d_50_per_class
                    final_result.ap3d_70_per_class = accuracy_result.ap3d_70_per_class
                    final_result.precision_50 = accuracy_result.precision_50
                    final_result.precision_70 = accuracy_result.precision_70
                    final_result.recall_50 = accuracy_result.recall_50
                    final_result.recall_70 = accuracy_result.recall_70
                    final_result.total_predictions = accuracy_result.total_predictions
                    final_result.total_ground_truth = accuracy_result.total_ground_truth
                    final_result.true_positives_50 = accuracy_result.true_positives_50
                    final_result.true_positives_70 = accuracy_result.true_positives_70

                    # AP3D pass/fail (SC-001/SC-002/SC-003)
                    final_result.target_ap3d = config.target_ap3d
                    if config.target_ap3d is not None and final_result.maps3d_70 is not None:
                        final_result.ap3d_passed = final_result.maps3d_70 >= config.target_ap3d

                # Overall pass: both FPS and AP3D must pass (if both targets specified)
                if config.target_fps is not None and config.target_ap3d is not None:
                    final_result.passed = (final_result.fps_passed or False) and (final_result.ap3d_passed or False)
                elif config.target_fps is not None:
                    final_result.passed = final_result.fps_passed
                elif config.target_ap3d is not None:
                    final_result.passed = final_result.ap3d_passed

                # Print and save results
                if args.json:
                    print(json.dumps(final_result.to_dict(), indent=2))
                else:
                    print_results(final_result)

                results_path = output_dir / "benchmark_results.json"
                with open(results_path, "w") as f:
                    json.dump(final_result.to_dict(), f, indent=2)
                LOGGER.info(f"Saved benchmark results to {results_path}")

        elif args.mode == "accuracy":
            # Accuracy-only benchmark (T052)
            result = run_accuracy_benchmark(model, config)

            # Set overall pass status based on AP3D target
            if config.target_ap3d is not None:
                result.passed = result.ap3d_passed

            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print_results(result)

            results_path = output_dir / "accuracy_results.json"
            with open(results_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            LOGGER.info(f"Saved accuracy results to {results_path}")

        return 0

    except Exception as e:
        LOGGER.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

