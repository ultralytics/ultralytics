#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Command-line script for profiling stereo 3D detection validation performance.

This script runs validation with profiling enabled on a subset of the KITTI dataset,
collects timing data, and generates bottleneck analysis reports.

Usage:
    python scripts/profile_validation.py --data /path/to/dataset.yaml --model /path/to/model.pt
    python scripts/profile_validation.py --data /path/to/dataset.yaml --model /path/to/model.pt --subset-size 150
    python scripts/profile_validation.py --data /path/to/dataset.yaml --model /path/to/model.pt --runs 5
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.profiling import generate_profiling_report, reset_profiling


def select_dataset_subset(
    dataset_root: Path,
    split: str,
    subset_size: int = 150,
    stratify_by_object_count: bool = True,
) -> List[str]:
    """Select a subset of images from the dataset with stratified sampling.
    
    This function implements stratified sampling by object count to ensure
    the subset is representative of the full dataset.
    
    Args:
        dataset_root: Root directory of the KITTI dataset.
        split: Dataset split ('train' or 'val').
        subset_size: Number of images to select (default: 150).
        stratify_by_object_count: If True, use stratified sampling by object count.
    
    Returns:
        List of image IDs (without extension) selected for profiling.
    """
    from ultralytics.data.kitti_stereo import KITTIStereoDataset
    
    # Load full dataset to get image IDs and object counts
    try:
        dataset = KITTIStereoDataset(root=dataset_root, split=split, filter_classes=True)
        all_image_ids = dataset.image_ids
    except Exception as e:
        LOGGER.error(f"Failed to load dataset: {e}")
        raise
    
    if len(all_image_ids) <= subset_size:
        LOGGER.info(f"Dataset has {len(all_image_ids)} images, using all for profiling")
        return all_image_ids
    
    if not stratify_by_object_count:
        # Simple random sampling
        return random.sample(all_image_ids, subset_size)
    
    # Stratified sampling by object count
    # Count objects per image
    object_counts: Dict[int, List[str]] = {}
    for image_id in all_image_ids:
        try:
            labels = dataset._parse_labels(dataset.label_dir / f"{image_id}.txt")
            count = len(labels)
            # Group into bins: 0, 1-5, 6-10, 11+
            if count == 0:
                bin_key = 0
            elif count <= 5:
                bin_key = 1
            elif count <= 10:
                bin_key = 2
            else:
                bin_key = 3
            
            if bin_key not in object_counts:
                object_counts[bin_key] = []
            object_counts[bin_key].append(image_id)
        except Exception as e:
            LOGGER.warning(f"Failed to count objects for {image_id}: {e}")
            # Add to a default bin
            if 0 not in object_counts:
                object_counts[0] = []
            object_counts[0].append(image_id)
    
    # Sample proportionally from each bin
    selected: List[str] = []
    total_images = len(all_image_ids)
    
    for bin_key, image_ids in object_counts.items():
        proportion = len(image_ids) / total_images
        bin_size = max(1, int(subset_size * proportion))
        bin_size = min(bin_size, len(image_ids))  # Don't exceed available images
        
        sampled = random.sample(image_ids, bin_size)
        selected.extend(sampled)
    
    # If we didn't get enough, randomly sample from remaining
    if len(selected) < subset_size:
        remaining = [img_id for img_id in all_image_ids if img_id not in selected]
        needed = subset_size - len(selected)
        if remaining:
            selected.extend(random.sample(remaining, min(needed, len(remaining))))
    
    # Shuffle to avoid bias
    random.shuffle(selected)
    
    LOGGER.info(
        f"Selected {len(selected)} images for profiling "
        f"(stratified by object count: {len(object_counts)} bins)"
    )
    
    return selected[:subset_size]


def create_subset_dataset_yaml(
    original_yaml: Path,
    subset_image_ids: List[str],
    output_yaml: Path,
) -> Path:
    """Create a temporary dataset YAML file for the subset.
    
    Args:
        original_yaml: Path to original dataset YAML.
        subset_image_ids: List of image IDs to include in subset.
        output_yaml: Path to save subset YAML.
    
    Returns:
        Path to created subset YAML file.
    """
    from ultralytics.utils import YAML
    
    # Load original YAML
    original_data = YAML.load(str(original_yaml))
    
    # Create subset dataset structure
    # Note: This creates a temporary structure - in practice, you might want to
    # create actual subset directories, but for profiling we can use the full dataset
    # and filter in the dataset loader
    
    # For now, we'll just use the original YAML and filter in the dataset
    # A more sophisticated implementation would create actual subset directories
    subset_data = original_data.copy()
    
    # Save subset YAML (for reference, actual filtering happens in dataset loader)
    YAML.save(str(output_yaml), subset_data)
    
    return output_yaml


def run_baseline_profiling(
    model_path: str,
    data_yaml: str,
    subset_size: int = 150,
    max_samples: int | None = None,
    num_runs: int = 3,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run baseline profiling with multiple runs and collect statistics.
    
    Args:
        model_path: Path to model weights file.
        data_yaml: Path to dataset YAML file.
        subset_size: Number of images to use for profiling.
        num_runs: Number of validation runs to perform (default: 3).
        output_dir: Directory to save profiling reports.
    
    Returns:
        Dictionary containing profiling statistics across all runs.
    """
    if output_dir is None:
        output_dir = Path("specs/002-stereo3d-val-pred/profiling")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset YAML to get root path
    from ultralytics.utils import YAML
    from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
    
    data_cfg = YAML.load(data_yaml)
    dataset_root = Path(data_cfg.get("path", ".")).resolve()
    val_split = data_cfg.get("val_split", "val")
    
    # Fix class names: Map from original KITTI IDs (0,3,5) to paper IDs (0,1,2)
    # This is required because the model expects 3 classes with indices 0-2
    paper_names = get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    original_names = data_cfg.get("names", {})
    
    # If dataset has original KITTI class IDs (0,3,5), remap to paper IDs (0,1,2)
    if original_names and max(original_names.keys()) >= 3:
        # Create a temporary YAML with remapped class IDs
        import tempfile
        temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_yaml_path = Path(temp_yaml.name)
        
        # Update data_cfg with paper class names
        data_cfg["names"] = paper_names
        data_cfg["nc"] = 3
        
        # Save to temporary file
        YAML.save(str(temp_yaml_path), data_cfg)
        temp_yaml.close()
        
        # Use temporary YAML for validation
        data_yaml = str(temp_yaml_path)
        LOGGER.info(f"Remapped class IDs from {list(original_names.keys())} to {list(paper_names.keys())}")
    
    # Use max_samples parameter if provided, otherwise fall back to subset_size for backward compatibility (T197)
    effective_max_samples = max_samples if max_samples is not None else subset_size
    
    if effective_max_samples is not None:
        LOGGER.info(f"Using max_samples={effective_max_samples} to limit dataset for profiling...")
    else:
        LOGGER.info("Using full dataset for profiling (no max_samples specified)")
    
    # Run validation multiple times and collect statistics (T178)
    all_reports = []
    all_total_times = []
    
    LOGGER.info(f"Running baseline profiling with {num_runs} runs...")
    
    for run_idx in range(num_runs):
        LOGGER.info(f"Run {run_idx + 1}/{num_runs}...")
        
        # Reset profiling collector
        reset_profiling()
        
        # Load model with task specified to ensure correct validator
        model = YOLO(model_path)
        
        # Ensure model task is set correctly
        if hasattr(model, 'task') and model.task != "stereo3ddet":
            model.task = "stereo3ddet"
        
        # Run validation with max_samples if specified (T197)
        # Note: max_samples is not a standard YOLO arg, so we need to set it on the validator after creation
        val_kwargs = {
            "data": data_yaml,
            "task": "stereo3ddet",  # CRITICAL: Specify task to use Stereo3DDetValidator
            "split": val_split,
            "imgsz": 384,
            "batch": 8,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "plots": False,
            "verbose": False,
        }
        
        try:
            # Create validator directly to set max_samples without validation errors
            from ultralytics.cfg import get_cfg
            from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
            from ultralytics.utils import YAML
            from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
            import tempfile
            
            # Create temporary YAML with remapped class IDs BEFORE creating validator
            # This ensures AutoBackend reads the correct names (0,1,2) instead of (0,3,5)
            data_cfg = YAML.load(data_yaml)
            original_names = data_cfg.get("names", {})
            paper_names = get_paper_class_names()
            
            # Always create temp YAML with paper class names to avoid AutoBackend validation errors
            temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            temp_yaml_path = Path(temp_yaml.name)
            data_cfg["names"] = paper_names
            data_cfg["nc"] = 3
            YAML.save(str(temp_yaml_path), data_cfg)
            temp_yaml.close()
            
            # Update val_kwargs to use temp YAML
            val_kwargs["data"] = str(temp_yaml_path)
            
            # Get validated args (without max_samples)
            args = get_cfg(overrides=val_kwargs)
            
            # Create validator
            validator = Stereo3DDetValidator(args=args, _callbacks=model.callbacks)
            
            # Set max_samples on validator args (bypasses validation)
            if effective_max_samples is not None:
                validator.args.max_samples = effective_max_samples
            
            # Run validation
            results = validator(model=model.model)
            
            # Generate profiling report for this run
            report = generate_profiling_report(collector=None, output_dir=None)
            all_reports.append(report)
            all_total_times.append(report.get("total_time", 0.0))
            
            LOGGER.info(f"Run {run_idx + 1} completed: {report.get('total_time', 0.0):.2f}s")
            
        except Exception as e:
            import traceback
            LOGGER.error(f"Error in run {run_idx + 1}: {e}")
            LOGGER.error(f"Traceback:\n{traceback.format_exc()}")
            continue
    
    if not all_reports:
        LOGGER.error("No successful profiling runs completed")
        return {}
    
    # Aggregate statistics across runs (T178)
    aggregated_report = {
        "num_runs": len(all_reports),
        "total_time": {
            "mean": statistics.mean(all_total_times),
            "std": statistics.stdev(all_total_times) if len(all_total_times) > 1 else 0.0,
            "min": min(all_total_times),
            "max": max(all_total_times),
        },
        "top_bottlenecks": _aggregate_bottlenecks(all_reports),
        "breakdown_by_category": _aggregate_categories(all_reports),
    }
    
    # Generate final report (T180)
    final_report_path = output_dir / "baseline_report.json"
    with open(final_report_path, "w") as f:
        json.dump(aggregated_report, f, indent=2)
    LOGGER.info(f"Saved aggregated baseline report to {final_report_path}")
    
    # Generate bottleneck analysis markdown (T180)
    bottleneck_analysis_path = output_dir / "bottleneck_analysis.md"
    _write_bottleneck_analysis(bottleneck_analysis_path, aggregated_report)
    LOGGER.info(f"Saved bottleneck analysis to {bottleneck_analysis_path}")
    
    return aggregated_report


def _aggregate_bottlenecks(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate bottleneck data across multiple runs.
    
    Args:
        reports: List of profiling reports from individual runs.
    
    Returns:
        Aggregated list of top bottlenecks with statistics.
    """
    # Collect all bottlenecks
    bottleneck_data: Dict[str, List[float]] = {}
    bottleneck_calls: Dict[str, List[int]] = {}
    
    for report in reports:
        for bottleneck in report.get("top_bottlenecks", []):
            name = bottleneck["function_name"]
            if name not in bottleneck_data:
                bottleneck_data[name] = []
                bottleneck_calls[name] = []
            bottleneck_data[name].append(bottleneck["total_time"])
            bottleneck_calls[name].append(bottleneck["call_count"])
    
    # Calculate statistics
    aggregated = []
    # Calculate total time across all functions for percentage calculation
    all_times = [t for times in bottleneck_data.values() for t in times]
    total_sum = sum(all_times) if all_times else 1.0
    
    for name, times in sorted(bottleneck_data.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        aggregated.append({
            "function_name": name,
            "total_time": {
                "mean": statistics.mean(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min": min(times),
                "max": max(times),
            },
            "call_count": {
                "mean": int(statistics.mean(bottleneck_calls[name])),
                "min": min(bottleneck_calls[name]),
                "max": max(bottleneck_calls[name]),
            },
            "percentage_of_total": (statistics.mean(times) / total_sum * 100.0) if total_sum > 0 else 0.0,
        })
    
    return aggregated[:5]  # Top 5


def _aggregate_categories(reports: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate category breakdown across multiple runs.
    
    Args:
        reports: List of profiling reports from individual runs.
    
    Returns:
        Aggregated category breakdown with statistics.
    """
    category_data: Dict[str, List[float]] = {}
    
    for report in reports:
        for category, percentage in report.get("breakdown_by_category", {}).items():
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(percentage)
    
    aggregated = {}
    for category, percentages in category_data.items():
        aggregated[category] = {
            "mean": statistics.mean(percentages),
            "std": statistics.stdev(percentages) if len(percentages) > 1 else 0.0,
            "min": min(percentages),
            "max": max(percentages),
        }
    
    return aggregated


def _write_bottleneck_analysis(path: Path, report: Dict[str, Any]):
    """Write bottleneck analysis report in Markdown format.
    
    Args:
        path: Path to save the analysis report.
        report: Aggregated profiling report.
    """
    with open(path, "w") as f:
        f.write("# Performance Profiling Bottleneck Analysis\n\n")
        
        # Summary statistics
        total_time = report.get("total_time", {})
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Number of Runs**: {report.get('num_runs', 0)}\n")
        f.write(f"- **Total Validation Time**: {total_time.get('mean', 0.0):.2f}s ± {total_time.get('std', 0.0):.2f}s\n")
        f.write(f"- **Time Range**: {total_time.get('min', 0.0):.2f}s - {total_time.get('max', 0.0):.2f}s\n\n")
        
        # Top bottlenecks
        f.write("## Top 5 Bottlenecks\n\n")
        f.write("| Function | Mean Time (s) | Std Dev | Min | Max | Mean Calls | % of Total |\n")
        f.write("|----------|---------------|---------|-----|-----|------------|------------|\n")
        
        for bottleneck in report.get("top_bottlenecks", []):
            time_stats = bottleneck.get("total_time", {})
            call_stats = bottleneck.get("call_count", {})
            f.write(
                f"| {bottleneck['function_name']} | "
                f"{time_stats.get('mean', 0.0):.4f} | "
                f"{time_stats.get('std', 0.0):.4f} | "
                f"{time_stats.get('min', 0.0):.4f} | "
                f"{time_stats.get('max', 0.0):.4f} | "
                f"{call_stats.get('mean', 0)} | "
                f"{bottleneck.get('percentage_of_total', 0.0):.2f}% |\n"
            )
        
        # Category breakdown
        f.write("\n## Time Breakdown by Category\n\n")
        f.write("| Category | Mean % | Std Dev | Min % | Max % |\n")
        f.write("|----------|--------|---------|-------|-------|\n")
        
        for category, stats in sorted(
            report.get("breakdown_by_category", {}).items(),
            key=lambda x: x[1].get("mean", 0.0),
            reverse=True,
        ):
            f.write(
                f"| {category} | "
                f"{stats.get('mean', 0.0):.2f}% | "
                f"{stats.get('std', 0.0):.2f}% | "
                f"{stats.get('min', 0.0):.2f}% | "
                f"{stats.get('max', 0.0):.2f}% |\n"
            )
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on the profiling data, consider optimizing the following:\n\n")
        
        # Generate recommendations from top bottlenecks
        for bottleneck in report.get("top_bottlenecks", [])[:3]:
            name = bottleneck["function_name"]
            percentage = bottleneck.get("percentage_of_total", 0.0)
            if percentage > 20.0:
                f.write(f"- **{name}** accounts for {percentage:.1f}% of total time - primary optimization target\n")
        
        f.write("\n---\n")
        f.write("*Report generated by profile_validation.py*\n")


def main():
    """Main entry point for profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile stereo 3D detection validation performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights file (.pt)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=150,
        help="Number of images to use for profiling (deprecated, use --max-samples instead) (default: 150)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load from dataset (default: None, loads all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of validation runs to perform (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="specs/002-stereo3d-val-pred/profiling",
        help="Directory to save profiling reports",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data).exists():
        LOGGER.error(f"Dataset YAML not found: {args.data}")
        return 1
    
    if not Path(args.model).exists():
        LOGGER.error(f"Model file not found: {args.model}")
        return 1
    
    # Run profiling
    try:
        report = run_baseline_profiling(
            model_path=args.model,
            data_yaml=args.data,
            subset_size=args.subset_size,
            max_samples=getattr(args, "max_samples", None),
            num_runs=args.runs,
            output_dir=Path(args.output_dir),
        )
        
        LOGGER.info("Profiling completed successfully!")
        LOGGER.info(f"Reports saved to: {args.output_dir}")
        
        return 0
    except Exception as e:
        LOGGER.error(f"Profiling failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

