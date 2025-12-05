# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Performance profiling utilities for stereo 3D detection validation."""

import cProfile
import functools
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ultralytics.utils import LOGGER


class ProfilingCollector:
    """Collects and aggregates profiling results from multiple function calls.
    
    This class maintains timing data for profiled functions and sections,
    aggregating results across multiple calls and calculating statistics.
    
    Attributes:
        results: Dictionary mapping function/section names to profiling data
    """

    def __init__(self):
        """Initialize the profiling collector."""
        self.results: Dict[str, Dict[str, Any]] = {}
        self._total_time: float = 0.0

    def record(self, name: str, elapsed_time: float, call_count: int = 1, **kwargs):
        """Record timing data for a function or section.
        
        Args:
            name: Name of the profiled function or section
            elapsed_time: Time elapsed in seconds
            call_count: Number of times the function was called (default: 1)
            **kwargs: Additional metadata to store (e.g., line_timings)
        """
        if name not in self.results:
            self.results[name] = {
                "total_time": 0.0,
                "call_count": 0,
                "avg_time_per_call": 0.0,
                "percentage_of_total": 0.0,
            }
            # Store any additional metadata
            for key, value in kwargs.items():
                self.results[name][key] = value

        # Aggregate timing data
        self.results[name]["total_time"] += elapsed_time
        self.results[name]["call_count"] += call_count
        self.results[name]["avg_time_per_call"] = (
            self.results[name]["total_time"] / self.results[name]["call_count"]
        )

        # Update total time
        self._total_time = sum(r["total_time"] for r in self.results.values())

        # Recalculate percentages for all functions
        if self._total_time > 0:
            for func_name in self.results:
                self.results[func_name]["percentage_of_total"] = (
                    self.results[func_name]["total_time"] / self._total_time * 100.0
                )

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated profiling results.
        
        Returns:
            Dictionary mapping function/section names to profiling data
        """
        return self.results.copy()

    def reset(self):
        """Reset all collected profiling data."""
        self.results.clear()
        self._total_time = 0.0

    def get_total_time(self) -> float:
        """Get total time across all profiled functions.
        
        Returns:
            Total time in seconds
        """
        return self._total_time


# Global profiling collector instance
_global_collector = ProfilingCollector()


def get_profiling_results() -> Dict[str, Dict[str, Any]]:
    """Get aggregated profiling results from the global collector.
    
    Returns:
        Dictionary mapping function/section names to profiling data
    """
    return _global_collector.get_results()


def reset_profiling():
    """Reset the global profiling collector."""
    _global_collector.reset()


def profile_function(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    collector: Optional[ProfilingCollector] = None,
):
    """Decorator for profiling function execution time.
    
    This decorator measures the execution time of a function and records
    it in the profiling collector. Uses time.perf_counter() for accurate timing.
    
    Can be used with or without parentheses:
        @profile_function
        def my_func():
            ...
        
        @profile_function(name="custom_name")
        def my_func():
            ...
    
    Args:
        func: Function to profile (when used without parentheses)
        name: Optional name for the profiled function (defaults to function name)
        collector: Optional ProfilingCollector instance (defaults to global collector)
    
    Example:
        @profile_function(name="compute_3d_iou_batch")
        def compute_3d_iou_batch(...):
            ...
    """
    def decorator(f: Callable) -> Callable:
        func_name = name or f.__name__
        profiler = collector or _global_collector

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Use time.perf_counter() for high-resolution timing
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.perf_counter() - start_time
                profiler.record(func_name, elapsed_time, call_count=1)

        return wrapper
    
    # Support both @profile_function and @profile_function(...)
    if func is None:
        # Called with parentheses: @profile_function(...)
        return decorator
    else:
        # Called without parentheses: @profile_function
        return decorator(func)


@contextmanager
def profile_section(name: str, collector: Optional[ProfilingCollector] = None):
    """Context manager for profiling code block execution time.
    
    This context manager measures the execution time of a code block
    and records it in the profiling collector.
    
    Args:
        name: Name for the profiled section
        collector: Optional ProfilingCollector instance (defaults to global collector)
    
    Example:
        with profile_section("update_metrics"):
            # Code to profile
            ...
    """
    profiler = collector or _global_collector
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_time = time.perf_counter() - start_time
        profiler.record(name, elapsed_time, call_count=1)


def generate_profiling_report(
    output_dir: Optional[Path] = None,
    collector: Optional[ProfilingCollector] = None,
) -> Dict[str, Any]:
    """Generate profiling report in JSON and Markdown formats.
    
    This function generates a comprehensive profiling report with:
    - Top bottlenecks sorted by total time
    - Time breakdown by category
    - Statistical summary
    
    Args:
        output_dir: Optional directory to save reports (defaults to current directory)
        collector: Optional ProfilingCollector instance (defaults to global collector)
    
    Returns:
        Dictionary containing report data
    """
    profiler = collector or _global_collector
    results = profiler.get_results()
    total_time = profiler.get_total_time()

    if not results:
        LOGGER.warning("No profiling data available. Run validation with profiling enabled first.")
        return {
            "total_time": 0.0,
            "top_bottlenecks": [],
            "breakdown_by_category": {},
            "summary": "No profiling data available",
        }

    # Sort by total time (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["total_time"],
        reverse=True,
    )

    # Extract top 5 bottlenecks
    top_bottlenecks = [
        {
            "function_name": name,
            "total_time": data["total_time"],
            "call_count": data["call_count"],
            "avg_time_per_call": data["avg_time_per_call"],
            "percentage_of_total": data["percentage_of_total"],
        }
        for name, data in sorted_results[:5]
    ]

    # Categorize by function name patterns
    breakdown_by_category: Dict[str, float] = {}
    for name, data in results.items():
        category = _categorize_function(name)
        if category not in breakdown_by_category:
            breakdown_by_category[category] = 0.0
        breakdown_by_category[category] += data["total_time"]

    # Convert category times to percentages
    if total_time > 0:
        breakdown_by_category = {
            cat: (time / total_time * 100.0)
            for cat, time in breakdown_by_category.items()
        }

    # Generate recommendations
    recommendations = _generate_recommendations(top_bottlenecks, breakdown_by_category)

    report = {
        "total_time": total_time,
        "top_bottlenecks": top_bottlenecks,
        "breakdown_by_category": breakdown_by_category,
        "recommendations": recommendations,
        "all_results": results,
    }

    # Save reports if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_dir / "baseline_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        LOGGER.info(f"Saved JSON profiling report to {json_path}")

        # Save Markdown report
        md_path = output_dir / "bottleneck_analysis.md"
        _write_markdown_report(md_path, report)
        LOGGER.info(f"Saved Markdown profiling report to {md_path}")

    return report


def _categorize_function(name: str) -> str:
    """Categorize a function name into a category for breakdown analysis.
    
    Args:
        name: Function or section name
    
    Returns:
        Category name (e.g., "IoU", "Decode", "Matching", etc.)
    """
    name_lower = name.lower()
    
    if "iou" in name_lower or "intersection" in name_lower:
        return "IoU Computation"
    elif "decode" in name_lower or "postprocess" in name_lower:
        return "Decode/Postprocess"
    elif "match" in name_lower or "assign" in name_lower:
        return "Matching"
    elif "update" in name_lower or "metrics" in name_lower:
        return "Metrics Update"
    elif "forward" in name_lower or "inference" in name_lower:
        return "Model Inference"
    elif "preprocess" in name_lower or "transform" in name_lower:
        return "Preprocessing"
    else:
        return "Other"


def _generate_recommendations(
    top_bottlenecks: list[Dict[str, Any]],
    breakdown_by_category: Dict[str, float],
) -> list[str]:
    """Generate optimization recommendations based on profiling data.
    
    Args:
        top_bottlenecks: List of top bottleneck functions
        breakdown_by_category: Time breakdown by category
    
    Returns:
        List of recommendation strings
    """
    recommendations = []

    if not top_bottlenecks:
        return recommendations

    # Check if IoU computation is a bottleneck
    iou_time = breakdown_by_category.get("IoU Computation", 0.0)
    if iou_time > 30.0:  # More than 30% of total time
        recommendations.append(
            "IoU computation is a major bottleneck (>30%). Consider vectorizing "
            "IoU calculations or using GPU acceleration."
        )

    # Check if decode is a bottleneck
    decode_time = breakdown_by_category.get("Decode/Postprocess", 0.0)
    if decode_time > 25.0:  # More than 25% of total time
        recommendations.append(
            "Decode/postprocess operations are slow (>25%). Consider batching "
            "tensor operations and reducing CPU-GPU synchronization."
        )

    # Check for frequently called functions
    for bottleneck in top_bottlenecks[:3]:
        if bottleneck["call_count"] > 100 and bottleneck["avg_time_per_call"] > 0.01:
            recommendations.append(
                f"{bottleneck['function_name']} is called {bottleneck['call_count']} times "
                f"with {bottleneck['avg_time_per_call']*1000:.2f}ms per call. "
                "Consider caching or batching operations."
            )

    # Check for single slow functions
    for bottleneck in top_bottlenecks[:2]:
        if bottleneck["percentage_of_total"] > 40.0:
            recommendations.append(
                f"{bottleneck['function_name']} accounts for "
                f"{bottleneck['percentage_of_total']:.1f}% of total time. "
                "This is the primary optimization target."
            )

    return recommendations


def _write_markdown_report(path: Path, report: Dict[str, Any]):
    """Write profiling report in Markdown format.
    
    Args:
        path: Path to save the Markdown report
        report: Report data dictionary
    """
    with open(path, "w") as f:
        f.write("# Performance Profiling Report\n\n")
        f.write(f"**Total Validation Time**: {report['total_time']:.2f} seconds\n\n")
        
        f.write("## Top 5 Bottlenecks\n\n")
        f.write("| Function | Total Time (s) | Calls | Avg Time/Call (s) | % of Total |\n")
        f.write("|----------|----------------|-------|-------------------|------------|\n")
        
        for bottleneck in report["top_bottlenecks"]:
            f.write(
                f"| {bottleneck['function_name']} | "
                f"{bottleneck['total_time']:.4f} | "
                f"{bottleneck['call_count']} | "
                f"{bottleneck['avg_time_per_call']:.6f} | "
                f"{bottleneck['percentage_of_total']:.2f}% |\n"
            )
        
        f.write("\n## Time Breakdown by Category\n\n")
        f.write("| Category | % of Total Time |\n")
        f.write("|----------|-----------------|\n")
        
        for category, percentage in sorted(
            report["breakdown_by_category"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            f.write(f"| {category} | {percentage:.2f}% |\n")
        
        if report["recommendations"]:
            f.write("\n## Optimization Recommendations\n\n")
            for i, rec in enumerate(report["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
        
        f.write("\n## All Profiled Functions\n\n")
        f.write("| Function | Total Time (s) | Calls | Avg Time/Call (s) | % of Total |\n")
        f.write("|----------|----------------|-------|-------------------|------------|\n")
        
        for name, data in sorted(
            report["all_results"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True,
        ):
            f.write(
                f"| {name} | "
                f"{data['total_time']:.4f} | "
                f"{data['call_count']} | "
                f"{data['avg_time_per_call']:.6f} | "
                f"{data['percentage_of_total']:.2f}% |\n"
            )

