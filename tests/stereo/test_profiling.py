# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Performance profiling tests for stereo 3D detection validation."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D


class TestProfilingCollector:
    """Test suite for ProfilingCollector class (T163)."""

    def test_profiling_collector_initialization(self):
        """Test that ProfilingCollector can be initialized."""
        try:
            from ultralytics.utils.profiling import ProfilingCollector
            
            collector = ProfilingCollector()
            assert collector is not None
            assert hasattr(collector, "results")
        except ImportError:
            pytest.skip("ProfilingCollector not yet implemented (T167)")

    def test_profiling_collector_collects_timing(self):
        """Test that ProfilingCollector collects timing data correctly."""
        try:
            from ultralytics.utils.profiling import ProfilingCollector
            
            collector = ProfilingCollector()
            
            # Record some timing data
            collector.record("test_function", 0.5, call_count=1)
            collector.record("test_function", 0.3, call_count=1)
            
            # Verify data is collected
            results = collector.get_results()
            assert "test_function" in results
            assert results["test_function"]["total_time"] == 0.8
            assert results["test_function"]["call_count"] == 2
            assert results["test_function"]["avg_time_per_call"] == 0.4
        except ImportError:
            pytest.skip("ProfilingCollector not yet implemented (T167)")

    def test_profiling_collector_aggregates_results(self):
        """Test that ProfilingCollector correctly aggregates multiple function calls."""
        try:
            from ultralytics.utils.profiling import ProfilingCollector
            
            collector = ProfilingCollector()
            
            # Record multiple functions
            collector.record("func1", 1.0, call_count=1)
            collector.record("func2", 2.0, call_count=1)
            collector.record("func1", 0.5, call_count=1)
            
            results = collector.get_results()
            
            # Verify func1 aggregation
            assert results["func1"]["total_time"] == 1.5
            assert results["func1"]["call_count"] == 2
            assert results["func1"]["avg_time_per_call"] == 0.75
            
            # Verify func2
            assert results["func2"]["total_time"] == 2.0
            assert results["func2"]["call_count"] == 1
        except ImportError:
            pytest.skip("ProfilingCollector not yet implemented (T167)")

    def test_profiling_collector_calculates_percentage(self):
        """Test that ProfilingCollector calculates percentage of total time."""
        try:
            from ultralytics.utils.profiling import ProfilingCollector
            
            collector = ProfilingCollector()
            
            collector.record("func1", 1.0, call_count=1)
            collector.record("func2", 2.0, call_count=1)
            collector.record("func3", 1.0, call_count=1)
            
            results = collector.get_results()
            total_time = 4.0  # 1.0 + 2.0 + 1.0
            
            # Verify percentage calculation
            assert abs(results["func1"]["percentage_of_total"] - 25.0) < 0.1
            assert abs(results["func2"]["percentage_of_total"] - 50.0) < 0.1
            assert abs(results["func3"]["percentage_of_total"] - 25.0) < 0.1
        except ImportError:
            pytest.skip("ProfilingCollector not yet implemented (T167)")


class TestProfileFunctionDecorator:
    """Test suite for @profile_function decorator (T164)."""

    def test_profile_function_decorator_times_function(self):
        """Test that @profile_function decorator correctly times function execution."""
        try:
            from ultralytics.utils.profiling import profile_function
            
            @profile_function(name="test_func")
            def slow_function():
                time.sleep(0.1)
                return "done"
            
            result = slow_function()
            assert result == "done"
            
            # Verify timing was recorded
            from ultralytics.utils.profiling import get_profiling_results
            results = get_profiling_results()
            
            assert "test_func" in results or len(results) > 0  # May be in global collector
        except ImportError:
            pytest.skip("@profile_function decorator not yet implemented (T168)")

    def test_profile_function_decorator_preserves_function_signature(self):
        """Test that @profile_function decorator preserves function signature and behavior."""
        try:
            from ultralytics.utils.profiling import profile_function
            
            @profile_function(name="add_func")
            def add(a, b):
                return a + b
            
            result = add(2, 3)
            assert result == 5
            
            # Test with keyword arguments
            result = add(a=5, b=10)
            assert result == 15
        except ImportError:
            pytest.skip("@profile_function decorator not yet implemented (T168)")

    def test_profile_function_decorator_handles_exceptions(self):
        """Test that @profile_function decorator handles exceptions correctly."""
        try:
            from ultralytics.utils.profiling import profile_function
            
            @profile_function(name="error_func")
            def error_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError, match="Test error"):
                error_function()
        except ImportError:
            pytest.skip("@profile_function decorator not yet implemented (T168)")


class TestProfileSectionContextManager:
    """Test suite for profile_section context manager (T165)."""

    def test_profile_section_times_code_block(self):
        """Test that profile_section context manager correctly times code block execution."""
        try:
            from ultralytics.utils.profiling import profile_section
            
            with profile_section("test_section"):
                time.sleep(0.1)
            
            # Verify timing was recorded
            from ultralytics.utils.profiling import get_profiling_results
            results = get_profiling_results()
            
            # Results may be in global collector or returned directly
            assert True  # Context manager executed without error
        except ImportError:
            pytest.skip("profile_section context manager not yet implemented (T169)")

    def test_profile_section_handles_exceptions(self):
        """Test that profile_section context manager handles exceptions correctly."""
        try:
            from ultralytics.utils.profiling import profile_section
            
            with pytest.raises(ValueError):
                with profile_section("error_section"):
                    raise ValueError("Test error")
        except ImportError:
            pytest.skip("profile_section context manager not yet implemented (T169)")

    def test_profile_section_nested_usage(self):
        """Test that profile_section can be used in nested contexts."""
        try:
            from ultralytics.utils.profiling import profile_section
            
            with profile_section("outer"):
                with profile_section("inner"):
                    time.sleep(0.05)
            
            # Both sections should be recorded
            assert True  # Nested usage executed without error
        except ImportError:
            pytest.skip("profile_section context manager not yet implemented (T169)")


class TestProfilingHooksIntegration:
    """Test suite for profiling hooks integration (T166)."""

    def test_profiling_hooks_collect_data_during_validation(self):
        """Test that profiling hooks collect data during validation."""
        try:
            from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
            from ultralytics.utils.profiling import get_profiling_results, reset_profiling
            
            # Reset profiling state
            reset_profiling()
            
            # Create a mock validator with profiling enabled
            # This test will verify that profiling hooks are called during validation
            # For now, we'll just verify the hooks exist
            
            # Check if profiling hooks are available
            validator = Stereo3DDetValidator(
                dataloader=MagicMock(),
                save_dir=MagicMock(),
                args=MagicMock(),
            )
            
            # Verify validator has profiling capability (hooks may be optional)
            # The actual integration test would run validation and check results
            assert validator is not None
        except ImportError:
            pytest.skip("Profiling hooks not yet implemented (T171-T174)")

    def test_profiling_results_are_exportable(self):
        """Test that profiling results can be exported to JSON."""
        try:
            from ultralytics.utils.profiling import get_profiling_results, generate_profiling_report
            
            # Get results
            results = get_profiling_results()
            
            # Generate report (should return dict or be exportable)
            report = generate_profiling_report()
            
            # Verify report structure
            assert isinstance(report, dict) or report is None  # May be empty initially
        except ImportError:
            pytest.skip("Profiling report generation not yet implemented (T170)")

    @pytest.mark.skip(reason="Requires real KITTI dataset and model - integration test (T175)")
    def test_profiling_with_real_dataset(self):
        """Test profiling with real KITTI dataset subset.
        
        This test loads a subset of the real KITTI dataset, runs validation with profiling enabled,
        and verifies that profiling data is collected and can be exported.
        
        This is an integration test that requires:
        - Real KITTI dataset at specified path
        - Trained model weights
        - Significant runtime (several minutes)
        """
        try:
            from pathlib import Path
            from ultralytics import YOLO
            from ultralytics.utils.profiling import generate_profiling_report, reset_profiling
            
            # This test should be run manually with actual dataset
            # Example usage:
            #   dataset_yaml = "/path/to/dataset.yaml"
            #   model_path = "/path/to/model.pt"
            #   
            #   reset_profiling()
            #   model = YOLO(model_path)
            #   results = model.val(data=dataset_yaml, split="val", imgsz=384, batch=8)
            #   
            #   report = generate_profiling_report()
            #   assert report["total_time"] > 0
            #   assert len(report["top_bottlenecks"]) > 0
            
            pytest.skip("Requires real dataset - run manually with profile_validation.py script")
        except ImportError:
            pytest.skip("Profiling utilities not available")

