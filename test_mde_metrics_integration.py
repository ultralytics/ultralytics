#!/usr/bin/env python3
"""
Test script for MDE metrics integration in the main metrics module.
"""

import sys
import os
import torch
import numpy as np

def test_mde_metrics_integration():
    """Test the MDEMetrics class integration."""
    print("🧪 Testing MDEMetrics integration...")
    
    try:
        from ultralytics.utils.metrics import MDEMetrics
        
        # Create MDE metrics instance
        names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        mde_metrics = MDEMetrics(names)
        
        print("   ✅ MDEMetrics created successfully!")
        print(f"   📊 Task: {mde_metrics.task}")
        print(f"   📊 Number of classes: {len(mde_metrics.names)}")
        
        # Test depth calculation methods
        pred_depths = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        gt_depths = torch.tensor([12.0, 18.0, 32.0, 38.0, 52.0])
        
        # Test depth error rate
        error_rate = mde_metrics.calculate_depth_error(pred_depths, gt_depths)
        print(f"   ✅ Depth error rate: {error_rate:.2f}%")
        
        # Test absolute error
        abs_error = mde_metrics.calculate_absolute_depth_error(pred_depths, gt_depths)
        print(f"   ✅ Absolute error: {abs_error:.2f}m")
        
        # Test RMSE
        rmse = mde_metrics.calculate_squared_depth_error(pred_depths, gt_depths)
        print(f"   ✅ RMSE: {rmse:.2f}m")
        
        # Test accuracy metrics
        acc_metrics = mde_metrics.calculate_depth_accuracy(pred_depths, gt_depths)
        print(f"   ✅ Accuracy metrics: {acc_metrics}")
        
        # Test depth statistics
        mde_metrics.print_depth_statistics(pred_depths, "Test Predictions")
        
        # Test keys property
        keys = mde_metrics.keys
        print(f"   ✅ Metrics keys: {len(keys)} keys")
        print(f"      Detection keys: {[k for k in keys if 'precision' in k or 'recall' in k or 'mAP' in k]}")
        print(f"      Depth keys: {[k for k in keys if 'depth' in k]}")
        
        # Test mean results
        mean_results = mde_metrics.mean_results()
        print(f"   ✅ Mean results: {len(mean_results)} metrics")
        
        # Test fitness
        fitness = mde_metrics.fitness
        print(f"   ✅ Fitness score: {fitness:.4f}")
        
        # Test curves
        curves = mde_metrics.curves
        print(f"   ✅ Curves: {len(curves)} curve types")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing MDEMetrics: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mde_metrics_with_stats():
    """Test MDEMetrics with actual statistics."""
    print("\n🧪 Testing MDEMetrics with statistics...")
    
    try:
        from ultralytics.utils.metrics import MDEMetrics
        
        # Create MDE metrics instance
        names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        mde_metrics = MDEMetrics(names)
        
        # Simulate some statistics
        mde_metrics.stats["pred_depths"] = [
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([15.0, 25.0, 35.0])
        ]
        mde_metrics.stats["target_depths"] = [
            torch.tensor([12.0, 18.0, 32.0]),
            torch.tensor([14.0, 24.0, 36.0])
        ]
        
        # Process the metrics
        stats = mde_metrics.process()
        
        print("   ✅ Statistics processed successfully!")
        print(f"   📊 Processed stats keys: {list(stats.keys())}")
        
        # Test mean results after processing
        mean_results = mde_metrics.mean_results()
        print(f"   ✅ Mean results after processing: {len(mean_results)} metrics")
        
        # Test summary
        summary = mde_metrics.summary()
        print(f"   ✅ Summary generated: {len(summary)} classes")
        
        if summary:
            print(f"   📊 Sample summary keys: {list(summary[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing MDEMetrics with stats: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_integration():
    """Test the training script integration with MDEMetrics."""
    print("\n🧪 Testing training script integration...")
    
    try:
        # Test importing the training script
        sys.path.append('/root/ultralytics/examples')
        from train_yolo11_mde import (
            predict_with_depth,
            evaluate_mde_model,
            train_yolo_mde
        )
        
        print("   ✅ Training script functions imported successfully!")
        
        # Test that MDEMetrics is properly imported
        from ultralytics.utils.metrics import MDEMetrics
        print("   ✅ MDEMetrics import in training script works!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing training script integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MDE metrics integration tests."""
    print("=" * 60)
    print("🎯 MDE Metrics Integration Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_mde_metrics_integration()
    success &= test_mde_metrics_with_stats()
    success &= test_training_script_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🏁 All MDE metrics integration tests passed!")
        print("=" * 60)
        print("\n📋 MDE metrics are now fully integrated!")
        print("\n🚀 Usage examples:")
        print("   # Create MDE metrics")
        print("   from ultralytics.utils.metrics import MDEMetrics")
        print("   mde_metrics = MDEMetrics(names)")
        print("\n   # Calculate depth metrics")
        print("   error_rate = mde_metrics.calculate_depth_error(pred_depths, gt_depths)")
        print("\n   # Get comprehensive results")
        print("   results = mde_metrics.mean_results()")
        print("   summary = mde_metrics.summary()")
    else:
        print("❌ Some tests failed!")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
