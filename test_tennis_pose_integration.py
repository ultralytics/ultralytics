#!/usr/bin/env python3
"""
Integration test for TennisBallPoseModel with the tennis ball dataloader.

This script tests the complete pipeline:
1. TennisBallPoseModel with 4-channel input
2. TennisBallDataset with motion masks
3. End-to-end inference pipeline
"""

import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add the ultralytics package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_tennis_pose_end_to_end():
    """Test complete end-to-end pipeline with TennisBallPoseModel and dataloader."""
    print("🧪 Testing TennisBallPoseModel end-to-end integration...")
    
    try:
        from ultralytics.nn.tasks import TennisBallPoseModel
        from ultralytics.data.ball_dataset import TennisBallDataset, create_ball_dataloader
        
        # Test 1: Initialize model
        print("  ✓ Initializing TennisBallPoseModel...")
        model = TennisBallPoseModel(ch=4, nc=1, data_kpt_shape=(1, 3), verbose=False)
        print("    ✅ Model initialized successfully")
        
        # Test 2: Check dataset availability
        print("  ✓ Checking dataset availability...")
        dataset_path = Path("Dataset/game1/Clip1")
        if not dataset_path.exists():
            print("    ⚠️  Dataset not found, creating synthetic test data...")
            return test_tennis_pose_synthetic()
        
        print("    ✅ Dataset found")
        
        # Test 3: Create dataset and dataloader
        print("  ✓ Creating dataset and dataloader...")
        
        # Use YOLO format dataset
        yolo_dataset_path = Path("Dataset_YOLO")
        data_yaml_path = yolo_dataset_path / "data.yaml"
        
        # Load YAML configuration
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Motion configuration
        class MotionConfig:
            def __init__(self):
                self.method = "frame_diff"
                self.pixel_threshold = 30
                self.delta = 1
                self.window_size = 3
                self.adaptive_threshold = False
                self.speed_based_threshold = False
                self.min_motion_pixels = 10
                self.max_motion_pixels = 10000
                self.blur_kernel_size = 3
                self.morphological_ops = True
        
        motion_config = MotionConfig()
        
        dataset = TennisBallDataset(
            img_path=str(yolo_dataset_path / "images" / "train"),
            data=data_config,
            use_motion_masks=True,
            motion_config=motion_config,
            imgsz=640,
            augment=False,
            cache=False
        )
        
        dataloader = create_ball_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print("    ✅ Dataset and dataloader created successfully")
        
        # Test 4: Test with first batch
        print("  ✓ Testing with first batch...")
        for i, batch in enumerate(dataloader):
            if i >= 1:  # Only test first batch
                break
                
            images = batch['img']
            print(f"    📊 Batch shape: {images.shape}")
            
            # Verify 4-channel input
            assert images.shape[1] == 4, f"Expected 4 channels, got {images.shape[1]}"
            assert images.shape[2] == 640, f"Expected height 640, got {images.shape[2]}"
            assert images.shape[3] == 640, f"Expected width 640, got {images.shape[3]}"
            
            # Test forward pass
            with torch.no_grad():
                output = model(images)
                assert isinstance(output, (list, tuple)), "Output should be list or tuple"
                print(f"    📊 Output type: {type(output)}")
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    print(f"    📊 Output[0] shape: {output[0].shape if hasattr(output[0], 'shape') else 'No shape'}")
            
            print("    ✅ Forward pass successful")
        
        print("    ✅ End-to-end integration successful")
        return True
        
    except Exception as e:
        print(f"    ❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tennis_pose_synthetic():
    """Test TennisBallPoseModel with synthetic data when real dataset is not available."""
    print("🧪 Testing TennisBallPoseModel with synthetic data...")
    
    try:
        from ultralytics.nn.tasks import TennisBallPoseModel
        from ultralytics.nn.modules.motion_utils import combine_rgb_motion
        
        # Test 1: Initialize model
        print("  ✓ Initializing TennisBallPoseModel...")
        model = TennisBallPoseModel(ch=4, nc=1, data_kpt_shape=(1, 3), verbose=False)
        print("    ✅ Model initialized successfully")
        
        # Test 2: Create synthetic 4-channel data
        print("  ✓ Creating synthetic 4-channel data...")
        
        # Create batch of synthetic data
        batch_size = 2
        height, width = 640, 640
        
        # Create RGB images (batch_size, 3, height, width)
        rgb_images = torch.rand(batch_size, 3, height, width)
        
        # Create motion masks (batch_size, 1, height, width)
        motion_masks = torch.rand(batch_size, 1, height, width)
        
        # Combine to create 4-channel input
        combined_images = torch.cat([rgb_images, motion_masks], dim=1)
        
        assert combined_images.shape == (batch_size, 4, height, width), f"Expected ({batch_size}, 4, {height}, {width}), got {combined_images.shape}"
        print("    ✅ Synthetic 4-channel data created successfully")
        
        # Test 3: Test forward pass
        print("  ✓ Testing forward pass with synthetic data...")
        with torch.no_grad():
            output = model(combined_images)
            assert isinstance(output, (list, tuple)), "Output should be list or tuple"
            print(f"    📊 Output type: {type(output)}")
            if isinstance(output, (list, tuple)) and len(output) > 0:
                print(f"    📊 Output[0] shape: {output[0].shape if hasattr(output[0], 'shape') else 'No shape'}")
        
        print("    ✅ Forward pass with synthetic data successful")
        
        # Test 4: Test with different batch sizes
        print("  ✓ Testing different batch sizes...")
        for bs in [1, 4, 8]:
            rgb_batch = torch.rand(bs, 3, height, width)
            motion_batch = torch.rand(bs, 1, height, width)
            combined_batch = torch.cat([rgb_batch, motion_batch], dim=1)
            
            with torch.no_grad():
                output = model(combined_batch)
                assert isinstance(output, (list, tuple)), f"Output should be list or tuple for batch size {bs}"
            
            print(f"    ✅ Batch size {bs} works correctly")
        
        print("    ✅ Synthetic data testing successful")
        return True
        
    except Exception as e:
        print(f"    ❌ Synthetic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tennis_pose_model_comparison():
    """Test TennisBallPoseModel vs regular PoseModel to ensure compatibility."""
    print("🧪 Testing TennisBallPoseModel vs PoseModel compatibility...")
    
    try:
        from ultralytics.nn.tasks import TennisBallPoseModel, PoseModel
        
        # Test 1: Compare 3-channel models
        print("  ✓ Comparing 3-channel models...")
        
        # Regular PoseModel with 3 channels
        pose_model = PoseModel(ch=3, nc=1, data_kpt_shape=(1, 3), verbose=False)
        
        # TennisBallPoseModel with 3 channels
        tennis_model_3ch = TennisBallPoseModel(ch=3, nc=1, data_kpt_shape=(1, 3), verbose=False)
        
        # Test with same input
        input_3ch = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            pose_output = pose_model(input_3ch)
            tennis_output = tennis_model_3ch(input_3ch)
        
        assert isinstance(pose_output, (list, tuple)), "PoseModel output should be list or tuple"
        assert isinstance(tennis_output, (list, tuple)), "TennisBallPoseModel output should be list or tuple"
        
        print("    ✅ 3-channel models are compatible")
        
        # Test 2: Test 4-channel TennisBallPoseModel
        print("  ✓ Testing 4-channel TennisBallPoseModel...")
        
        tennis_model_4ch = TennisBallPoseModel(ch=4, nc=1, data_kpt_shape=(1, 3), verbose=False)
        input_4ch = torch.randn(1, 4, 640, 640)
        
        with torch.no_grad():
            tennis_4ch_output = tennis_model_4ch(input_4ch)
        
        assert isinstance(tennis_4ch_output, (list, tuple)), "4-channel TennisBallPoseModel output should be list or tuple"
        
        print("    ✅ 4-channel TennisBallPoseModel works correctly")
        
        # Test 3: Verify model properties
        print("  ✓ Verifying model properties...")
        
        assert tennis_model_3ch.use_motion_masks == False, "3-channel model should not have motion masks"
        assert tennis_model_4ch.use_motion_masks == True, "4-channel model should have motion masks"
        assert tennis_model_3ch.tennis_ball_keypoints == ["center"], "Should have center keypoint"
        assert tennis_model_4ch.tennis_ball_keypoints == ["center"], "Should have center keypoint"
        
        print("    ✅ Model properties verified")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Model comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests for TennisBallPoseModel."""
    print("🚀 Starting TennisBallPoseModel integration tests...\n")
    
    tests = [
        test_tennis_pose_end_to_end,
        test_tennis_pose_model_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! TennisBallPoseModel is ready for use.")
        return True
    else:
        print("⚠️  Some integration tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
