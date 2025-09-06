#!/usr/bin/env python3
"""
Test script for TennisBallDataset with full ultralytics integration
Tests the complete 4-channel dataloader pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_tennis_ball_dataset():
    """Test TennisBallDataset with real data."""
    print("Testing TennisBallDataset")
    print("=" * 50)
    
    try:
        from ultralytics.data.ball_dataset import TennisBallDataset, create_ball_dataloader
        from ultralytics.nn.modules.motion_utils import MotionConfig
        
        # Dataset configuration
        data_config = {
            "names": {0: "tennis_ball"},
            "kpt_shape": [1, 3],  # 1 keypoint (ball center) with 3 dimensions (x, y, visibility)
            "channels": 4  # 4-channel input
        }
        
        # Motion configuration
        motion_config = MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=5,
            adaptive_threshold=True,
            min_motion_pixels=100,
            max_motion_pixels=50000
        )
        
        # Test with a single clip
        dataset_path = Path("Dataset/game1/Clip1")
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return False
        
        print(f"Dataset path: {dataset_path}")
        
        # Create dataset
        dataset = TennisBallDataset(
            img_path=str(dataset_path),
            data=data_config,
            motion_config=motion_config,
            dataset_path=dataset_path,
            imgsz=640,
            cache=False  # Disable caching for testing
        )
        
        print(f"Dataset created successfully")
        print(f"Dataset length: {len(dataset)}")
        
        # Test single sample
        print("\n1. Testing single sample...")
        sample = dataset[0]
        
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['img'].shape}")
        print(f"Image dtype: {sample['img'].dtype}")
        print(f"Keypoints shape: {sample['keypoints'].shape}")
        print(f"Keypoints dtype: {sample['keypoints'].dtype}")
        
        # Verify 4-channel input
        if sample['img'].shape[0] == 4:
            print("✅ 4-channel input confirmed")
        else:
            print(f"❌ Expected 4 channels, got {sample['img'].shape[0]}")
            return False
        
        # Test multiple samples
        print("\n2. Testing multiple samples...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: img={sample['img'].shape}, kpts={sample['keypoints'].shape}")
        
        # Test dataloader
        print("\n3. Testing DataLoader...")
        dataloader = create_ball_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"Dataloader created: {len(dataloader)} batches")
        
        # Test batch loading
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch image shape: {batch['img'].shape}")
        print(f"Batch keypoints shape: {batch['keypoints'].shape}")
        
        # Verify batch dimensions
        if batch['img'].shape[1] == 4:  # [batch, channels, height, width]
            print("✅ 4-channel batch confirmed")
        else:
            print(f"❌ Expected 4 channels in batch, got {batch['img'].shape[1]}")
            return False
        
        print("\n✅ All TennisBallDataset tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_integration():
    """Test motion mask integration with dataset."""
    print("\nTesting Motion Integration")
    print("=" * 50)
    
    try:
        from ultralytics.nn.modules.motion_utils import FlexibleMotionMaskGenerator, MotionConfig
        
        # Test motion mask generation
        config = MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=5,
            adaptive_threshold=True
        )
        
        generator = FlexibleMotionMaskGenerator(config)
        
        # Load test frames
        clip_path = Path("Dataset/game1/Clip1")
        if not clip_path.exists():
            print(f"Clip not found: {clip_path}")
            return False
        
        frame_files = sorted(list(clip_path.glob("*.jpg")))[:5]
        frames = []
        
        import cv2
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 3:
            print("Not enough frames for testing")
            return False
        
        print(f"Loaded {len(frames)} frames")
        
        # Generate motion mask
        motion_mask = generator.generate_enhanced(frames)
        print(f"Motion mask shape: {motion_mask.shape}")
        print(f"Motion pixels: {np.sum(motion_mask > 0)}")
        
        print("✅ Motion integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Motion integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test dataloader performance."""
    print("\nTesting Performance")
    print("=" * 50)
    
    try:
        from ultralytics.data.ball_dataset import TennisBallDataset, create_ball_dataloader
        from ultralytics.nn.modules.motion_utils import MotionConfig
        
        # Configuration
        data_config = {
            "names": {0: "tennis_ball"},
            "kpt_shape": [1, 3],
            "channels": 4
        }
        
        motion_config = MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=5,
            adaptive_threshold=True
        )
        
        dataset_path = Path("Dataset/game1/Clip1")
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            return False
        
        # Create dataset
        dataset = TennisBallDataset(
            img_path=str(dataset_path),
            data=data_config,
            motion_config=motion_config,
            dataset_path=dataset_path,
            imgsz=640,
            cache=False
        )
        
        # Create dataloader
        dataloader = create_ball_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        # Time batch loading
        import time
        start_time = time.time()
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processed {batch_count} batches in {total_time:.2f}s")
        print(f"Average time per batch: {total_time/batch_count:.2f}s")
        
        print("✅ Performance test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Tennis Ball Dataset Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test motion integration first
    if not test_motion_integration():
        success = False
    
    # Test main dataset
    if not test_tennis_ball_dataset():
        success = False
    
    # Test performance
    if not test_performance():
        success = False
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("\nThe 4-channel dataloader is ready for training!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
