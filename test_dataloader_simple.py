#!/usr/bin/env python3
"""
Simple test for dataloader functionality without full ultralytics imports
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Import motion utilities directly
sys.path.append(str(Path(__file__).parent / "ultralytics" / "nn" / "modules"))
from motion_utils import MotionConfig, FlexibleMotionMaskGenerator, generate_motion_mask

def test_motion_mask_generation():
    """Test motion mask generation with real frames."""
    print("Testing Motion Mask Generation")
    print("=" * 40)
    
    # Test with a real clip
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return None
    
    # Get frame files
    frame_files = sorted(list(clip_path.glob("*.jpg")))
    if len(frame_files) < 3:
        print(f"Not enough frames in {clip_path}")
        return None
    
    print(f"Found {len(frame_files)} frames")
    
    # Load first 3 frames
    frames = []
    for i in range(3):
        frame = cv2.imread(str(frame_files[i]))
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Failed to load frame: {frame_files[i]}")
            return None
    
    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    
    # Test basic motion mask generation
    print("\n1. Testing basic motion mask generation...")
    motion_mask = generate_motion_mask(frames, pixel_threshold=15, delta=1)
    print(f"Motion mask shape: {motion_mask.shape}")
    print(f"Motion mask dtype: {motion_mask.dtype}")
    print(f"Motion pixels: {np.sum(motion_mask > 0)}")
    
    # Test with MotionConfig
    print("\n2. Testing MotionConfig...")
    config = MotionConfig(
        pixel_threshold=15,
        delta=1,
        window_size=5,
        adaptive_threshold=True,
        min_motion_pixels=100,
        max_motion_pixels=50000
    )
    print(f"Config: {config}")
    
    # Test FlexibleMotionMaskGenerator
    print("\n3. Testing FlexibleMotionMaskGenerator...")
    generator = FlexibleMotionMaskGenerator(config)
    enhanced_mask = generator.generate_enhanced(frames)
    print(f"Enhanced mask shape: {enhanced_mask.shape}")
    print(f"Enhanced motion pixels: {np.sum(enhanced_mask > 0)}")
    
    # Test 4-channel combination
    print("\n4. Testing 4-channel combination...")
    from motion_utils import combine_rgb_motion
    
    rgb_frame = frames[1]  # Use middle frame
    combined = combine_rgb_motion(rgb_frame, enhanced_mask)
    print(f"Combined shape: {combined.shape}")
    print(f"Combined dtype: {combined.dtype}")
    
    # Save test results
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "test_frame.jpg"), frames[1])
    cv2.imwrite(str(output_dir / "test_motion_mask.jpg"), motion_mask * 255)
    cv2.imwrite(str(output_dir / "test_enhanced_mask.jpg"), enhanced_mask * 255)
    
    # Save combined as separate channels for visualization
    combined_vis = np.zeros((combined.shape[0], combined.shape[1] * 4, 3), dtype=np.uint8)
    combined_vis[:, :combined.shape[1]] = combined[:, :, :3]  # RGB
    combined_vis[:, combined.shape[1]:combined.shape[1]*2, 0] = combined[:, :, 3] * 255  # Motion as red
    combined_vis[:, combined.shape[1]*2:combined.shape[1]*3, 1] = combined[:, :, 3] * 255  # Motion as green
    combined_vis[:, combined.shape[1]*3:combined.shape[1]*4, 2] = combined[:, :, 3] * 255  # Motion as blue
    
    cv2.imwrite(str(output_dir / "test_combined_4channel.jpg"), combined_vis)
    
    print(f"\nTest results saved to {output_dir}/")
    print("✅ All tests passed!")
    
    return True

def test_dataset_structure():
    """Test dataset structure and CSV loading."""
    print("\nTesting Dataset Structure")
    print("=" * 40)
    
    # Test CSV loading
    csv_path = Path("Dataset/game1/Clip1/Label.csv")
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return False
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Check for valid annotations
    valid_annotations = df[df['visibility'] > 0]
    print(f"Valid annotations: {len(valid_annotations)}/{len(df)}")
    
    return True

if __name__ == "__main__":
    print("Tennis Ball Dataloader Test")
    print("=" * 50)
    
    # Test dataset structure
    if not test_dataset_structure():
        print("❌ Dataset structure test failed")
        sys.exit(1)
    
    # Test motion mask generation
    if not test_motion_mask_generation():
        print("❌ Motion mask generation test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed successfully!")
