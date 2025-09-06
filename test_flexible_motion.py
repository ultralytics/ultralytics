#!/usr/bin/env python3
"""
Test script for flexible motion utilities
Tests the enhanced motion mask generation with flexible configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

# Add ultralytics to path
sys.path.append(str(Path(__file__).parent))

# Import the enhanced motion utilities
from ultralytics.nn.modules.motion_utils import (
    MotionConfig, 
    FlexibleMotionMaskGenerator,
    adaptive_motion_threshold,
    create_motion_visualization
)

def test_flexible_motion_configs():
    """Test different flexible motion configurations."""
    print("Testing Flexible Motion Configurations")
    print("=" * 50)
    
    # Load test frames
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return
    
    frames = []
    for i in range(5):
        frame_path = clip_path / f"{i:04d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path)
            frames.append(np.array(img))
    
    if len(frames) < 2:
        print("Not enough frames for testing")
        return
    
    print(f"Loaded {len(frames)} frames from {clip_path}")
    
    # Test different configurations
    configs = [
        MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=5,
            adaptive_threshold=False,
            min_motion_pixels=100,
            max_motion_pixels=50000
        ),
        MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=5,
            adaptive_threshold=True,
            min_motion_pixels=100,
            max_motion_pixels=50000
        ),
        MotionConfig(
            pixel_threshold=20,
            delta=1,
            window_size=3,
            adaptive_threshold=True,
            min_motion_pixels=200,
            max_motion_pixels=30000,
            morphological_ops=True
        ),
        MotionConfig(
            pixel_threshold=10,
            delta=2,
            window_size=5,
            adaptive_threshold=True,
            min_motion_pixels=50,
            max_motion_pixels=80000,
            blur_kernel_size=5
        )
    ]
    
    fig, axes = plt.subplots(2, len(configs), figsize=(20, 10))
    fig.suptitle('Flexible Motion Configuration Test', fontsize=16)
    
    for i, config in enumerate(configs):
        # Create flexible generator
        generator = FlexibleMotionMaskGenerator(config)
        
        # Generate enhanced motion mask
        motion_mask = generator.generate_enhanced(frames)
        
        # Plot original frame
        axes[0, i].imshow(frames[0])
        axes[0, i].set_title(f'Original Frame\nConfig {i+1}')
        axes[0, i].axis('off')
        
        # Plot motion mask
        axes[1, i].imshow(motion_mask, cmap='gray')
        axes[1, i].set_title(f'Enhanced Motion Mask\nConfig {i+1}')
        axes[1, i].axis('off')
        
        # Print statistics
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.size
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        print(f"Config {i+1}:")
        print(f"  Threshold: {config.pixel_threshold}, Adaptive: {config.adaptive_threshold}")
        print(f"  Window: {config.window_size}, Delta: {config.delta}")
        print(f"  Motion pixels: {motion_pixels:,} ({motion_percentage:.2f}%)")
        print(f"  Min/Max motion: {config.min_motion_pixels}/{config.max_motion_pixels}")
        print()
    
    plt.tight_layout()
    plt.savefig('flexible_motion_configs_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_adaptive_threshold():
    """Test adaptive threshold functionality."""
    print("Testing Adaptive Threshold")
    print("=" * 30)
    
    # Load frames from different clips to test adaptive thresholding
    test_clips = [
        "Dataset/game1/Clip1",
        "Dataset/game4/Clip1",  # High speed clip
        "Dataset/game6/Clip1",  # Long trajectory clip
    ]
    
    fig, axes = plt.subplots(len(test_clips), 3, figsize=(18, 6*len(test_clips)))
    fig.suptitle('Adaptive Threshold Test Across Different Clips', fontsize=16)
    
    for clip_idx, clip_path in enumerate(test_clips):
        clip_path = Path(clip_path)
        if not clip_path.exists():
            continue
        
        # Load frames
        frames = []
        for i in range(5):
            frame_path = clip_path / f"{i:04d}.jpg"
            if frame_path.exists():
                img = Image.open(frame_path)
                frames.append(np.array(img))
        
        if len(frames) < 2:
            continue
        
        # Test fixed vs adaptive threshold
        fixed_config = MotionConfig(pixel_threshold=15, adaptive_threshold=False)
        adaptive_config = MotionConfig(pixel_threshold=15, adaptive_threshold=True)
        
        fixed_generator = FlexibleMotionMaskGenerator(fixed_config)
        adaptive_generator = FlexibleMotionMaskGenerator(adaptive_config)
        
        fixed_mask = fixed_generator.generate_enhanced(frames)
        adaptive_mask = adaptive_generator.generate_enhanced(frames)
        
        # Calculate adaptive threshold
        adaptive_thresh = adaptive_motion_threshold(frames, base_threshold=15)
        
        # Plot results
        axes[clip_idx, 0].imshow(frames[0])
        axes[clip_idx, 0].set_title(f'{clip_path.name}\nOriginal Frame')
        axes[clip_idx, 0].axis('off')
        
        axes[clip_idx, 1].imshow(fixed_mask, cmap='gray')
        axes[clip_idx, 1].set_title(f'Fixed Threshold (15)\n{np.sum(fixed_mask > 0):,} pixels')
        axes[clip_idx, 1].axis('off')
        
        axes[clip_idx, 2].imshow(adaptive_mask, cmap='gray')
        axes[clip_idx, 2].set_title(f'Adaptive Threshold ({adaptive_thresh})\n{np.sum(adaptive_mask > 0):,} pixels')
        axes[clip_idx, 2].axis('off')
        
        print(f"{clip_path.name}:")
        print(f"  Fixed threshold (15): {np.sum(fixed_mask > 0):,} motion pixels")
        print(f"  Adaptive threshold ({adaptive_thresh}): {np.sum(adaptive_mask > 0):,} motion pixels")
        print()
    
    plt.tight_layout()
    plt.savefig('adaptive_threshold_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_motion_visualization():
    """Test the enhanced motion visualization."""
    print("Testing Enhanced Motion Visualization")
    print("=" * 40)
    
    # Load frames
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return
    
    frames = []
    for i in range(5):
        frame_path = clip_path / f"{i:04d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path)
            frames.append(np.array(img))
    
    if len(frames) < 2:
        print("Not enough frames for testing")
        return
    
    # Create enhanced configuration
    config = MotionConfig(
        pixel_threshold=15,
        delta=1,
        window_size=5,
        adaptive_threshold=True,
        min_motion_pixels=100,
        max_motion_pixels=50000,
        morphological_ops=True
    )
    
    # Generate motion mask
    generator = FlexibleMotionMaskGenerator(config)
    motion_mask = generator.generate_enhanced(frames)
    
    # Create visualization
    create_motion_visualization(
        frames, 
        motion_mask, 
        config, 
        save_path="enhanced_motion_visualization.png"
    )

def test_motion_analysis_integration():
    """Test integration with motion analysis findings."""
    print("Testing Motion Analysis Integration")
    print("=" * 40)
    
    # Based on motion analysis findings, test different scenarios
    scenarios = [
        {
            "name": "High Speed Motion (game4/Clip1)",
            "clip": "Dataset/game4/Clip1",
            "config": MotionConfig(
                pixel_threshold=20,  # Higher threshold for high speed
                adaptive_threshold=True,
                min_motion_pixels=500,
                max_motion_pixels=100000,
                morphological_ops=True
            )
        },
        {
            "name": "Long Trajectory (game6/Clip1)", 
            "clip": "Dataset/game6/Clip1",
            "config": MotionConfig(
                pixel_threshold=15,
                adaptive_threshold=True,
                min_motion_pixels=200,
                max_motion_pixels=80000,
                blur_kernel_size=3
            )
        },
        {
            "name": "Standard Motion (game1/Clip1)",
            "clip": "Dataset/game1/Clip1", 
            "config": MotionConfig(
                pixel_threshold=15,
                adaptive_threshold=True,
                min_motion_pixels=100,
                max_motion_pixels=50000
            )
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        clip_path = Path(scenario["clip"])
        if not clip_path.exists():
            continue
        
        # Load frames
        frames = []
        for i in range(5):
            frame_path = clip_path / f"{i:04d}.jpg"
            if frame_path.exists():
                img = Image.open(frame_path)
                frames.append(np.array(img))
        
        if len(frames) < 2:
            continue
        
        # Generate motion mask
        generator = FlexibleMotionMaskGenerator(scenario["config"])
        motion_mask = generator.generate_enhanced(frames)
        
        # Calculate statistics
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.size
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        results.append({
            "scenario": scenario["name"],
            "motion_pixels": motion_pixels,
            "motion_percentage": motion_percentage,
            "config": scenario["config"]
        })
        
        print(f"{scenario['name']}:")
        print(f"  Motion pixels: {motion_pixels:,} ({motion_percentage:.2f}%)")
        print(f"  Config: threshold={scenario['config'].pixel_threshold}, "
              f"adaptive={scenario['config'].adaptive_threshold}")
        print()
    
    return results

def main():
    """Run all flexible motion tests."""
    print("Flexible Motion Utils Test Suite")
    print("=" * 50)
    
    # Test 1: Flexible configurations
    test_flexible_motion_configs()
    
    # Test 2: Adaptive threshold
    test_adaptive_threshold()
    
    # Test 3: Enhanced visualization
    test_motion_visualization()
    
    # Test 4: Motion analysis integration
    results = test_motion_analysis_integration()
    
    print("All flexible motion tests completed!")
    print("Generated files:")
    print("  - flexible_motion_configs_test.png")
    print("  - adaptive_threshold_test.png")
    print("  - enhanced_motion_visualization.png")
    
    # Summary
    print("\nTest Summary:")
    for result in results:
        print(f"  {result['scenario']}: {result['motion_pixels']:,} pixels "
              f"({result['motion_percentage']:.2f}%)")

if __name__ == "__main__":
    main()
