#!/usr/bin/env python3
"""
Standalone test for motion mask generation
Tests motion mask generation without ultralytics dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Union

def generate_motion_mask(
    frames: Union[List[np.ndarray], np.ndarray], 
    pixel_threshold: int = 15, 
    delta: int = 1,
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Generate motion mask from a sequence of frames.
    
    This function calculates motion between consecutive frames and creates a binary mask
    indicating areas of significant motion. Used for tennis ball tracking with 4-channel input.

    Args:
        frames: Sequence of frames (H, W, C) or (C, H, W)
        pixel_threshold: Threshold for significant motion detection (default: 15)
        delta: Frame step for motion calculation (default: 1 for consecutive frames)
        window_size: Number of frames to consider. If None, uses all frames.

    Returns:
        Binary motion mask (H, W) with values 0 or 255
    """
    if not frames:
        raise ValueError("Empty frame sequence provided")
    
    # Convert to list if single array
    if isinstance(frames, np.ndarray):
        frames = [frames]
    
    # Limit window size if specified
    if window_size is not None:
        frames = frames[:window_size]
    
    if len(frames) < 2:
        # Return empty mask if insufficient frames
        return np.zeros_like(frames[0][:, :, 0] if len(frames[0].shape) == 3 else frames[0], dtype=np.uint8)
    
    # Convert to grayscale for motion detection
    gray_frames = []
    for frame in frames:
        if len(frame.shape) == 3:
            # Convert RGB to grayscale
            gray_frame = np.mean(frame, axis=2)
        else:
            gray_frame = frame
        gray_frames.append(gray_frame.astype(np.float32))
    
    # Calculate motion between frames with specified delta
    motion_masks = []
    for i in range(len(gray_frames) - delta):
        diff = np.abs(gray_frames[i + delta] - gray_frames[i])
        motion_mask = (diff > pixel_threshold).astype(np.uint8) * 255
        motion_masks.append(motion_mask)
    
    # Combine motion masks (OR operation)
    if motion_masks:
        combined_mask = np.zeros_like(motion_masks[0])
        for mask in motion_masks:
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask
    
    return np.zeros_like(gray_frames[0], dtype=np.uint8)

class MotionMaskGenerator:
    """
    Motion mask generator with configurable parameters and caching support.
    """
    
    def __init__(
        self,
        pixel_threshold: int = 15,
        delta: int = 1,
        window_size: int = 5,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize motion mask generator.

        Args:
            pixel_threshold: Threshold for significant motion detection
            delta: Frame step for motion calculation
            window_size: Number of frames to consider for motion detection
            cache_dir: Directory for caching pre-computed masks
        """
        self.pixel_threshold = pixel_threshold
        self.delta = delta
        self.window_size = window_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self, 
        frames: Union[List[np.ndarray], np.ndarray],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate motion mask with optional caching.

        Args:
            frames: Sequence of frames
            cache_key: Optional key for caching the result

        Returns:
            Motion mask as numpy array
        """
        # Check cache first
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_motion.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Generate motion mask
        motion_mask = generate_motion_mask(
            frames, 
            self.pixel_threshold, 
            self.delta, 
            self.window_size
        )
        
        # Save to cache
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_motion.npy"
            np.save(cache_path, motion_mask)
        
        return motion_mask

def test_motion_mask_generation():
    """Test motion mask generation with real data."""
    print("Testing Motion Mask Generation")
    print("=" * 40)
    
    # Test with a real clip
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return
    
    # Load first 5 frames
    frames = []
    for i in range(5):
        frame_path = clip_path / f"{i:04d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path)
            frames.append(np.array(img))
        else:
            print(f"Frame not found: {frame_path}")
            break
    
    if len(frames) < 2:
        print("Not enough frames for motion detection")
        return
    
    print(f"Loaded {len(frames)} frames from {clip_path}")
    print(f"Frame shape: {frames[0].shape}")
    
    # Test different thresholds
    thresholds = [10, 15, 20, 25]
    
    fig, axes = plt.subplots(2, len(thresholds), figsize=(20, 10))
    fig.suptitle('Motion Mask Generation Test - Different Thresholds', fontsize=16)
    
    for i, threshold in enumerate(thresholds):
        # Generate motion mask
        motion_mask = generate_motion_mask(frames, pixel_threshold=threshold, delta=1, window_size=5)
        
        # Plot original frame
        axes[0, i].imshow(frames[0])
        axes[0, i].set_title(f'Original Frame (Threshold={threshold})')
        axes[0, i].axis('off')
        
        # Plot motion mask
        axes[1, i].imshow(motion_mask, cmap='gray')
        axes[1, i].set_title(f'Motion Mask (Threshold={threshold})')
        axes[1, i].axis('off')
        
        # Calculate motion statistics
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.size
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        print(f"Threshold {threshold}: {motion_pixels:,} motion pixels ({motion_percentage:.2f}%)")
    
    plt.tight_layout()
    plt.savefig('motion_standalone_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return motion_mask

def test_motion_visualization():
    """Test motion visualization similar to the expected output."""
    print("\nTesting Motion Visualization")
    print("=" * 40)
    
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return
    
    # Load frames
    frames = []
    for i in range(5):
        frame_path = clip_path / f"{i:04d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path)
            frames.append(np.array(img))
    
    if len(frames) < 2:
        print("Not enough frames for motion detection")
        return
    
    # Generate motion mask
    motion_mask = generate_motion_mask(frames, pixel_threshold=15, delta=1, window_size=5)
    
    # Create visualization similar to expected output
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tennis Ball Motion Analysis - Game1/Clip1', fontsize=16)
    
    # 1. Original frame
    axes[0, 0].imshow(frames[0])
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # 2. Motion mask
    axes[0, 1].imshow(motion_mask, cmap='gray')
    axes[0, 1].set_title('Motion Mask (Threshold=15)')
    axes[0, 1].axis('off')
    
    # 3. Motion overlay on original frame
    axes[1, 0].imshow(frames[0])
    # Overlay motion mask in red
    motion_overlay = np.zeros((*frames[0].shape[:2], 4))
    motion_overlay[:, :, 0] = motion_mask / 255.0  # Red channel
    motion_overlay[:, :, 3] = motion_mask / 255.0 * 0.5  # Alpha channel
    axes[1, 0].imshow(motion_overlay)
    axes[1, 0].set_title('Motion Overlay on Frame')
    axes[1, 0].axis('off')
    
    # 4. Motion statistics
    axes[1, 1].text(0.1, 0.8, 'Motion Statistics:', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    motion_pixels = np.sum(motion_mask > 0)
    total_pixels = motion_mask.size
    motion_percentage = (motion_pixels / total_pixels) * 100
    
    axes[1, 1].text(0.1, 0.7, f'Motion pixels: {motion_pixels:,}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Total pixels: {total_pixels:,}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Motion %: {motion_percentage:.2f}%', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Frames processed: {len(frames)}', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f'Threshold: 15', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, f'Window size: 5', fontsize=12, transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('motion_visualization_standalone.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Motion visualization saved as 'motion_visualization_standalone.png'")
    print(f"Motion statistics: {motion_pixels:,} pixels ({motion_percentage:.2f}%)")

def test_motion_mask_generator():
    """Test the MotionMaskGenerator class."""
    print("\nTesting MotionMaskGenerator Class")
    print("=" * 40)
    
    # Create generator with different configurations
    configs = [
        {"pixel_threshold": 15, "delta": 1, "window_size": 5},
        {"pixel_threshold": 20, "delta": 1, "window_size": 3},
        {"pixel_threshold": 10, "delta": 2, "window_size": 5},
    ]
    
    clip_path = Path("Dataset/game1/Clip1")
    if not clip_path.exists():
        print(f"Clip not found: {clip_path}")
        return
    
    # Load frames
    frames = []
    for i in range(7):  # Load more frames for testing
        frame_path = clip_path / f"{i:04d}.jpg"
        if frame_path.exists():
            img = Image.open(frame_path)
            frames.append(np.array(img))
    
    if len(frames) < 3:
        print("Not enough frames for testing")
        return
    
    fig, axes = plt.subplots(2, len(configs), figsize=(18, 8))
    fig.suptitle('MotionMaskGenerator Test - Different Configurations', fontsize=16)
    
    for i, config in enumerate(configs):
        generator = MotionMaskGenerator(**config)
        motion_mask = generator.generate(frames)
        
        # Plot original frame
        axes[0, i].imshow(frames[0])
        axes[0, i].set_title(f'Original Frame\n{config}')
        axes[0, i].axis('off')
        
        # Plot motion mask
        axes[1, i].imshow(motion_mask, cmap='gray')
        axes[1, i].set_title(f'Motion Mask\n{config}')
        axes[1, i].axis('off')
        
        print(f"Config {i+1}: {config}")
        print(f"  Motion pixels: {np.sum(motion_mask > 0):,}")
    
    plt.tight_layout()
    plt.savefig('motion_generator_standalone_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all tests."""
    print("Motion Utils Standalone Test Suite")
    print("=" * 50)
    
    # Test 1: Basic motion mask generation
    motion_mask = test_motion_mask_generation()
    
    # Test 2: MotionMaskGenerator class
    test_motion_mask_generator()
    
    # Test 3: Motion visualization
    test_motion_visualization()
    
    print("\nAll tests completed!")
    print("Generated files:")
    print("  - motion_standalone_test.png")
    print("  - motion_generator_standalone_test.png")
    print("  - motion_visualization_standalone.png")

if __name__ == "__main__":
    main()
