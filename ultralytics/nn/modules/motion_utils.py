# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from pathlib import Path

__all__ = (
    "generate_motion_mask",
    "generate_motion_mask_batch", 
    "combine_rgb_motion",
    "MotionMaskGenerator",
    "precompute_motion_masks",
    "adaptive_motion_threshold",
    "MotionConfig",
    "create_motion_visualization"
)


def generate_motion_mask(
    frames: Union[List[np.ndarray], torch.Tensor], 
    pixel_threshold: int = 15, 
    delta: int = 1,
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Generate motion mask from a sequence of frames.
    
    This function calculates motion between consecutive frames and creates a binary mask
    indicating areas of significant motion. Used for tennis ball tracking with 4-channel input.

    Args:
        frames (List[np.ndarray] | torch.Tensor): Sequence of frames (H, W, C) or (C, H, W)
        pixel_threshold (int): Threshold for significant motion detection (default: 15)
        delta (int): Frame step for motion calculation (default: 1 for consecutive frames)
        window_size (int, optional): Number of frames to consider. If None, uses all frames.

    Returns:
        (np.ndarray): Binary motion mask (H, W) with values 0 or 255

    Examples:
        >>> import numpy as np
        >>> frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        >>> motion_mask = generate_motion_mask(frames, pixel_threshold=15)
        >>> motion_mask.shape
        (480, 640)
        >>> motion_mask.dtype
        dtype('uint8')
    """
    if not frames:
        raise ValueError("Empty frame sequence provided")
    
    # Convert to numpy if torch tensor
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
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


def generate_motion_mask_batch(
    frame_batch: torch.Tensor,
    pixel_threshold: int = 15,
    delta: int = 1
) -> torch.Tensor:
    """
    Generate motion masks for a batch of frame sequences.
    
    Optimized version for batch processing during training/inference.

    Args:
        frame_batch (torch.Tensor): Batch of frame sequences (B, T, C, H, W)
        pixel_threshold (int): Threshold for significant motion detection
        delta (int): Frame step for motion calculation

    Returns:
        (torch.Tensor): Batch of motion masks (B, 1, H, W)

    Examples:
        >>> import torch
        >>> batch = torch.randn(4, 5, 3, 480, 640)  # 4 sequences, 5 frames each
        >>> motion_masks = generate_motion_mask_batch(batch, pixel_threshold=15)
        >>> motion_masks.shape
        torch.Size([4, 1, 480, 640])
    """
    B, T, C, H, W = frame_batch.shape
    
    if T < 2:
        return torch.zeros(B, 1, H, W, device=frame_batch.device, dtype=torch.uint8)
    
    # Convert to grayscale
    gray_batch = torch.mean(frame_batch, dim=2, keepdim=True)  # (B, T, 1, H, W)
    
    # Calculate motion between frames
    motion_masks = []
    for i in range(T - delta):
        diff = torch.abs(gray_batch[:, i + delta] - gray_batch[:, i])  # (B, 1, H, W)
        motion_mask = (diff > pixel_threshold).float() * 255.0
        motion_masks.append(motion_mask)
    
    # Combine motion masks (max operation)
    if motion_masks:
        combined_mask = torch.stack(motion_masks, dim=1)  # (B, T-delta, 1, H, W)
        combined_mask = torch.max(combined_mask, dim=1)[0]  # (B, 1, H, W)
        return combined_mask.to(torch.uint8)
    
    return torch.zeros(B, 1, H, W, device=frame_batch.device, dtype=torch.uint8)


def combine_rgb_motion(
    rgb_frames: Union[np.ndarray, torch.Tensor], 
    motion_masks: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Combine RGB frames with motion masks to create 4-channel input.
    
    Concatenates RGB (3 channels) with motion mask (1 channel) for YOLO11Pose adaptation.

    Args:
        rgb_frames: RGB frames (H, W, 3) or (B, H, W, 3) for numpy, (B, 3, H, W) for torch
        motion_masks: Motion masks (H, W) or (B, H, W) for numpy, (B, 1, H, W) for torch

    Returns:
        4-channel input (H, W, 4) or (B, H, W, 4) for numpy, (B, 4, H, W) for torch

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> # Numpy arrays
        >>> rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> motion = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        >>> combined = combine_rgb_motion(rgb, motion)
        >>> combined.shape
        (480, 640, 4)
        >>> # Torch tensors
        >>> rgb = torch.randn(2, 3, 480, 640)
        >>> motion = torch.randint(0, 2, (2, 1, 480, 640)) * 255
        >>> combined = combine_rgb_motion(rgb, motion)
        >>> combined.shape
        torch.Size([2, 4, 480, 640])
    """
    # Handle numpy arrays
    if isinstance(rgb_frames, np.ndarray) and isinstance(motion_masks, np.ndarray):
        # Ensure motion mask is normalized
        if motion_masks.dtype == np.uint8:
            motion_masks = motion_masks.astype(np.float32) / 255.0
        
        # Add channel dimension to motion mask if needed
        if motion_masks.ndim == 2:
            motion_masks = motion_masks[..., np.newaxis]
        
        # Concatenate along channel dimension
        return np.concatenate([rgb_frames, motion_masks], axis=-1)
    
    # Handle torch tensors
    elif isinstance(rgb_frames, torch.Tensor) and isinstance(motion_masks, torch.Tensor):
        # Ensure motion mask is normalized
        if motion_masks.dtype == torch.uint8:
            motion_masks = motion_masks.float() / 255.0
        
        # Concatenate along channel dimension
        return torch.cat([rgb_frames, motion_masks], dim=-3)
    
    else:
        raise TypeError("Both rgb_frames and motion_masks must be of the same type (numpy.ndarray or torch.Tensor)")


class MotionMaskGenerator:
    """
    Motion mask generator with configurable parameters and caching support.
    
    Provides a flexible interface for generating motion masks with different
    configurations and optional pre-computation caching.
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
            pixel_threshold (int): Threshold for significant motion detection
            delta (int): Frame step for motion calculation
            window_size (int): Number of frames to consider for motion detection
            cache_dir (str | Path, optional): Directory for caching pre-computed masks
        """
        self.pixel_threshold = pixel_threshold
        self.delta = delta
        self.window_size = window_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self, 
        frames: Union[List[np.ndarray], torch.Tensor],
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
    
    def generate_batch(
        self, 
        frame_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate motion masks for a batch of frames.

        Args:
            frame_batch: Batch of frame sequences

        Returns:
            Batch of motion masks
        """
        return generate_motion_mask_batch(
            frame_batch, 
            self.pixel_threshold, 
            self.delta
        )


def precompute_motion_masks(
    dataset_path: Union[str, Path],
    output_dir: Union[str, Path],
    pixel_threshold: int = 15,
    delta: int = 1,
    window_size: int = 5,
    max_clips_per_game: int = 3
) -> None:
    """
    Pre-compute motion masks for all clips in the dataset.
    
    Useful for faster training by pre-computing motion masks instead of
    generating them on-the-fly during data loading.

    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save pre-computed motion masks
        pixel_threshold: Threshold for motion detection
        delta: Frame step for motion calculation
        window_size: Number of frames to consider
        max_clips_per_game: Maximum clips to process per game

    Examples:
        >>> precompute_motion_masks(
        ...     dataset_path="Dataset",
        ...     output_dir="motion_masks_cache",
        ...     pixel_threshold=15,
        ...     window_size=5
        ... )
    """
    from PIL import Image
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = MotionMaskGenerator(
        pixel_threshold=pixel_threshold,
        delta=delta,
        window_size=window_size
    )
    
    processed_count = 0
    
    for game_dir in dataset_path.iterdir():
        if not game_dir.is_dir() or not game_dir.name.startswith('game'):
            continue
        
        game_name = game_dir.name
        clip_dirs = [d for d in game_dir.iterdir() if d.is_dir() and d.name.startswith('Clip')]
        clip_dirs.sort(key=lambda x: int(x.name.replace('Clip', '')))
        
        for clip_dir in clip_dirs[:max_clips_per_game]:
            clip_name = clip_dir.name
            cache_key = f"{game_name}_{clip_name}"
            
            # Check if already processed
            cache_path = output_dir / f"{cache_key}_motion.npy"
            if cache_path.exists():
                continue
            
            # Load frames
            frames = []
            frame_files = sorted([f for f in clip_dir.iterdir() if f.suffix == '.jpg'])
            
            for frame_file in frame_files[:window_size]:
                try:
                    img = Image.open(frame_file)
                    frames.append(np.array(img))
                except Exception as e:
                    print(f"Error loading {frame_file}: {e}")
                    break
            
            if len(frames) >= 2:
                # Generate and save motion mask
                motion_mask = generator.generate(frames, cache_key)
                np.save(cache_path, motion_mask)
                processed_count += 1
                print(f"Processed {game_name}/{clip_name}: {len(frames)} frames")
    
    print(f"Pre-computed motion masks for {processed_count} clips")


class MotionConfig:
    """
    Configuration class for motion detection parameters.
    
    Provides flexible configuration options based on motion analysis findings.
    """
    
    def __init__(
        self,
        pixel_threshold: int = 15,
        delta: int = 1,
        window_size: int = 5,
        adaptive_threshold: bool = False,
        speed_based_threshold: bool = False,
        min_motion_pixels: int = 100,
        max_motion_pixels: int = 50000,
        blur_kernel_size: int = 3,
        morphological_ops: bool = False
    ):
        """
        Initialize motion configuration.

        Args:
            pixel_threshold: Base threshold for motion detection
            delta: Frame step for motion calculation
            window_size: Number of frames to consider
            adaptive_threshold: Enable adaptive threshold based on frame content
            speed_based_threshold: Adjust threshold based on estimated ball speed
            min_motion_pixels: Minimum motion pixels to consider valid
            max_motion_pixels: Maximum motion pixels to avoid noise
            blur_kernel_size: Kernel size for motion mask blurring
            morphological_ops: Apply morphological operations to clean mask
        """
        self.pixel_threshold = pixel_threshold
        self.delta = delta
        self.window_size = window_size
        self.adaptive_threshold = adaptive_threshold
        self.speed_based_threshold = speed_based_threshold
        self.min_motion_pixels = min_motion_pixels
        self.max_motion_pixels = max_motion_pixels
        self.blur_kernel_size = blur_kernel_size
        self.morphological_ops = morphological_ops
    
    def get_effective_threshold(self, frame_stats: Optional[dict] = None) -> int:
        """
        Get effective threshold based on configuration and frame statistics.
        
        Args:
            frame_stats: Optional frame statistics for adaptive thresholding
            
        Returns:
            Effective threshold value
        """
        threshold = self.pixel_threshold
        
        if self.adaptive_threshold and frame_stats:
            # Adjust threshold based on frame brightness and contrast
            brightness = frame_stats.get('brightness', 128)
            contrast = frame_stats.get('contrast', 50)
            
            # Darker frames need lower thresholds
            if brightness < 100:
                threshold = max(5, threshold - 5)
            elif brightness > 200:
                threshold = min(30, threshold + 5)
            
            # High contrast frames can use higher thresholds
            if contrast > 80:
                threshold = min(30, threshold + 3)
        
        return threshold

    def __repr__(self) -> str:
        return (f"MotionConfig(pixel_threshold={self.pixel_threshold}, delta={self.delta}, "
                f"window_size={self.window_size}, adaptive_threshold={self.adaptive_threshold}, "
                f"speed_based_threshold={self.speed_based_threshold}, min_motion_pixels={self.min_motion_pixels}, "
                f"max_motion_pixels={self.max_motion_pixels}, blur_kernel_size={self.blur_kernel_size}, "
                f"morphological_ops={self.morphological_ops})")


def adaptive_motion_threshold(
    frames: Union[List[np.ndarray], torch.Tensor],
    base_threshold: int = 15,
    min_threshold: int = 5,
    max_threshold: int = 30
) -> int:
    """
    Calculate adaptive motion threshold based on frame characteristics.
    
    Based on motion analysis findings showing variable motion patterns across clips.

    Args:
        frames: Sequence of frames
        base_threshold: Base threshold value
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold

    Returns:
        Adaptive threshold value

    Examples:
        >>> frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
        >>> threshold = adaptive_motion_threshold(frames, base_threshold=15)
        >>> 5 <= threshold <= 30
        True
    """
    if not frames:
        return base_threshold
    
    # Convert to numpy if torch tensor
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Calculate frame statistics
    frame_stats = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray_frame = np.mean(frame, axis=2)
        else:
            gray_frame = frame
        
        brightness = np.mean(gray_frame)
        contrast = np.std(gray_frame)
        frame_stats.append({'brightness': brightness, 'contrast': contrast})
    
    # Calculate average statistics
    avg_brightness = np.mean([s['brightness'] for s in frame_stats])
    avg_contrast = np.mean([s['contrast'] for s in frame_stats])
    
    # Adjust threshold based on statistics
    threshold = base_threshold
    
    # Darker frames need lower thresholds
    if avg_brightness < 100:
        threshold -= 5
    elif avg_brightness > 200:
        threshold += 5
    
    # High contrast frames can use higher thresholds
    if avg_contrast > 80:
        threshold += 3
    elif avg_contrast < 30:
        threshold -= 3
    
    # Clamp to valid range
    return max(min_threshold, min(max_threshold, threshold))


def create_motion_visualization(
    frames: List[np.ndarray],
    motion_mask: np.ndarray,
    config: MotionConfig,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create comprehensive motion visualization.
    
    Generates visualization similar to the expected output with motion statistics.

    Args:
        frames: Sequence of input frames
        motion_mask: Generated motion mask
        config: Motion configuration used
        save_path: Optional path to save visualization

    Examples:
        >>> frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
        >>> motion_mask = generate_motion_mask(frames)
        >>> config = MotionConfig(pixel_threshold=15)
        >>> create_motion_visualization(frames, motion_mask, config, "motion_viz.png")
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tennis Ball Motion Analysis', fontsize=16)
    
    # 1. Original frame
    axes[0, 0].imshow(frames[0])
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # 2. Motion mask
    axes[0, 1].imshow(motion_mask, cmap='gray')
    axes[0, 1].set_title(f'Motion Mask (Threshold={config.pixel_threshold})')
    axes[0, 1].axis('off')
    
    # 3. Motion overlay
    axes[1, 0].imshow(frames[0])
    motion_overlay = np.zeros((*frames[0].shape[:2], 4))
    motion_overlay[:, :, 0] = motion_mask / 255.0  # Red channel
    motion_overlay[:, :, 3] = motion_mask / 255.0 * 0.5  # Alpha channel
    axes[1, 0].imshow(motion_overlay)
    axes[1, 0].set_title('Motion Overlay on Frame')
    axes[1, 0].axis('off')
    
    # 4. Motion statistics
    motion_pixels = np.sum(motion_mask > 0)
    total_pixels = motion_mask.size
    motion_percentage = (motion_pixels / total_pixels) * 100
    
    stats_text = [
        'Motion Statistics:',
        f'Motion pixels: {motion_pixels:,}',
        f'Total pixels: {total_pixels:,}',
        f'Motion %: {motion_percentage:.2f}%',
        f'Frames processed: {len(frames)}',
        f'Threshold: {config.pixel_threshold}',
        f'Window size: {config.window_size}',
        f'Delta: {config.delta}',
        f'Adaptive: {config.adaptive_threshold}',
        f'Min motion: {config.min_motion_pixels}',
        f'Max motion: {config.max_motion_pixels}'
    ]
    
    for i, text in enumerate(stats_text):
        weight = 'bold' if i == 0 else 'normal'
        axes[1, 1].text(0.05, 0.9 - i*0.08, text, fontsize=10, fontweight=weight, 
                       transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Motion visualization saved to {save_path}")
    
    plt.show()


# Enhanced MotionMaskGenerator with flexible configuration
class FlexibleMotionMaskGenerator(MotionMaskGenerator):
    """
    Enhanced motion mask generator with flexible configuration options.
    
    Extends the base MotionMaskGenerator with adaptive thresholding and
    advanced motion detection features based on analysis findings.
    """
    
    def __init__(self, config: MotionConfig, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize flexible motion mask generator.

        Args:
            config: Motion configuration object
            cache_dir: Optional directory for caching
        """
        super().__init__(
            pixel_threshold=config.pixel_threshold,
            delta=config.delta,
            window_size=config.window_size,
            cache_dir=cache_dir
        )
        self.config = config
    
    def generate_enhanced(
        self, 
        frames: Union[List[np.ndarray], torch.Tensor],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate enhanced motion mask with flexible configuration.

        Args:
            frames: Sequence of frames
            cache_key: Optional key for caching

        Returns:
            Enhanced motion mask
        """
        # Check cache first
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_enhanced_motion.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Convert to numpy if torch tensor
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # Get effective threshold
        if self.config.adaptive_threshold:
            frame_stats = self._calculate_frame_stats(frames)
            effective_threshold = self.config.get_effective_threshold(frame_stats)
        else:
            effective_threshold = self.config.pixel_threshold
        
        # Generate base motion mask
        motion_mask = generate_motion_mask(
            frames, 
            effective_threshold, 
            self.config.delta, 
            self.config.window_size
        )
        
        # Apply enhancements
        motion_mask = self._apply_enhancements(motion_mask)
        
        # Save to cache
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}_enhanced_motion.npy"
            np.save(cache_path, motion_mask)
        
        return motion_mask
    
    def _calculate_frame_stats(self, frames: List[np.ndarray]) -> dict:
        """Calculate frame statistics for adaptive thresholding."""
        if not frames:
            return {}
        
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray_frame = np.mean(frame, axis=2)
            else:
                gray_frame = frame
            gray_frames.append(gray_frame)
        
        brightness = np.mean([np.mean(gf) for gf in gray_frames])
        contrast = np.mean([np.std(gf) for gf in gray_frames])
        
        return {'brightness': brightness, 'contrast': contrast}
    
    def _apply_enhancements(self, motion_mask: np.ndarray) -> np.ndarray:
        """Apply enhancements to motion mask."""
        # Check motion pixel count
        motion_pixels = np.sum(motion_mask > 0)
        
        if motion_pixels < self.config.min_motion_pixels:
            # Too little motion, might be noise
            return np.zeros_like(motion_mask)
        
        if motion_pixels > self.config.max_motion_pixels:
            # Too much motion, might be camera shake
            # Apply more aggressive thresholding
            motion_mask = self._reduce_motion_noise(motion_mask)
        
        # Apply morphological operations if enabled
        if self.config.morphological_ops:
            motion_mask = self._apply_morphological_ops(motion_mask)
        
        return motion_mask
    
    def _reduce_motion_noise(self, motion_mask: np.ndarray) -> np.ndarray:
        """Reduce motion noise by applying additional filtering."""
        # Apply Gaussian blur to reduce noise
        if self.config.blur_kernel_size > 0:
            import cv2
            motion_mask = cv2.GaussianBlur(motion_mask, 
                                         (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)
            motion_mask = (motion_mask > 127).astype(np.uint8) * 255
        
        return motion_mask
    
    def _apply_morphological_ops(self, motion_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean motion mask."""
        import cv2
        
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small noise
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        return motion_mask
