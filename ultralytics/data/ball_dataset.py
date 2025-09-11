# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Format
from ultralytics.nn.modules.motion_utils import (
    MotionConfig,
    FlexibleMotionMaskGenerator,
    generate_motion_mask,
    combine_rgb_motion
)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.instance import Instances

class TennisBallTransform:
    """
    Custom transform for tennis ball dataset with 4-channel input support.
    
    This transform handles the conversion from 4-channel numpy arrays to
    properly formatted torch tensors for the model.
    """
    
    def __init__(self):
        """Initialize the transform."""
        pass
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Transform a 4-channel image to properly formatted numpy array.
        
        Args:
            img: 4-channel image as numpy array (H, W, 4)
            
        Returns:
            Transformed image as numpy array (H, W, 4)
        """
        # Ensure image is contiguous and in correct format
        img = np.ascontiguousarray(img)
        
        # Ensure float32 and normalize to [0, 1] if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype != np.float32:
            img = img.astype(np.float32)
        
        return img


__all__ = (
    "TennisBallDataset",
    "TennisBallTransform",
    "MotionDataLoader", 
    "BallDataTransforms",
    "create_ball_dataloader"
)


class TennisBallDataset(YOLODataset):
    """
    Dataset class for tennis ball tracking with 4-channel input (RGB + motion mask).
    
    Handles multiple games and clips with proper sequential frame loading for motion detection.
    Extends YOLODataset to support motion mask generation and 4-channel input
    for enhanced tennis ball tracking using YOLO11Pose adaptation.

    Attributes:
        motion_config (MotionConfig): Configuration for motion mask generation
        motion_generator (FlexibleMotionMaskGenerator): Motion mask generator
        use_motion_masks (bool): Whether to generate motion masks
        motion_cache_dir (Optional[Path]): Directory for caching motion masks
        frame_window_size (int): Number of frames to use for motion detection
        precompute_motion (bool): Whether to pre-compute motion masks
        clip_frame_map (Dict): Mapping from frame paths to their clip sequences
        game_clips (Dict): Organized structure of games and clips

    Methods:
        build_clip_structure: Build organized structure of games, clips and frames
        load_motion_mask: Load or generate motion mask for a frame sequence
        get_4channel_input: Get 4-channel input (RGB + motion mask)
        cache_motion_masks: Pre-compute and cache motion masks
        get_labels: Override to include motion mask information

    Examples:
        >>> from ultralytics.data.ball_dataset import TennisBallDataset
        >>> dataset = TennisBallDataset(
        ...     img_path="Dataset",  # Now points to root Dataset directory
        ...     data={"names": {0: "tennis_ball"}, "kpt_shape": [1, 3]},
        ...     task="pose",
        ...     motion_config=MotionConfig(pixel_threshold=15)
        ... )
        >>> batch = next(iter(dataset))
        >>> print(batch["img"].shape)  # Should be (4, H, W) for 4-channel input
    """

    def __init__(
        self,
        *args,
        data: Dict | None = None,
        task: str = "pose",
        motion_config: Optional[MotionConfig] = None,
        use_motion_masks: bool = True,
        motion_cache_dir: Optional[Union[str, Path]] = None,
        frame_window_size: int = 5,
        precompute_motion: bool = False,
        **kwargs
    ):
        """
        Initialize TennisBallDataset.

        Args:
            *args: Positional arguments for parent YOLODataset
            data: Dataset configuration dictionary
            task: Task type (should be "pose" for tennis ball tracking)
            motion_config: Configuration for motion mask generation
            use_motion_masks: Whether to generate motion masks
            motion_cache_dir: Directory for caching motion masks
            frame_window_size: Number of frames to use for motion detection
            precompute_motion: Whether to pre-compute motion masks
            **kwargs: Additional keyword arguments for parent class
        """
        # Extract custom parameters from kwargs
        # TODO: dataset_path is not used, should be removed.
        dataset_path = kwargs.pop('dataset_path', None)
        
        # Set motion-related attributes before parent initialization
        self.use_motion_masks = use_motion_masks
        self.motion_cache_dir = Path(motion_cache_dir) if motion_cache_dir else None
        self.frame_window_size = frame_window_size
        self.precompute_motion = precompute_motion
        
        # Motion detection configuration
        if motion_config is not None:
            LOGGER.warning(f"{colorstr('TennisBallDataset')}: Overriding default MotionConfig with user-provided config.")
        self.motion_config = motion_config or MotionConfig(
            pixel_threshold=15,
            delta=1,
            window_size=frame_window_size,
            adaptive_threshold=True,
            min_motion_pixels=100,
            max_motion_pixels=50000
        )
        # Initialize motion generator
        self.motion_generator = FlexibleMotionMaskGenerator(
            self.motion_config,
            cache_dir=self.motion_cache_dir
        )
        
        # Ensure data is not None
        if data is None:
            data = {}
        
        # Initialize clip structure mappings
        self.clip_frame_map = {}
        self.game_clips = {}
        
        # Initialize parent class
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Initializing with motion_masks={self.use_motion_masks}")
        super().__init__(*args, data=data, task=task, **kwargs)
        
        # Build clip structure after parent initialization
        self.build_clip_structure()
        
        # Check if transforms were set up
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Transforms type: {type(self.transforms)}")
        
        # Store dataset path
        self.dataset_path = dataset_path
        
        # Override channels for 4-channel input
        if self.use_motion_masks:
            self.channels = 4  # RGB + motion mask
            LOGGER.info(f"{colorstr('TennisBallDataset')}: Using 4-channel input (RGB + motion mask)")
        else:
            self.channels = 3  # RGB only
            LOGGER.info(f"{colorstr('TennisBallDataset')}: Using 3-channel input (RGB only)")
        
        # Pre-compute motion masks if requested
        # TODO: this line seems useless, it do nothing, the cache should follow ultralytics original logic.
        if self.precompute_motion and self.use_motion_masks:
            self.cache_motion_masks()

    def build_clip_structure(self):
        """
        Build organized structure of games, clips and frames for proper motion mask generation.
        
        This creates a mapping that allows us to find the sequential frames within each clip
        for proper motion mask generation.
        """
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Building clip structure for motion detection...")
        
        # Group image files by their clip directories
        for img_path in self.im_files:
            img_path = Path(img_path)
            
            # Extract game and clip info from path structure
            # Expected structure: .../Dataset/gameX/ClipY/frame.jpg or .../Dataset_YOLO/images/train|val/gameX_ClipY_frame.jpg
            # TODO: This is the magic word for processed or not, it should be considered more carefully.
            if "Dataset_YOLO" in str(img_path):
                # Handle converted YOLO format: gameX_ClipY_frame.jpg
                parts = img_path.stem.split('_')
                if len(parts) >= 3:
                    game = parts[0]  # gameX
                    clip = parts[1]  # ClipY
                    frame = '_'.join(parts[2:])  # frame number
                    
                    clip_key = f"{game}_{clip}"
                    if clip_key not in self.game_clips:
                        self.game_clips[clip_key] = []
                    
                    self.game_clips[clip_key].append(img_path)
                    self.clip_frame_map[str(img_path)] = {
                        'game': game,
                        'clip': clip,
                        'frame': frame,
                        'clip_key': clip_key
                    }
            else:
                # Handle original Dataset format: Dataset/gameX/ClipY/frame.jpg
                path_parts = img_path.parts
                if len(path_parts) >= 3:
                    # Find Dataset, game, and clip in path
                    try:
                        dataset_idx = next(i for i, part in enumerate(path_parts) if 'dataset' in part.lower())
                        if dataset_idx + 2 < len(path_parts):
                            game = path_parts[dataset_idx + 1]  # gameX
                            clip = path_parts[dataset_idx + 2]  # ClipY
                            
                            clip_key = f"{game}_{clip}"
                            if clip_key not in self.game_clips:
                                self.game_clips[clip_key] = []
                            
                            self.game_clips[clip_key].append(img_path)
                            self.clip_frame_map[str(img_path)] = {
                                'game': game,
                                'clip': clip,
                                'frame': img_path.stem,
                                'clip_key': clip_key
                            }
                    except (StopIteration, IndexError):
                        LOGGER.warning(f"Could not parse path structure for: {img_path}")
        
        # Sort frames within each clip for proper sequential order
        for clip_key in self.game_clips:
            self.game_clips[clip_key].sort(key=lambda x: x.stem)
        
        total_clips = len(self.game_clips)
        total_frames = sum(len(frames) for frames in self.game_clips.values())
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Found {total_clips} clips with {total_frames} total frames")
        
        # Log clip statistics
        for clip_key, frames in self.game_clips.items():
            LOGGER.info(f"  {clip_key}: {len(frames)} frames")
    
    def load_motion_mask(
        self, 
        frame_paths: List[Path], 
        current_frame_idx: int
    ) -> np.ndarray:
        """
        Load or generate motion mask for a frame sequence within a clip.

        Args:
            frame_paths: List of frame file paths in sequential order within a clip
            current_frame_idx: Index of current frame in the sequence

        Returns:
            Motion mask as numpy array (H, W)
        """
        if not self.use_motion_masks:
            # Return empty mask if motion masks are disabled
            return np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        
        # Determine frame window for motion detection
        start_idx = max(0, current_frame_idx - self.frame_window_size // 2)
        end_idx = min(len(frame_paths), start_idx + self.frame_window_size)
        
        # Ensure we have at least 2 frames for motion detection
        if end_idx - start_idx < 2:
            # Expand window to get minimum frames
            if current_frame_idx == 0:
                # At beginning, use frames ahead
                end_idx = min(len(frame_paths), 2)
                start_idx = 0
            elif current_frame_idx == len(frame_paths) - 1:
                # At end, use frames behind
                start_idx = max(0, len(frame_paths) - 2)
                end_idx = len(frame_paths)
            else:
                # In middle, ensure we have at least 2 frames
                start_idx = max(0, current_frame_idx - 1)
                end_idx = min(len(frame_paths), current_frame_idx + 2)
        
        # Load frame sequence
        frames = []
        valid_indices = []
        for i in range(start_idx, end_idx):
            if i < len(frame_paths) and frame_paths[i].exists():
                try:
                    img = Image.open(frame_paths[i])
                    frames.append(np.array(img))
                    valid_indices.append(i)
                except Exception as e:
                    LOGGER.warning(f"Failed to load frame {frame_paths[i]}: {e}")
                    continue
        
        if len(frames) < 2:
            # Not enough frames for motion detection
            LOGGER.warning(f"Only {len(frames)} frames available for motion detection in clip")
            return np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        
        # Generate motion mask using frame sequence
        try:
            current_frame_path = frame_paths[current_frame_idx]
            cache_key = f"{current_frame_path.parent.name}_{current_frame_path.stem}_motion"
            motion_mask = self.motion_generator.generate_enhanced(frames, cache_key)
        except Exception as e:
            LOGGER.warning(f"Failed to generate motion mask for {frame_paths[current_frame_idx]}: {e}")
            return np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        
        # Resize motion mask to match target image size
        # TODO: The resize should be down in transforms.
        # if motion_mask.shape != (self.imgsz, self.imgsz):
        #     import cv2
        #     motion_mask = cv2.resize(motion_mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        
        return motion_mask
    
    def get_4channel_input(
        self, 
        rgb_image: np.ndarray, 
        motion_mask: np.ndarray
    ) -> np.ndarray:
        """
        Combine RGB image with motion mask to create 4-channel input.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            motion_mask: Motion mask as numpy array (H, W)

        Returns:
            4-channel input as numpy array (H, W, 4)
        """
        if not self.use_motion_masks:
            return rgb_image
        
        # Ensure motion mask is the same size as RGB image
        if motion_mask.shape[:2] != rgb_image.shape[:2]:
            motion_mask = cv2.resize(motion_mask, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Normalize motion mask to [0, 1]
        motion_mask_normalized = motion_mask.astype(np.float32) / 255.0
        
        # Combine RGB and motion mask
        four_channel = np.concatenate([rgb_image, motion_mask_normalized[..., None]], axis=-1)
        
        return four_channel
    
    def cache_motion_masks(self) -> None:
        """Pre-compute and cache motion masks for all frames organized by clips."""
        if not self.use_motion_masks or not self.motion_cache_dir:
            return
        
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Pre-computing motion masks for all clips...")
        
        # Use the organized clip structure for caching
        total_cached = 0
        for clip_key, frame_paths in self.game_clips.items():
            LOGGER.info(f"Caching motion masks for {clip_key} ({len(frame_paths)} frames)...")
            
            for i, frame_path in enumerate(frame_paths):
                try:
                    motion_mask = self.load_motion_mask(frame_paths, i)
                    total_cached += 1
                    # Cache is handled by motion_generator.generate_enhanced()
                except Exception as e:
                    LOGGER.warning(f"Failed to cache motion mask for {frame_path}: {e}")
        
        LOGGER.info(f"{colorstr('TennisBallDataset')}: Motion mask caching completed for {total_cached} frames")
    
    def get_labels(self) -> List[Dict[str, Any]]:
        """
        Get labels with motion mask information.

        Returns:
            List of label dictionaries with motion mask data
        """
        labels = super().get_labels()
        
        # Don't load motion masks during initialization for performance
        # They will be loaded on-demand during __getitem__
        if not self.use_motion_masks:
            return labels
        
        # Just mark that motion masks will be available
        for label in labels:
            label["motion_mask"] = None  # Will be loaded on demand
            label["has_motion"] = True   # Assume motion exists, will be calculated on demand
        
        return labels
    
    def load_image(self, i: int, rect_mode: bool = True) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Load image and motion mask.

        Args:
            i: Image index
            rect_mode: Whether to use rectangular mode

        Returns:
            Tuple of (image, ori_shape, resized_shape)
        """
        # Load RGB image from parent
        rgb_image, ori_shape, resized_shape = super().load_image(i, rect_mode)
        
        if not self.use_motion_masks:
            return rgb_image, ori_shape, resized_shape
        
        # Get frame info from clip structure
        im_file_path = str(self.im_files[i])
        
        if im_file_path in self.clip_frame_map:
            # Use clip structure for sequential frame loading
            clip_info = self.clip_frame_map[im_file_path]
            clip_key = clip_info['clip_key']
            clip_frames = self.game_clips[clip_key]
            
            try:
                current_idx = clip_frames.index(Path(im_file_path))
                motion_mask = self.load_motion_mask(clip_frames, current_idx)
            except (ValueError, IndexError):
                motion_mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        else:
            # Fallback to directory-based loading
            im_file = Path(self.im_files[i])
            clip_dir = im_file.parent
            frame_files = sorted([f for f in clip_dir.glob("*.jpg")])
            
            try:
                current_idx = frame_files.index(im_file)
                motion_mask = self.load_motion_mask(frame_files, current_idx)
            except (ValueError, IndexError):
                motion_mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        
        # Resize motion mask to match RGB image dimensions
        if motion_mask.shape[:2] != rgb_image.shape[:2]:
            import cv2
            motion_mask = cv2.resize(motion_mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Combine RGB and motion mask
        combined_image = combine_rgb_motion(rgb_image, motion_mask)
        
        # Ensure it's numpy array and apply our custom transform
        if isinstance(combined_image, torch.Tensor):
            combined_image = combined_image.numpy()
        
        transform = TennisBallTransform()
        combined_image = transform(combined_image)
        
        return combined_image, ori_shape, resized_shape
    
    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        """
        Get image and label with 4-channel input support.

        Args:
            index: Image index

        Returns:
            Dictionary containing image, label, and motion information
        """
        # Get base image and label
        result = super().get_image_and_label(index)
        
        if not self.use_motion_masks:
            return result
        
        # Load RGB image and motion mask
        rgb_image, ori_shape, resized_shape = self.load_image(index)
        
        # The load_image method already returns the combined 4-channel image
        # Update result with 4-channel input
        result["img"] = rgb_image
        result["ori_shape"] = ori_shape
        result["resized_shape"] = resized_shape
        
        return result


class BallDataTransforms:
    """
    Motion-aware data augmentation pipeline for tennis ball tracking.
    
    Provides augmentation transforms that preserve temporal consistency
    between RGB images and motion masks.
    """
    
    def __init__(
        self,
        imgsz: int = 640,
        augment: bool = True,
        motion_aware: bool = True
    ):
        """
        Initialize ball data transforms.

        Args:
            imgsz: Target image size
            augment: Whether to apply augmentation
            motion_aware: Whether to use motion-aware augmentation
        """
        self.imgsz = imgsz
        self.augment = augment
        self.motion_aware = motion_aware
    
    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to labels.

        Args:
            labels: Label dictionary

        Returns:
            Transformed labels
        """
        if not self.augment:
            return labels
        
        # Apply motion-aware augmentation
        if self.motion_aware and "motion_mask" in labels:
            labels = self._apply_motion_aware_augmentation(labels)
        
        return labels
    
    def _apply_motion_aware_augmentation(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply motion-aware augmentation that preserves temporal consistency.

        Args:
            labels: Label dictionary

        Returns:
            Augmented labels
        """
        # This would implement motion-aware augmentation
        # For now, return labels as-is
        # TODO: Implement motion-aware augmentation pipeline
        return labels
    


class MotionDataLoader:
    """
    Custom DataLoader for tennis ball tracking with 4-channel input support.
    
    Provides efficient loading and batching of 4-channel inputs with
    motion mask information.
    """
    
    def __init__(
        self,
        dataset: TennisBallDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True
    ):
        """
        Initialize MotionDataLoader.

        Args:
            dataset: TennisBallDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Create DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for 4-channel input batches.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        # Use parent collate function for standard fields
        collated = self.dataset.collate_fn(batch)
        
        # Handle motion-specific fields
        if "motion_mask" in batch[0]:
            motion_masks = [sample["motion_mask"] for sample in batch]
            collated["motion_mask"] = torch.stack([torch.from_numpy(mask) for mask in motion_masks])
        
        if "has_motion" in batch[0]:
            has_motion = [sample["has_motion"] for sample in batch]
            collated["has_motion"] = torch.tensor(has_motion, dtype=torch.bool)
        
        return collated
    
    def __iter__(self):
        """Iterate over the dataloader."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get dataloader length."""
        return len(self.dataloader)


def create_ball_dataloader(
    dataset: TennisBallDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 8,
    **kwargs
) -> MotionDataLoader:
    """
    Create a tennis ball dataloader with 4-channel input support.

    Args:
        img_path: Path to images directory
        data: Dataset configuration dictionary
        task: Task type (should be "pose")
        motion_config: Motion detection configuration
        batch_size: Batch size
        imgsz: Image size
        augment: Whether to apply augmentation
        **kwargs: Additional arguments for dataset

    Returns:
        MotionDataLoader instance

    Examples:
        >>> from ultralytics.data.ball_dataset import create_ball_dataloader
        >>> dataloader = create_ball_dataloader(
        ...     img_path="Dataset/game1/Clip1",
        ...     data={"names": {0: "tennis_ball"}, "kpt_shape": [1, 3]},
        ...     batch_size=8
        ... )
        >>> for batch in dataloader:
        ...     print(batch["img"].shape)  # (8, 4, 640, 640)
        ...     break
    """
    # Create dataloader
    return MotionDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
