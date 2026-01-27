from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.augment import LetterBox
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import IMG_FORMATS, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.stereo3ddet.augment import (
    StereoAugmentationPipeline,
    StereoCalibration,
    stereo3d_transforms,
)
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.data.stereo.target_improved import TargetGenerator
import math

def _to_hw(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize imgsz to (H, W)."""
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        return int(imgsz[0]), int(imgsz[1])
    raise TypeError(f"imgsz must be int or (h,w), got {imgsz} ({type(imgsz).__name__})")


class Stereo3DDetDataset(BaseDataset):
    """Stereo 3D detection dataset (single, unified dataset).

    - Returns keys: 'img' (6-channel), 'targets' (dict), 'im_file', 'ori_shape'.
    - Builds a single 6-channel tensor by stacking left and right RGB after identical letterbox.
    - Converts normalized left 2D boxes to resized+letterboxed normalized xywh.
    - Inherits from BaseDataset for label caching and image loading infrastructure.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        imgsz: int | tuple[int, int] | list[int],
        names: Dict[int, str] | List[str] | None = None,
        max_samples: int | None = None,
        output_size: Tuple[int, int] | None = None,
        mean_dims: Dict[str, List[float]] | None = None,
        std_dims: Dict[str, List[float]] | None = None,
        filter_occluded: bool = True,
        max_occlusion_level: int = 1,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        data: dict | None = None,
    ):
        """Initialize Stereo3DDetDataset.

        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split ('train' or 'val').
            imgsz (int): Target image size for letterboxing.
            names (Dict[int, str] | List[str] | None): Class names mapping. If None, uses default.
            max_samples (int | None): Maximum number of samples to load. If None, loads all available samples.
                If specified, only the first max_samples samples will be loaded. Defaults to None.
            output_size (Tuple[int, int] | None): Output feature map size (H, W) for target generation.
                If None, computed from imgsz assuming 8x downsampling (P3 architecture).
            mean_dims (Dict[str, List[float]] | None): Mean dimensions per class [L, W, H] in meters.
                If None, uses default KITTI values.
            std_dims (Dict[str, List[float]] | None): Standard deviation of dimensions per class [L, W, H] in meters.
                Used for normalized offset prediction. If None, defaults to reasonable estimates.
            filter_occluded (bool): Whether to filter out heavily occluded objects during training.
                If True, objects with occlusion level > max_occlusion_level are excluded from training.
                Defaults to False.
            max_occlusion_level (int): Maximum occlusion level to include when filter_occluded is True.
                KITTI occlusion levels: 0=fully visible, 1=partially occluded, 2=heavily occluded, 3=unknown.
                Default is 1 (exclude heavily occluded and unknown objects).
            cache: Cache images to RAM or disk.
            augment: Whether to apply augmentation.
            hyp: Hyperparameters dict.
            prefix: Prefix for log messages.
            data: Dataset configuration dict.
        """
        self.root = Path(root)
        self.split = split
        self.imgsz_tuple = _to_hw(imgsz)  # (H, W) - stored separately for transforms
        # BaseDataset expects imgsz to be an integer for load_image() operations
        # For non-square imgsz (e.g., 384,1248), use max dimension for BaseDataset compatibility.
        # BaseDataset.load_image() will resize long side to this value, then LetterBox transform
        # will handle the final non-square sizing (e.g., to 384x1248).
        imgsz_int = max(self.imgsz_tuple) if isinstance(imgsz, (tuple, list)) else imgsz
        self.names = names or {}
        self.data = data or {}
        self.filter_occluded = filter_occluded
        self.max_occlusion_level = max_occlusion_level

        # Dataset layout (KITTI-style stereo):
        # - images/{split}/left, images/{split}/right
        # - labels/{split}
        # - calib/{split}
        self.left_dir = self.root / "images" / split / "left"
        self.right_dir = self.root / "images" / split / "right"
        self.label_dir = self.root / "labels" / split
        self.calib_dir = self.root / "calib" / split

        if not self.left_dir.exists():
            raise FileNotFoundError(f"Left image directory not found: {self.left_dir}")
        if not self.right_dir.exists():
            raise FileNotFoundError(f"Right image directory not found: {self.right_dir}")
        if not self.calib_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # Get image IDs first (before calling super().__init__)
        self.image_ids = self._get_image_ids()
        if len(self.image_ids) == 0:
            raise ValueError(f"No stereo pairs found in {self.left_dir} (missing/empty)")

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            total = len(self.image_ids)
            self.image_ids = self.image_ids[:max_samples]
            if len(self.image_ids) < total:
                LOGGER.info(f"Limited stereo dataset to {len(self.image_ids)} samples (from {total} total)")

        # Occlusion filtering settings
        if filter_occluded:
            LOGGER.info(
                f"Occlusion filtering enabled: excluding objects with occlusion level > {max_occlusion_level} "
                f"(0=visible, 1=partial, 2=heavy, 3=unknown)"
            )

        # Initialize BaseDataset with left image directory as img_path
        # BaseDataset will call get_img_files() and get_labels()
        # BaseDataset expects imgsz to be an integer for load_image() operations
        super().__init__(
            img_path=str(self.left_dir),
            imgsz=imgsz_int,  # Use integer for BaseDataset compatibility
            cache=cache,
            augment=augment,
            hyp=hyp,
            prefix=prefix,
            channels=3,  # RGB images
        )

        # Keep imgsz as integer for BaseDataset operations
        # Use imgsz_tuple when tuple format is needed (e.g., in transforms)

        # Initialize target generator for generating training/validation targets
        # Compute output_size if not provided (default to 8x downsampling for P3)
        if output_size is None:
            # Default to 8x downsampling (P3 architecture). This can be overridden by the trainer if model is known.
            input_h, input_w = self.imgsz_tuple  # Use tuple format
            output_size = (input_h // 8, input_w // 8)
        self.output_size = output_size

        # Get number of classes
        num_classes = len(self.names) if self.names else 3

        # Initialize target generator
        self.target_generator = TargetGenerator(
            output_size=output_size,
            num_classes=num_classes,
            mean_dims=mean_dims,
            std_dims=std_dims,  # std_dims will be loaded from dataset YAML if available
            class_names=self.names,  # Pass class names mapping for dataset-agnostic operation
        )

    def _get_image_ids(self) -> list[str]:
        """Return image ids that exist in left/right dirs AND have required metadata files.

        This prevents runtime errors where an image exists but its calibration file is missing.
        """
        left_files = sorted(f for f in self.left_dir.glob("*.*") if f.suffix[1:].lower() in IMG_FORMATS)

        if not left_files:
            raise FileNotFoundError(f"No image files found in {self.left_dir}")

        image_ids: list[str] = []
        for lf in left_files:
            rf = self.right_dir / lf.name
            image_id = lf.stem
            if not rf.exists():
                LOGGER.warning(f"Missing right image for {image_id}, skipping")
                continue

            calib_file = self.calib_dir / f"{image_id}.txt"
            if not calib_file.exists():
                LOGGER.warning(f"Missing calibration for {image_id}, skipping: {calib_file}")
                continue

            label_file = self.label_dir / f"{image_id}.txt"
            if not label_file.exists():
                LOGGER.warning(f"Missing label file for {image_id}, skipping: {label_file}")
                continue

            image_ids.append(image_id)
        return image_ids

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Override to return left image files filtered by image_ids."""
        # Get all left image files
        left_files = sorted(f for f in self.left_dir.glob("*.*") if f.suffix[1:].lower() in IMG_FORMATS)
        # Filter by image_ids
        im_files = [str(f) for f in left_files if f.stem in self.image_ids]
        return im_files

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load stereo image pair (left + right) from dataset index 'i'.
        
        Overrides BaseDataset.load_image() to load both left and right images,
        resize them identically, and concatenate into a 6-channel stereo image.
        Supports caching (RAM and disk) like BaseDataset.
        
        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.
            
        Returns:
            im (np.ndarray): 6-channel stereo image [H, W, 6] (left RGB + right RGB).
            hw_original (tuple[int, int]): Original image dimensions (height, width).
            hw_resized (tuple[int, int]): Resized image dimensions (height, width).
            
        Raises:
            FileNotFoundError: If the image file is not found.
        """
        from ultralytics.utils.patches import imread
        
        # TODO: We should also support cache for stereo images.
        # Check cache first
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        h0, w0 = None, None  # Will be set when loading from files
        
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                    # If we have cached original shape, use it; otherwise we'll need to load to get it
                    if self.im_hw0[i] is not None:
                        h0, w0 = self.im_hw0[i]
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    # Fall through to load from file
                    im = None
            
            if im is None:  # not loaded from cache, load from files
                # Load left image
                left_img = imread(f, flags=self.cv2_flag)  # BGR
                if left_img is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
                
                # Load right image
                image_id = Path(f).stem
                right_img_path = self.right_dir / Path(f).name
                if not right_img_path.exists():
                    # Try different extension
                    for ext in [".png", ".jpg", ".jpeg"]:
                        right_img_path = self.right_dir / f"{image_id}{ext}"
                        if right_img_path.exists():
                            break
                
                right_img = imread(str(right_img_path), flags=self.cv2_flag)  # BGR
                if right_img is None:
                    raise FileNotFoundError(f"Right image not found: {right_img_path}")
                
                # Get original shapes
                left_h0, left_w0 = left_img.shape[:2]
                right_h0, right_w0 = right_img.shape[:2]
                h0, w0 = left_h0, left_w0  # Store for return
                
                # Apply resize logic (same as BaseDataset.load_image())
                if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                    # Resize left image
                    r_left = self.imgsz / max(left_h0, left_w0)  # ratio
                    if r_left != 1:
                        w, h = (min(math.ceil(left_w0 * r_left), self.imgsz), min(math.ceil(left_h0 * r_left), self.imgsz))
                        left_img = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Resize right image to match left's resized shape
                    right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                elif not (left_h0 == left_w0 == self.imgsz):  # resize by stretching image to square imgsz
                    left_img = cv2.resize(left_img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                    right_img = cv2.resize(right_img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                
                # Handle grayscale
                if left_img.ndim == 2:
                    left_img = left_img[..., None]
                if right_img.ndim == 2:
                    right_img = right_img[..., None]
                
                # Convert BGR to RGB and concatenate to 6-channel stereo image
                left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                im = np.concatenate([left_rgb, right_rgb], axis=2)  # [H, W, 6]
            
            if im is None:
                raise FileNotFoundError(f"Failed to load stereo image pair for {f}")
            
            # If we loaded from npy cache and don't have original shape, use resized shape as fallback
            if h0 is None or w0 is None:
                h0, w0 = im.shape[:2]  # Fallback to resized shape
                # Store it for future use
                if self.im_hw0[i] is None:
                    self.im_hw0[i] = (h0, w0)
            
            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            
            return im, (h0, w0), im.shape[:2]
        
        # Return from cache
        cached_hw0 = self.im_hw0[i]
        cached_hw = self.im_hw[i]
        # If cached original shape is missing, use resized shape as fallback
        if cached_hw0 is None:
            cached_hw0 = cached_hw if cached_hw is not None else im.shape[:2]
        if cached_hw is None:
            cached_hw = im.shape[:2]
        return im, cached_hw0, cached_hw

    def cache_images_to_disk(self, i: int) -> None:
        """Save stereo image pair (6-channel) as an *.npy file for faster loading.
        
        Overrides BaseDataset.cache_images_to_disk() to handle 6-channel stereo images.
        
        Args:
            i (int): Index of the image to cache.
        """
        f = self.npy_files[i]
        if not f.exists():
            # Load the stereo image pair using load_image (which returns 6-channel)
            stereo_img, _, _ = self.load_image(i, rect_mode=True)
            # Save the 6-channel stereo image
            np.save(f.as_posix(), stereo_img, allow_pickle=False)

    def get_labels(self) -> list[dict[str, Any]]:
        """Load and cache stereo 3D labels.

        Returns:
            List of label dictionaries with stereo-specific format.
        """
        cache_path = Path(self.label_dir).parent / f"{self.prefix}stereo3d_{self.split}.cache"
        if cache_path.exists():
            try:
                cache = load_dataset_cache_file(cache_path)
                if cache.get("version") == "1.0.0" and len(cache.get("labels", [])) == len(self.im_files):
                    labels = cache["labels"]
                    LOGGER.info(f"{self.prefix}Loaded {len(labels)} labels from cache: {cache_path}")
                    return labels
            except Exception as e:
                LOGGER.warning(f"{self.prefix}Cache loading failed: {e}")

        # Parse labels
        labels = []
        for im_file in self.im_files:
            image_id = Path(im_file).stem
            label_file = self.label_dir / f"{image_id}.txt"
            calib_file = self.calib_dir / f"{image_id}.txt"

            # Load image to get shape
            left_img = cv2.imread(im_file)
            if left_img is None:
                LOGGER.warning(f"Could not load image: {im_file}")
                continue

            h, w = left_img.shape[:2]

            # Parse labels
            parsed_labels = self._parse_labels(label_file)
            # Parse calibration
            calib = self._parse_calibration(calib_file)

            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "labels": parsed_labels,
                    "calibration": calib,
                    "image_id": image_id,
                }
            )

        # Save cache
        try:
            save_dataset_cache_file(self.prefix, cache_path, {"labels": labels}, "1.0.0")
            LOGGER.info(f"{self.prefix}Saved {len(labels)} labels to cache: {cache_path}")
        except Exception as e:
            LOGGER.warning(f"{self.prefix}Cache saving failed: {e}")

        return labels

    def build_transforms(self, hyp: dict[str, Any] | None = None) -> Any:
        """Build stereo 3D augmentation pipeline.

        Args:
            hyp: Hyperparameters dict.

        Returns:
            Compose: Composed transform pipeline.
        """
        if hyp is None:
            hyp = DEFAULT_CFG

        if self.augment:
            return stereo3d_transforms(self, self.imgsz_tuple, hyp)  # Use tuple for transforms
        else:
            from ultralytics.data.augment import Compose
            from ultralytics.models.yolo.stereo3ddet.augment import StereoLetterBox, StereoFormat

            return Compose(
                [
                    StereoLetterBox(new_shape=self.imgsz_tuple, scaleup=False, stride=32),  # Use tuple
                    StereoFormat(normalize=True),
                ]
            )

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Convert stereo labels to Instances format with 3D data included.

        Args:
            label: Label dict from BaseDataset.get_image_and_label().
                  The 'img' key already contains the 6-channel stereo image from load_image().

        Returns:
            Updated label dict with Instances containing 3D data.
        """
        from ultralytics.utils.instance import Instances
        
        stereo_img = label["img"]  # Already 6-channel from load_image()
        
        # Verify it's 6-channel
        if stereo_img.shape[-1] != 6:
            raise ValueError(
                f"Expected 6-channel stereo image, but got {stereo_img.shape[-1]} channels. "
                f"Image shape: {stereo_img.shape}"
            )

        # Get labels and calibration from cached data
        im_file = label["im_file"]
        idx = self.im_files.index(im_file)
        cached_label = self.labels[idx]
        label_list = cached_label["labels"]  # List of dict objects
        label["calibration"] = cached_label["calibration"]
        
        # Convert to Instances format with 3D data included
        if len(label_list) == 0:
            # Empty case
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.int64)
            right_bboxes = np.zeros((0, 4), dtype=np.float32)
            dimensions_3d = np.zeros((0, 3), dtype=np.float32)
            location_3d = np.zeros((0, 3), dtype=np.float32)
            rotation_y = np.zeros((0,), dtype=np.float32)
            truncated = np.zeros((0,), dtype=np.float32)
            occluded = np.zeros((0,), dtype=np.int32)
            vertices = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            # Extract left boxes (normalized xywh) and class IDs
            bboxes = np.array([
                [obj["left_box"]["center_x"], obj["left_box"]["center_y"],
                 obj["left_box"]["width"], obj["left_box"]["height"]]
                for obj in label_list
            ], dtype=np.float32)
            cls = np.array([obj["class_id"] for obj in label_list], dtype=np.int64)
            
            # Extract right boxes
            right_bboxes = np.array([
                [obj["right_box"]["center_x"], obj["right_box"]["center_y"],
                 obj["right_box"]["width"], obj["right_box"]["height"]]
                for obj in label_list
            ], dtype=np.float32)
            
            # Extract 3D data
            dimensions_3d = np.array([
                [obj["dimensions"]["length"], obj["dimensions"]["width"], obj["dimensions"]["height"]]
                for obj in label_list
            ], dtype=np.float32)
            
            location_3d = np.array([
                [obj["location_3d"]["x"], obj["location_3d"]["y"], obj["location_3d"]["z"]]
                for obj in label_list
            ], dtype=np.float32)
            
            rotation_y = np.array([obj["rotation_y"] for obj in label_list], dtype=np.float32)
            
            # Extract additional metadata
            truncated = np.array([obj.get("truncated", 0.0) for obj in label_list], dtype=np.float32)
            occluded = np.array([obj.get("occluded", 0) for obj in label_list], dtype=np.int32)
            
            # Extract vertices if available
            vertices_list = []
            for obj in label_list:
                if "vertices" in obj and obj["vertices"]:
                    v = obj["vertices"]
                    vertices_list.append([
                        [v.get("v1", [0, 0])[0], v.get("v1", [0, 0])[1]],
                        [v.get("v2", [0, 0])[0], v.get("v2", [0, 0])[1]],
                        [v.get("v3", [0, 0])[0], v.get("v3", [0, 0])[1]],
                        [v.get("v4", [0, 0])[0], v.get("v4", [0, 0])[1]],
                    ])
                else:
                    vertices_list.append([[0, 0], [0, 0], [0, 0], [0, 0]])
            vertices = np.array(vertices_list, dtype=np.float32)
        
        # Create Instances with 3D data included (normalized xywh format)
        from ultralytics.utils import LOGGER
        if len(bboxes) > 0:
            LOGGER.info(f"[Stereo3DDetDataset] Creating Instances with {len(bboxes)} objects")
            LOGGER.info(f"[Stereo3DDetDataset] Initial left bboxes (xywh normalized): {bboxes}")
            if right_bboxes is not None and len(right_bboxes) > 0:
                LOGGER.info(f"[Stereo3DDetDataset] Initial right bboxes (xywh normalized): {right_bboxes}")
            LOGGER.info(f"[Stereo3DDetDataset] Image shape: {label.get('img', np.array([])).shape}")
            if "calibration" in label:
                calib = label["calibration"]
                LOGGER.info(f"[Stereo3DDetDataset] Calibration: fx={calib.get('fx', 0):.2f}, fy={calib.get('fy', 0):.2f}, "
                          f"cx={calib.get('cx', 0):.2f}, cy={calib.get('cy', 0):.2f}, "
                          f"width={calib.get('width', calib.get('image_width', 0))}, "
                          f"height={calib.get('height', calib.get('image_height', 0))}")
        
        label["instances"] = Instances(
            bboxes=bboxes,
            bbox_format="xywh",
            normalized=True,
            right_bboxes=right_bboxes,
            dimensions_3d=dimensions_3d,
            location_3d=location_3d,
            rotation_y=rotation_y,
        )
        label["cls"] = cls
        
        # Store stereo_labels for backward compatibility with collate_fn and metrics
        label["stereo_labels"] = label_list
        
        # Store additional metadata that's not part of Instances
        label["truncated"] = truncated
        label["occluded"] = occluded
        label["vertices"] = vertices

        return label

    def _parse_calibration(self, calib_file: Path) -> dict[str, Any]:
        """Parse calibration file into a structured dictionary.

        Supports both original KITTI format (P0..P3, R0_rect, Tr_*) and the simplified converted
        format (fx, fy, cx, cy, right_cx, right_cy, baseline, image_width, image_height).
        """
        with open(calib_file, "r") as f:
            lines = f.readlines()

        calib_dict: dict[str, Any] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value_str = parts[1].strip()

            if key in {"fx", "fy", "cx", "cy", "right_cx", "right_cy", "baseline"}:
                try:
                    calib_dict[key] = float(value_str)
                except ValueError:
                    continue
                continue
            if key in {"image_width", "image_height"}:
                try:
                    calib_dict[key] = int(float(value_str))
                except ValueError:
                    continue
                continue

            try:
                values = [float(x) for x in value_str.split()]
            except ValueError:
                continue

            if key == "P0":
                calib_dict["P0"] = np.array(values).reshape(3, 4)
            elif key == "P1":
                calib_dict["P1"] = np.array(values).reshape(3, 4)
            elif key == "P2":
                calib_dict["P2"] = np.array(values).reshape(3, 4)
            elif key == "P3":
                calib_dict["P3"] = np.array(values).reshape(3, 4)
            elif key == "R0_rect":
                calib_dict["R0_rect"] = np.array(values).reshape(3, 3)
            elif key == "Tr_velo_to_cam":
                calib_dict["Tr_velo_to_cam"] = np.array(values).reshape(3, 4)
            elif key == "Tr_imu_to_velo":
                calib_dict["Tr_imu_to_velo"] = np.array(values).reshape(3, 4)

        # Derive intrinsics from P2 if needed
        if "P2" in calib_dict and not all(k in calib_dict for k in ("fx", "fy", "cx", "cy")):
            P2 = calib_dict["P2"]
            calib_dict["fx"] = P2[0, 0]
            calib_dict["fy"] = P2[1, 1]
            calib_dict["cx"] = P2[0, 2]
            calib_dict["cy"] = P2[1, 2]

        if "P3" in calib_dict and not all(k in calib_dict for k in ("right_cx", "right_cy")):
            P3 = calib_dict["P3"]
            calib_dict["right_cx"] = P3[0, 2]
            calib_dict["right_cy"] = P3[1, 2]

        if "P2" in calib_dict and "P3" in calib_dict and "fx" in calib_dict and "baseline" not in calib_dict:
            P2 = calib_dict["P2"]
            P3 = calib_dict["P3"]
            fx = calib_dict["fx"]
            baseline = (P2[0, 3] - P3[0, 3]) / fx
            calib_dict["baseline"] = abs(baseline)

        return calib_dict

    def _parse_labels(self, label_file: Path) -> list[dict[str, Any]]:
        """Parse YOLO 3D label file (26-value format)."""
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels: list[dict[str, Any]] = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 26:
                    LOGGER.warning(f"Invalid label format in {label_file}: expected 26 values, got {len(parts)}")
                    continue

                values = [float(x) for x in parts]
                class_id = int(values[0])
                label_dict = {
                    "class_id": class_id,
                    "left_box": {
                        "center_x": values[1],
                        "center_y": values[2],
                        "width": values[3],
                        "height": values[4],
                    },
                    "right_box": {
                        "center_x": values[5],
                        "center_y": values[6],
                        "width": values[7],
                        "height": values[8],
                    },
                    "dimensions": {
                        "length": values[9],
                        "width": values[10],
                        "height": values[11],
                    },
                    "rotation_y": values[15],
                    "vertices": {
                        "v1": [values[16], values[17]],
                        "v2": [values[18], values[19]],
                        "v3": [values[20], values[21]],
                        "v4": [values[22], values[23]],
                    },
                    "location_3d": {"x": values[12], "y": values[13], "z": values[14]},
                    "truncated": float(values[24]),
                    "occluded": int(values[25]),
                }

                # Keep existing assertions for early data-quality feedback
                assert -np.pi <= label_dict["rotation_y"] <= np.pi, (
                    f"rotation_y is out of range: {label_dict['rotation_y']}"
                )
                labels.append(label_dict)
        return labels

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_occlusion_stats(self) -> Dict[str, Any]:
        """Get statistics about occlusion levels in the dataset.
        
        Returns:
            Dict with occlusion distribution statistics.
        """
        from collections import Counter
        
        occlusion_counts = Counter()
        total_objects = 0
        
        for image_id in self.image_ids:
            label_file = self.label_dir / f"{image_id}.txt"
            if not label_file.exists():
                continue
            
            labels = self._parse_labels(label_file)
            for lab in labels:
                occlusion = lab.get("occluded", 0)
                occlusion_counts[occlusion] += 1
                total_objects += 1
        
        # KITTI occlusion level descriptions
        occlusion_descriptions = {
            0: "fully visible",
            1: "partially occluded", 
            2: "heavily occluded",
            3: "unknown/completely blocked"
        }
        
        stats = {
            "total_objects": total_objects,
            "total_images": len(self.image_ids),
            "occlusion_distribution": {
                level: {
                    "count": occlusion_counts.get(level, 0),
                    "percentage": occlusion_counts.get(level, 0) / total_objects * 100 if total_objects > 0 else 0,
                    "description": occlusion_descriptions.get(level, "unknown")
                }
                for level in range(4)
            },
            "filter_config": {
                "filter_occluded": self.filter_occluded,
                "max_occlusion_level": self.max_occlusion_level
            }
        }
        
        if self.filter_occluded:
            # Calculate what would be filtered
            excluded_count = sum(
                occlusion_counts.get(level, 0) 
                for level in range(self.max_occlusion_level + 1, 4)
            )
            stats["objects_after_filtering"] = total_objects - excluded_count
            stats["objects_excluded"] = excluded_count
            stats["exclusion_percentage"] = excluded_count / total_objects * 100 if total_objects > 0 else 0
        
        return stats

    def _transform_labels_for_letterbox(
        self,
        labels: List[Dict[str, Any]],
        scale_w: float,
        scale_h: float,
        pad_left: int,
        pad_top: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Dict[str, Any]]:
        """Transform label coordinates from original image space to letterboxed space.
        
        Args:
            labels: List of label dictionaries with left_box, right_box, and optionally vertices.
            scale_w: Effective letterbox x-scale (matches actual integer resize width).
            scale_h: Effective letterbox y-scale (matches actual integer resize height).
            pad_left: Left padding added by letterbox.
            pad_top: Top padding added by letterbox.
            orig_w: Original image width (after augmentation, before letterbox).
            orig_h: Original image height (after augmentation, before letterbox).
            
        Returns:
            List of transformed label dictionaries.
        """
        transformed_labels = []
        
        for label in labels:
            new_label = dict(label)  # Copy label to avoid modifying original
            
            input_h, input_w = self.imgsz_tuple  # Use tuple format

            # Transform left_box
            if "left_box" in new_label:
                lb = new_label["left_box"]
                # Denormalize from original image
                cx_px = float(lb.get("center_x", 0)) * orig_w
                cy_px = float(lb.get("center_y", 0)) * orig_h
                w_px = float(lb.get("width", 0)) * orig_w
                h_px = float(lb.get("height", 0)) * orig_h
                
                # Apply letterbox scale (use effective per-axis scales to match actual cv2.resize integer output)
                cx_px = cx_px * scale_w
                cy_px = cy_px * scale_h
                w_px = w_px * scale_w
                h_px = h_px * scale_h
                
                # Add letterbox padding
                cx_px = cx_px + pad_left
                cy_px = cy_px + pad_top
                
                # Normalize to letterboxed image size (H, W)
                new_label["left_box"] = {
                    "center_x": float(cx_px / input_w),
                    "center_y": float(cy_px / input_h),
                    "width": float(w_px / input_w),
                    "height": float(h_px / input_h),
                }
            
            # Transform right_box
            if "right_box" in new_label:
                rb = new_label["right_box"]
                # Denormalize from original image
                rx_px = float(rb["center_x"]) * orig_w
                ry_px = float(rb["center_y"]) * orig_h
                rw_px = float(rb["width"]) * orig_w
                rh_px = float(rb["height"]) * orig_h

                # Apply letterbox scale (effective per-axis)
                rx_px = rx_px * scale_w
                ry_px = ry_px * scale_h
                rw_px = rw_px * scale_w
                rh_px = rh_px * scale_h

                rx_px = rx_px + pad_left
                ry_px = ry_px + pad_top
                # Normalize to letterboxed image size (H, W)
                new_label["right_box"] = {
                    "center_x": float(rx_px / input_w),
                    "center_y": float(ry_px / input_h),
                    "width": float(rw_px / input_w),
                    "height": float(rh_px / input_h),
                }
            
            # Transform vertices if present
            # Vertices are stored normalized to original image size [0, 1]
            if "vertices" in new_label:
                vertices = new_label["vertices"]
                transformed_vertices = {}
                for v_key in ["v1", "v2", "v3", "v4"]:
                    if v_key in vertices:
                        vx, vy = vertices[v_key]
                        # Denormalize from original image (vertices are normalized to [0, 1])
                        vx_px = float(vx) * orig_w
                        vy_px = float(vy) * orig_h
                        
                        # Apply letterbox scale (effective per-axis)
                        vx_px = vx_px * scale_w
                        vy_px = vy_px * scale_h
                        
                        # Add letterbox padding
                        vx_px = vx_px + pad_left
                        vy_px = vy_px + pad_top
                        
                        # Normalize to letterboxed image size (H, W)
                        transformed_vertices[v_key] = [
                            float(vx_px / input_w),
                            float(vy_px / input_h),
                        ]
                
                if transformed_vertices:
                    new_label["vertices"] = transformed_vertices
            
            transformed_labels.append(new_label)
        
        return transformed_labels

    def _transform_calib_for_letterbox(
        self,
        calib: Dict[str, float],
        scale_w: float,
        scale_h: float,
        pad_left: int,
        pad_top: int,
    ) -> Dict[str, float]:
        """Transform calibration parameters from original image space to letterboxed space.
        
        Args:
            calib: Calibration dictionary with fx, fy, cx, cy, baseline.
            scale_w: Effective letterbox x-scale (matches actual integer resize width).
            scale_h: Effective letterbox y-scale (matches actual integer resize height).
            pad_left: Left padding added by letterbox.
            pad_top: Top padding added by letterbox.
            
        Returns:
            Transformed calibration dictionary.
        """
        new_calib = dict(calib)  # Copy to avoid modifying original
        
        # Scale focal lengths (use effective per-axis scales)
        new_calib["fx"] = float(calib.get("fx", 0.0) * scale_w)
        new_calib["fy"] = float(calib.get("fy", 0.0) * scale_h)
        
        # Scale and shift principal point
        new_calib["cx"] = float(calib.get("cx", 0.0) * scale_w + pad_left)
        new_calib["cy"] = float(calib.get("cy", 0.0) * scale_h + pad_top)
        
        # Baseline is in meters, not affected by image transformations
        # new_calib["baseline"] remains unchanged
        
        
        return new_calib

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item using BaseDataset's transform pipeline.

        BaseDataset.__getitem__ calls get_image_and_label() then applies transforms.
        Our update_labels_info() prepares the stereo image, and transforms handle the rest.
        """
        # Use BaseDataset's __getitem__ which applies transforms
        result = super().__getitem__(idx)

        # Ensure we have the right keys for collate_fn
        # Transforms should have handled img, labels, calibration
        # But we need to make sure labels and calib are in the right format
        if "labels" not in result:
            # Extract from the label dict structure
            if "instances" in result:
                # Convert instances back to labels format if needed
                pass

        # Ensure im_file is set
        if "im_file" not in result:
            result["im_file"] = self.im_files[idx]

        return result

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that generates targets for training/validation.
        
        Args:
            batch: List of samples from __getitem__.
            
        Returns:
            Dictionary with batched images, targets, and metadata.
        """
        imgs = torch.stack([b["img"] for b in batch], 0)  # (B,6,H,W)
        
        # Validate batch has 6 channels (stereo: left RGB + right RGB)
        assert imgs.shape[1] == 6, (
            f"Stereo batch must have 6 channels, but got shape {imgs.shape}. "
            f"Expected (B, 6, H, W) for stereo input."
        )
        
        # Extract from instances and stereo-specific keys
        calibs = [b.get("calibration", b.get("calib", {})) for b in batch]
        ori_shapes = [b.get("ori_shape", b.get("resized_shape", (0, 0))) for b in batch]

        # ---------------------------------------------------------------------
        # YOLO11-style detection targets (P3-only first)
        # ---------------------------------------------------------------------
        # Produce standard keys used by TaskAlignedAssigner-based losses:
        # - batch_idx: [N, 1]
        # - cls: [N] (per-object class ids)
        # - bboxes: [N, 4] normalized xywh in input image space (letterboxed)
        #
        # Also produce per-object stereo/3D targets, padded per image:
        # aux_targets[name]: [B, max_n, C] in feature-map units for P3 (stride=8).
        input_h, input_w = self.imgsz_tuple  # Use tuple format
        stride = 8.0  # P3/8 for first iteration
        out_h = int(round(input_h / stride))
        out_w = int(round(input_w / stride))

        all_batch_idx: list[int] = []
        all_cls: list[int] = []
        all_bboxes: list[list[float]] = []

        # Per-image lists (will be padded)
        per_image_aux: dict[str, list[torch.Tensor]] = {
            "lr_distance": [],
            "right_width": [],
            "dimensions": [],
            "orientation": [],
            "vertices": [],
            "vertex_offset": [],
            "vertex_dist": [],
        }
        per_image_counts: list[int] = []

        # Reuse mean_dims and std_dims mapping from TargetGenerator for consistency.
        mean_dims = self.target_generator.mean_dims
        std_dims = self.target_generator.std_dims
        class_names_map = self.target_generator.class_names_map

        for i, (b, calib, ori_shape) in enumerate(zip(batch, calibs, ori_shapes)):
            # Extract from instances (which now contains 3D data)
            instances = b.get("instances")
            cls_array = b.get("cls", np.array([], dtype=np.int64))
            occluded = b.get("occluded", np.zeros((0,), dtype=np.int32))
            stereo_labels = b.get("stereo_labels", [])  # For backward compatibility with target_generator
            
            # Get number of objects and extract data from Instances
            if instances is not None and len(instances) > 0:
                n = len(instances)
                # Convert instances to xywh format if needed
                instances.convert_bbox(format="xywh")
                bboxes_xywh = instances.bboxes  # Already normalized
                
                # Extract 3D data from Instances (now stored as attributes)
                right_bboxes = instances.right_bboxes if instances.right_bboxes is not None else np.zeros((n, 4), dtype=np.float32)
                dimensions_3d = instances.dimensions_3d if instances.dimensions_3d is not None else np.zeros((n, 3), dtype=np.float32)
                location_3d = instances.location_3d if instances.location_3d is not None else np.zeros((n, 3), dtype=np.float32)
                rotation_y = instances.rotation_y if instances.rotation_y is not None else np.zeros((n,), dtype=np.float32)
            else:
                n = 0
                right_bboxes = np.zeros((0, 4), dtype=np.float32)
                dimensions_3d = np.zeros((0, 3), dtype=np.float32)
                location_3d = np.zeros((0, 3), dtype=np.float32)
                rotation_y = np.zeros((0,), dtype=np.float32)
            
            # Filter occluded objects if enabled (for training only)
            if self.filter_occluded and n > 0:
                valid_mask = occluded <= self.max_occlusion_level
                filtered_count = n - valid_mask.sum()
                
                if filtered_count > 0:
                    # Filter using Instances indexing (preserves 3D attributes)
                    if instances is not None and len(instances) > 0:
                        instances = instances[valid_mask]
                        bboxes_xywh = instances.bboxes
                        right_bboxes = instances.right_bboxes if instances.right_bboxes is not None else np.zeros((0, 4), dtype=np.float32)
                        dimensions_3d = instances.dimensions_3d if instances.dimensions_3d is not None else np.zeros((0, 3), dtype=np.float32)
                        location_3d = instances.location_3d if instances.location_3d is not None else np.zeros((0, 3), dtype=np.float32)
                        rotation_y = instances.rotation_y if instances.rotation_y is not None else np.zeros((0,), dtype=np.float32)
                    cls_array = cls_array[valid_mask]
                    occluded = occluded[valid_mask]
                    if len(stereo_labels) > 0:
                        stereo_labels = [stereo_labels[j] for j in range(len(stereo_labels)) if valid_mask[j]]
                    n = valid_mask.sum()
                    
                    # Log filtered objects for the first few batches
                    if i < 5:
                        LOGGER.debug(
                            f"Image {i}: filtered {filtered_count}/{n + filtered_count} occluded objects "
                            f"(max_occlusion_level={self.max_occlusion_level})"
                        )
            
            per_image_counts.append(n)
            if n == 0:
                # keep empty placeholders
                continue

            # Build per-object aux targets for this image
            lr_list = []
            rw_list = []
            dim_list = []
            ori_list = []
            v_list = []
            vo_list = []
            vd_list = []

            for j in range(n):
                cls_i = int(cls_array[j])
                
                # Normalized xywh (letterboxed input space) for detection loss.
                cx, cy, bw, bh = bboxes_xywh[j]
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

                all_batch_idx.append(i)
                all_cls.append(cls_i)
                all_bboxes.append([cx, cy, bw, bh])

                # -------------------------
                # Stereo 2D aux targets (feature-map units)
                # -------------------------
                center_x_px = cx * input_w
                right_cx, right_cy, right_bw, right_bh = right_bboxes[j]
                right_center_x_px = float(right_cx) * input_w
                disparity_px = center_x_px - right_center_x_px
                lr_feat = disparity_px / stride  # feature units
                lr_list.append(torch.tensor([lr_feat], dtype=torch.float32))

                right_w_px = float(right_bw) * input_w
                right_w_feat = right_w_px / stride
                rw_list.append(torch.tensor([right_w_feat], dtype=torch.float32))

                # -------------------------
                # 3D aux targets (object-level, reused from TargetGenerator logic)
                # -------------------------
                dims = dimensions_3d[j]  # [length, width, height] in meters
                # mean_dims now uses integer keys (class_id) instead of string keys (class_name)
                mean_dim = mean_dims.get(cls_i, [1.0, 1.0, 1.0])
                std_dim = std_dims.get(cls_i, [0.2, 0.2, 0.5])
                # [ΔH, ΔW, ΔL]
                dim_offset = torch.tensor(
                    [
                        float(dims[2] - mean_dim[2]) / std_dim[2],  # height
                        float(dims[1] - mean_dim[1]) / std_dim[1],  # width
                        float(dims[0] - mean_dim[0]) / std_dim[0],  # length
                    ],
                    dtype=torch.float32,
                )
                dim_list.append(dim_offset)

                # Orientation encoding (8-d) using same encoding as target generator
                rot_y = float(rotation_y[j])
                loc_3d = location_3d[j]
                x_3d = float(loc_3d[0])
                z_3d = float(loc_3d[2])
                ray_angle = math.atan2(x_3d, z_3d)
                alpha = rot_y - ray_angle
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                # 2-bin encoding: [conf0, conf1, sin0, cos0, sin1, cos1, pad, pad]
                enc = torch.zeros(8, dtype=torch.float32)
                if alpha < 0:
                    bin_idx = 0
                    bin_center = -math.pi / 2
                else:
                    bin_idx = 1
                    bin_center = math.pi / 2
                residual = alpha - bin_center
                enc[0] = 1.0 if bin_idx == 0 else 0.0
                enc[1] = 1.0 if bin_idx == 1 else 0.0
                if bin_idx == 0:
                    enc[2] = math.sin(residual)
                    enc[3] = math.cos(residual)
                else:
                    enc[4] = math.sin(residual)
                    enc[5] = math.cos(residual)
                ori_list.append(enc)

                # Vertices targets as object-level offsets relative to object center (normalized by feature map size)
                # We reuse the same projection logic by calling into the target generator, then extracting computed values.
                # NOTE: For YOLO-based detection, we use the generator but read values at center cell.
                # This generates 3D targets for vertices while using YOLO TaskAlignedAssigner for bbox detection.
                # Convert back to dict format for target_generator (backward compatibility)
                if len(stereo_labels) > j:
                    lab = stereo_labels[j]
                else:
                    # Reconstruct dict format if stereo_labels not available
                    lab = {
                        "class_id": cls_i,
                        "left_box": {"center_x": cx, "center_y": cy, "width": bw, "height": bh},
                        "right_box": {"center_x": right_cx, "center_y": right_cy, "width": right_bw, "height": right_bh},
                        "dimensions": {"length": dims[0], "width": dims[1], "height": dims[2]},
                        "location_3d": {"x": x_3d, "y": loc_3d[1], "z": z_3d},
                        "rotation_y": rot_y,
                    }
                
                num_labels = 1
                calib_list = [calib] * num_labels
                ori_shape_list = [ori_shape] * num_labels
                tmp = self.target_generator.generate_targets(
                    [lab],
                    input_size=self.imgsz_tuple,  # Use tuple format
                    calib=calib_list,
                    original_size=ori_shape_list,
                )
                # Find center cell and read out targets written there.
                center_x_out = (cx * input_w) / stride
                center_y_out = (cy * input_h) / stride
                cx_i = int(center_x_out)
                cy_i = int(center_y_out)
                cx_i = max(0, min(cx_i, out_w - 1))
                cy_i = max(0, min(cy_i, out_h - 1))

                v_list.append(tmp["vertices"][:, cy_i, cx_i].float())
                vo_list.append(tmp["vertex_offset"][:, cy_i, cx_i].float())
                vd_list.append(tmp["vertex_dist"][:, cy_i, cx_i].float())

            per_image_aux["lr_distance"].append(torch.stack(lr_list, 0))
            per_image_aux["right_width"].append(torch.stack(rw_list, 0))
            per_image_aux["dimensions"].append(torch.stack(dim_list, 0))
            per_image_aux["orientation"].append(torch.stack(ori_list, 0))
            per_image_aux["vertices"].append(torch.stack(v_list, 0))
            per_image_aux["vertex_offset"].append(torch.stack(vo_list, 0))
            per_image_aux["vertex_dist"].append(torch.stack(vd_list, 0))

        # Build detection tensors
        if all_bboxes:
            batch_idx_t = torch.tensor(all_batch_idx, dtype=torch.long).view(-1, 1)
            cls_t = torch.tensor(all_cls, dtype=torch.long)
            bboxes_t = torch.tensor(all_bboxes, dtype=torch.float32)
        else:
            batch_idx_t = torch.zeros((0, 1), dtype=torch.long)
            cls_t = torch.zeros((0,), dtype=torch.long)
            bboxes_t = torch.zeros((0, 4), dtype=torch.float32)

        # Pad aux targets per image to [B, max_n, C]
        max_n = max(per_image_counts) if per_image_counts else 0
        aux_targets: dict[str, torch.Tensor] = {}
        for k in per_image_aux.keys():
            # Determine channel count
            c = {
                "lr_distance": 1,
                "right_width": 1,
                "dimensions": 3,
                "orientation": 8,
                "vertices": 8,
                "vertex_offset": 8,
                "vertex_dist": 4,
            }[k]
            padded = torch.zeros((len(batch), max_n, c), dtype=torch.float32)
            for bi in range(len(batch)):
                if per_image_counts[bi] == 0:
                    continue
                # If missing (e.g. empty but list absent), skip
                if bi >= len(per_image_aux[k]):
                    continue
                t = per_image_aux[k][bi]  # [n, c]
                padded[bi, : t.shape[0]] = t
            aux_targets[k] = padded

        # Reconstruct labels_list from instances and stereo data for backward compatibility
        labels_list = []
        for i, b in enumerate(batch):
            instances = b.get("instances")
            cls_array = b.get("cls", np.array([], dtype=np.int64))
            right_bboxes = b.get("right_bboxes", np.zeros((0, 4), dtype=np.float32))
            dimensions_3d = b.get("dimensions_3d", np.zeros((0, 3), dtype=np.float32))
            location_3d = b.get("location_3d", np.zeros((0, 3), dtype=np.float32))
            rotation_y = b.get("rotation_y", np.zeros((0,), dtype=np.float32))
            stereo_labels = b.get("stereo_labels", [])
            
            if instances is not None and len(instances) > 0:
                instances.convert_bbox(format="xywh")
                bboxes_xywh = instances.bboxes
                n = len(instances)
                
                image_labels = []
                for j in range(n):
                    if j < len(stereo_labels):
                        # Use original stereo_labels if available
                        image_labels.append(stereo_labels[j])
                    else:
                        # Reconstruct dict format
                        cx, cy, bw, bh = bboxes_xywh[j]
                        right_cx, right_cy, right_bw, right_bh = right_bboxes[j] if j < len(right_bboxes) else [0, 0, 0, 0]
                        dims = dimensions_3d[j] if j < len(dimensions_3d) else [1, 1, 1]
                        loc_3d = location_3d[j] if j < len(location_3d) else [0, 0, 1]
                        rot_y = rotation_y[j] if j < len(rotation_y) else 0.0
                        
                        image_labels.append({
                            "class_id": int(cls_array[j]),
                            "left_box": {"center_x": float(cx), "center_y": float(cy), "width": float(bw), "height": float(bh)},
                            "right_box": {"center_x": float(right_cx), "center_y": float(right_cy), "width": float(right_bw), "height": float(right_bh)},
                            "dimensions": {"length": float(dims[0]), "width": float(dims[1]), "height": float(dims[2])},
                            "location_3d": {"x": float(loc_3d[0]), "y": float(loc_3d[1]), "z": float(loc_3d[2])},
                            "rotation_y": float(rot_y),
                        })
                labels_list.append(image_labels)
            else:
                labels_list.append([])
        
        return {
            "img": imgs,
            "labels": labels_list,  # Keep labels for metrics computation
            "calib": calibs,  # For downstream metrics/tools
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": ori_shapes,
            # YOLO-style detection labels
            "batch_idx": batch_idx_t,
            "cls": cls_t,
            "bboxes": bboxes_t,
            # Aux targets for stereo/3D heads (also available as 'targets' for backward compatibility)
            "targets": aux_targets,  # Primary key used by model.loss()
            "aux_targets": aux_targets,  # Keep for backward compatibility
        }
