from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.kitti_stereo import KITTIStereoDataset
from ultralytics.models.yolo.stereo3ddet.augment import StereoAugmentationPipeline, StereoCalibration, PhotometricAugmentor


def _letterbox(image: np.ndarray, new_shape: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


class Stereo3DDetAdapterDataset(Dataset):
    """Wraps KITTIStereoDataset to emit YOLO-style batches for training.

    - Returns keys: 'img' (6-channel), 'targets' (dict), 'im_file', 'ori_shape'.
    - Builds a single 6-channel tensor by stacking left and right RGB after identical letterbox.
    - Converts normalized left 2D boxes to resized+letterboxed normalized xywh.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        imgsz: int,
        names: Dict[int, str] | List[str] | None = None,
        max_samples: int | None = None,
        output_size: Tuple[int, int] | None = None,
        mean_dims: Dict[str, List[float]] | None = None,
    ):
        """Initialize Stereo3DDetAdapterDataset.

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
        """
        self.root = Path(root)
        self.split = split
        self.imgsz = int(imgsz)
        self.names = names or {}

        # Determine if we should filter classes based on names parameter
        # If names contains only 3 classes (Car, Pedestrian, Cyclist), enable filtering
        from ultralytics.models.yolo.stereo3ddet.utils import PAPER_CLASS_NAMES
        # Fix filter_classes logic bug: should be == not != (T125)
        self.filter_classes = set(self.names.values()) == set(PAPER_CLASS_NAMES.values())

        # Validate that names parameter matches actual label classes (T123)
        # Pass max_samples to validation so it only checks the files that will be loaded
        # Initialize base dataset with class filtering and max_samples if needed (T193, T194)
        # DEBUG: MAX_SAMPLES = 1000
        MAX_SAMPLES = max_samples
        self.base = KITTIStereoDataset(
            root=self.root,
            split=self.split,
            filter_classes=self.filter_classes,
            max_samples=MAX_SAMPLES
        )
        self.left_dir = self.root / "images" / split / "left"
        # Full stereo augmentation pipeline (photometric + geometric)
        self._aug = StereoAugmentationPipeline(
            photometric=PhotometricAugmentor(p_apply=0.9)
        )
        
        # Initialize target generator for generating training/validation targets
        # Compute output_size if not provided (default to 8x downsampling for P3)
        if output_size is None:
            # Default to 8x downsampling (P3 architecture)
            # This can be overridden when building the dataset if model architecture is known
            output_h = imgsz // 8
            output_w = imgsz // 8
            output_size = (output_h, output_w)
        self.output_size = output_size
        
        # Get number of classes
        num_classes = len(self.names) if self.names else 3
        
        # Initialize target generator
        from ultralytics.data.stereo.target_improved import TargetGenerator
        self.target_generator = TargetGenerator(
            output_size=output_size,
            num_classes=num_classes,
            mean_dims=mean_dims,
        )

    def _get_actual_label_classes(self, max_samples: int | None = None) -> set[int]:
        """Scan label files and extract unique class IDs from dataset (T120).
        
        Args:
            max_samples: If specified, only scan the first max_samples label files.
        
        Returns:
            set[int]: Set of unique original class IDs found in label files.
        """
        label_dir = self.root / "labels" / self.split
        if not label_dir.exists():
            return set()
        
        actual_classes = set()
        label_files = sorted(label_dir.glob("*.txt"))
        
        # If max_samples is specified, only check the first max_samples files
        if max_samples is not None:
            label_files = label_files[:max_samples]
        
        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(float(parts[0]))  # Handle float format
                                actual_classes.add(class_id)
                            except (ValueError, IndexError):
                                continue
            except (IOError, OSError):
                continue
        
        return actual_classes

    def _map_class_ids_to_names(self, class_ids: set[int]) -> dict[int, str]:
        """Map actual class IDs to class names using ORIGINAL_TO_PAPER mapping (T121).
        
        Args:
            class_ids: Set of original class IDs from label files.
            
        Returns:
            dict[int, str]: Mapping from paper class ID to class name for classes found in labels.
        """
        from ultralytics.models.yolo.stereo3ddet.utils import ORIGINAL_TO_PAPER, PAPER_CLASS_NAMES
        
        mapped_names = {}
        for original_id in class_ids:
            # Check if this class is in the paper set
            if original_id in ORIGINAL_TO_PAPER:
                paper_id = ORIGINAL_TO_PAPER[original_id]
                if paper_id in PAPER_CLASS_NAMES:
                    mapped_names[paper_id] = PAPER_CLASS_NAMES[paper_id]
        
        return mapped_names

    def _validate_names_against_labels(self, max_samples: int | None = None) -> None:
        """Compare names parameter with actual label classes and raise ValueError if mismatch (T122, T124).
        
        Args:
            max_samples: If specified, only validate against the first max_samples label files.
        
        Raises:
            ValueError: If names parameter doesn't match actual label classes.
        """
        # Get actual class IDs from label files (only check files that will be loaded)
        actual_class_ids = self._get_actual_label_classes(max_samples=max_samples)
        
        # If no labels found, skip validation (edge case: empty dataset)
        if not actual_class_ids:
            return
        
        # Map actual class IDs to paper class names
        actual_names = self._map_class_ids_to_names(actual_class_ids)
        
        # If no paper classes found in labels (all filtered out), skip validation
        # This handles the edge case where labels only contain non-paper classes
        if not actual_names:
            return
        
        # Compare with provided names parameter
        expected_names_set = set(self.names.values())
        actual_names_set = set(actual_names.values())
        
        if expected_names_set != actual_names_set:
            # Build clear error message (T124)
            expected_str = ", ".join(sorted(expected_names_set)) if expected_names_set else "(none)"
            actual_str = ", ".join(sorted(actual_names_set)) if actual_names_set else "(none)"
            
            raise ValueError(
                f"names parameter does not match actual label classes. "
                f"Expected classes in names: {expected_str}, "
                f"but found classes in labels: {actual_str}. "
                f"Please ensure the names parameter matches the classes present in your label files."
            )

    def __len__(self) -> int:
        return len(self.base)

    def _transform_labels_for_letterbox(
        self,
        labels: List[Dict[str, Any]],
        scale: float,
        pad_left: int,
        pad_top: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Dict[str, Any]]:
        """Transform label coordinates from original image space to letterboxed space.
        
        Args:
            labels: List of label dictionaries with left_box, right_box, and optionally vertices.
            scale: Letterbox scale factor (min(imgsz / h, imgsz / w)).
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
            
            # Transform left_box
            if "left_box" in new_label:
                lb = new_label["left_box"]
                # Denormalize from original image
                cx_px = float(lb.get("center_x", 0)) * orig_w
                cy_px = float(lb.get("center_y", 0)) * orig_h
                w_px = float(lb.get("width", 0)) * orig_w
                h_px = float(lb.get("height", 0)) * orig_h
                
                # Apply letterbox scale
                cx_px = cx_px * scale
                cy_px = cy_px * scale
                w_px = w_px * scale
                h_px = h_px * scale
                
                # Add letterbox padding
                cx_px = cx_px + pad_left
                cy_px = cy_px + pad_top
                
                # Normalize to letterboxed image size
                new_label["left_box"] = {
                    "center_x": float(cx_px / self.imgsz),
                    "center_y": float(cy_px / self.imgsz),
                    "width": float(w_px / self.imgsz),
                    "height": float(h_px / self.imgsz),
                }
            
            # Transform right_box
            if "right_box" in new_label:
                rb = new_label["right_box"]
                # Denormalize from original image
                rx_px = float(rb.get("center_x", 0)) * orig_w
                rw_px = float(rb.get("width", 0)) * orig_w
                
                # Apply letterbox scale
                rx_px = rx_px * scale
                rw_px = rw_px * scale
                
                # Add letterbox padding (only x, since right box only has center_x and width)
                rx_px = rx_px + pad_left
                
                # Normalize to letterboxed image size
                new_label["right_box"] = {
                    "center_x": float(rx_px / self.imgsz),
                    "width": float(rw_px / self.imgsz),
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
                        
                        # Apply letterbox scale
                        vx_px = vx_px * scale
                        vy_px = vy_px * scale
                        
                        # Add letterbox padding
                        vx_px = vx_px + pad_left
                        vy_px = vy_px + pad_top
                        
                        # Normalize to letterboxed image size
                        transformed_vertices[v_key] = [
                            float(vx_px / self.imgsz),
                            float(vy_px / self.imgsz),
                        ]
                
                if transformed_vertices:
                    new_label["vertices"] = transformed_vertices
            
            transformed_labels.append(new_label)
        
        return transformed_labels

    def _transform_calib_for_letterbox(
        self,
        calib: Dict[str, float],
        scale: float,
        pad_left: int,
        pad_top: int,
    ) -> Dict[str, float]:
        """Transform calibration parameters from original image space to letterboxed space.
        
        Args:
            calib: Calibration dictionary with fx, fy, cx, cy, baseline.
            scale: Letterbox scale factor (min(imgsz / h, imgsz / w)).
            pad_left: Left padding added by letterbox.
            pad_top: Top padding added by letterbox.
            
        Returns:
            Transformed calibration dictionary.
        """
        new_calib = dict(calib)  # Copy to avoid modifying original
        
        # Scale focal lengths (same scale for both dimensions in letterbox)
        new_calib["fx"] = float(calib.get("fx", 0.0) * scale)
        new_calib["fy"] = float(calib.get("fy", 0.0) * scale)
        
        # Scale and shift principal point
        new_calib["cx"] = float(calib.get("cx", 0.0) * scale + pad_left)
        new_calib["cy"] = float(calib.get("cy", 0.0) * scale + pad_top)
        
        # Baseline is in meters, not affected by image transformations
        # new_calib["baseline"] remains unchanged
        
        return new_calib

    def _labels_to_tensors(
        self, labels: List[Dict[str, Any]], scale: float, pad_left: int, pad_top: int, ori_w: int, ori_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_list: List[float] = []
        bboxes_list: List[List[float]] = []

        # Import class mapping utilities if filtering is enabled
        if self.filter_classes:
            from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        for obj in labels:
            try:
                original_cid = int(obj.get("class_id", -1))
                if original_cid < 0:
                    continue
                cid = original_cid

                lb = obj.get("left_box", {})
                cx, cy, bw, bh = float(lb.get("center_x", 0)), float(lb.get("center_y", 0)), float(
                    lb.get("width", 0)
                ), float(lb.get("height", 0))
                # denormalize to original pixels
                x = cx * ori_w
                y = cy * ori_h
                w = bw * ori_w
                h = bh * ori_h
                # apply resize + pad
                x = x * scale + pad_left
                y = y * scale + pad_top
                w = w * scale
                h = h * scale
                # normalize to new square size (imgsz)
                cxn = x / self.imgsz
                cyn = y / self.imgsz
                wn = w / self.imgsz
                hn = h / self.imgsz
                # clamp
                cxn = float(min(max(cxn, 0.0), 1.0))
                cyn = float(min(max(cyn, 0.0), 1.0))
                wn = float(min(max(wn, 0.0), 1.0))
                hn = float(min(max(hn, 0.0), 1.0))

                cls_list.append(float(cid))
                bboxes_list.append([cxn, cyn, wn, hn])
            except Exception:
                continue

        if len(cls_list) == 0:
            cls_t = torch.zeros((0,), dtype=torch.float32)
            bboxes_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            cls_t = torch.tensor(cls_list, dtype=torch.float32)
            bboxes_t = torch.tensor(bboxes_list, dtype=torch.float32)
        return cls_t, bboxes_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        left_img: np.ndarray = sample["left_img"]
        right_img: np.ndarray = sample["right_img"]
        h0, w0 = left_img.shape[:2]

        # Get calibration (needed for both train and val) (T143)
        calib = sample["calib"]
        
        # Optional stereo augmentation (train split only)
        if self.split == "train":
            calib_obj = StereoCalibration(
                fx=float(calib["fx"]),
                fy=float(calib["fy"]),
                cx=float(calib["cx"]),
                cy=float(calib["cy"]),
                baseline=float(calib["baseline"]),
                height=h0,
                width=w0,
            )
            left_img, right_img, labels_aug, calib_obj_aug = self._aug.augment(left_img, right_img, sample["labels"], calib_obj)
            # Use augmented calibration if available, otherwise use original
            if calib_obj_aug is not None:
                calib = calib_obj_aug.to_dict()
            # Get image size after augmentation (before letterbox)
            h_aug, w_aug = left_img.shape[:2]
        else:
            labels_aug = sample["labels"]
            # No augmentation, so image size is still original
            h_aug, w_aug = h0, w0

        # BGR -> RGB for both
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        # Letterbox both identically using left's parameters, then reapply to right by computing separately
        left_resized, scale_l, pad_left_l, pad_top_l = _letterbox(left_rgb, self.imgsz)
        right_resized, _, _, _ = _letterbox(right_rgb, self.imgsz)
        
        # Transform labels from original/augmented image space to letterboxed space
        labels_transformed = self._transform_labels_for_letterbox(
            labels_aug, scale_l, pad_left_l, pad_top_l, w_aug, h_aug
        )
        
        # Transform calibration from original/augmented image space to letterboxed space
        calib_transformed = self._transform_calib_for_letterbox(
            calib, scale_l, pad_left_l, pad_top_l
        )
        
        # Stack to 6 channels (left RGB + right RGB)
        left_t = torch.from_numpy(left_resized).permute(2, 0, 1).contiguous()
        right_t = torch.from_numpy(right_resized).permute(2, 0, 1).contiguous()
        img6 = torch.cat([left_t, right_t], dim=0)
        
        # Runtime validation: ensure stereo image has exactly 6 channels
        assert img6.shape[0] == 6, (
            f"Stereo image must have 6 channels (left RGB + right RGB), "
            f"but got {img6.shape[0]} channels. Check image loading."
        )
        
        if img6.dtype != torch.uint8:
            img6 = img6.to(torch.uint8)
        
        image_id = sample["image_id"]
        im_file = str(self.left_dir / f"{image_id}.png")

        return {
            "img": img6,  # uint8 [0,255], shape (6,H,W)
            "labels": labels_transformed,  # Labels transformed to letterboxed coordinates
            "calib": calib_transformed,  # Calibration transformed to letterboxed space
            "im_file": im_file,
            "ori_shape": (h0, w0),  # Original image size (before augmentation and letterbox)
        }

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
        
        labels_list = [b["labels"] for b in batch]
        calibs = [b["calib"] for b in batch]  # Collect calib for each sample (T145)
        ori_shapes = [b["ori_shape"] for b in batch]  # Original image sizes
        
        # Generate targets for each sample in the batch
        # Targets are generated in letterboxed input space (imgsz x imgsz)
        targets_list = []
        for labels, calib, ori_shape in zip(labels_list, calibs, ori_shapes):
            # TargetGenerator expects calib and original_size to match length of labels
            # All labels in a single image share the same calib and original_size
            num_labels = len(labels)
            calib_list = [calib] * num_labels 
            ori_shape_list = [ori_shape] * num_labels
            
            target = self.target_generator.generate_targets(
                labels,
                input_size=(self.imgsz, self.imgsz),  # Letterboxed input size
                calib=calib_list,
                original_size=ori_shape_list,
            )
            targets_list.append(target)
        
        # Stack targets across batch dimension
        # Each target is a dict with tensors of shape [C, H, W]
        # We need to stack to [B, C, H, W]
        batched_targets = {}
        for key in targets_list[0].keys():
            batched_targets[key] = torch.stack([t[key] for t in targets_list], dim=0)
        
        # Extract class IDs for compatibility with trainer
        # Trainer expects cls to be a tensor with shape [batch_size]
        # We'll use the number of objects as a proxy
        cls_tensor = torch.tensor([len(labels) for labels in labels_list], dtype=torch.long)
        return {
            "img": imgs,
            "labels": labels_list,  # Keep labels for metrics computation
            "targets": batched_targets,  # Generated targets for training/validation
            "calib": calibs,  # Add calib for validation metrics computation (T145)
            "cls": cls_tensor,  # Add cls for trainer compatibility (number of objects per image)
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": ori_shapes,
        }
