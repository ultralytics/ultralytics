from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.stereo3ddet.augment import StereoAugmentationPipeline, StereoCalibration, PhotometricAugmentor
from ultralytics.utils import LOGGER
from ultralytics.data.stereo.target_improved import TargetGenerator
import math


def _to_hw(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize imgsz to (H, W)."""
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        return int(imgsz[0]), int(imgsz[1])
    raise TypeError(f"imgsz must be int or (h,w), got {imgsz} ({type(imgsz).__name__})")


class Stereo3DDetDataset(Dataset):
    """Stereo 3D detection dataset (single, unified dataset).

    - Returns keys: 'img' (6-channel), 'targets' (dict), 'im_file', 'ori_shape'.
    - Builds a single 6-channel tensor by stacking left and right RGB after identical letterbox.
    - Converts normalized left 2D boxes to resized+letterboxed normalized xywh.
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
        filter_occluded: bool = False,
        max_occlusion_level: int = 1,
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
        """
        self.root = Path(root)
        self.split = split
        self.imgsz = _to_hw(imgsz)  # (H, W)
        self.names = names or {}
        self._letterbox = LetterBox(new_shape=self.imgsz, auto=False, scale_fill=False, scaleup=True, stride=32)

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
        self.filter_occluded = filter_occluded
        self.max_occlusion_level = max_occlusion_level
        if filter_occluded:
            LOGGER.info(
                f"Occlusion filtering enabled: excluding objects with occlusion level > {max_occlusion_level} "
                f"(0=visible, 1=partial, 2=heavy, 3=unknown)"
            )

        # Full stereo augmentation pipeline (photometric + geometric)
        self._aug = StereoAugmentationPipeline(
            photometric=PhotometricAugmentor(p_apply=0.9)
        )
        
        # Initialize target generator for generating training/validation targets
        # Compute output_size if not provided (default to 8x downsampling for P3)
        if output_size is None:
            # Default to 8x downsampling (P3 architecture). This can be overridden by the trainer if model is known.
            input_h, input_w = self.imgsz
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
        left_files = sorted(self.left_dir.glob("*.png"))
        if not left_files:
            left_files = sorted(self.left_dir.glob("*.jpg"))

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
            
            input_h, input_w = self.imgsz

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
        image_id = self.image_ids[idx]

        # Load images (BGR)
        left_img_path = self.left_dir / f"{image_id}.png"
        if not left_img_path.exists():
            left_img_path = self.left_dir / f"{image_id}.jpg"
        right_img_path = self.right_dir / left_img_path.name

        left_img = cv2.imread(str(left_img_path))
        right_img = cv2.imread(str(right_img_path))
        if left_img is None:
            raise FileNotFoundError(f"Could not load left image: {left_img_path}")
        if right_img is None:
            raise FileNotFoundError(f"Could not load right image: {right_img_path}")

        h0, w0 = left_img.shape[:2]

        # Get calibration (needed for both train and val) (T143)
        calib_file = self.calib_dir / f"{image_id}.txt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        calib = self._parse_calibration(calib_file)

        # Load labels (normalized coordinates in original/augmented image space)
        label_file = self.label_dir / f"{image_id}.txt"
        labels = self._parse_labels(label_file)
        
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
            left_img, right_img, labels_aug, calib_obj_aug = self._aug.augment(left_img, right_img, labels, calib_obj)
            # Use augmented calibration if available, otherwise use original
            if calib_obj_aug is not None:
                calib = calib_obj_aug.to_dict()
            # Get image size after augmentation (before letterbox)
            h_aug, w_aug = left_img.shape[:2]
        else:
            labels_aug = labels
            # No augmentation, so image size is still original
            h_aug, w_aug = h0, w0

        # BGR -> RGB for both
        # left_rgb = left_img.copy()
        # right_rgb = right_img.copy()
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        # Letterbox stereo pair identically by letterboxing the stacked 6-channel image once.
        # This guarantees perfect pixel alignment between left/right views.
        stereo6 = np.concatenate([left_rgb, right_rgb], axis=2)  # [H, W, 6]

        # Compute the exact scale/pad that LetterBox will apply (to transform labels/calibration consistently).
        in_h, in_w = stereo6.shape[:2]
        out_h, out_w = self.imgsz
        scale_l = min(out_h / in_h, out_w / in_w)
        new_unpad_w, new_unpad_h = round(in_w * scale_l), round(in_h * scale_l)
        # Effective scale actually used after rounding to integer resize sizes (can differ slightly from scale_l)
        scale_w_eff = new_unpad_w / float(in_w) if in_w else 1.0
        scale_h_eff = new_unpad_h / float(in_h) if in_h else 1.0
        dw, dh = out_w - new_unpad_w, out_h - new_unpad_h
        dw /= 2
        dh /= 2
        pad_left_l = int(round(dw - 0.1))
        pad_top_l = int(round(dh - 0.1))

        stereo6_resized = self._letterbox(image=stereo6)
        left_resized = stereo6_resized[:, :, :3]
        right_resized = stereo6_resized[:, :, 3:]
        
        # Transform labels from original/augmented image space to letterboxed space
        labels_transformed = self._transform_labels_for_letterbox(
            labels_aug, scale_w_eff, scale_h_eff, pad_left_l, pad_top_l, w_aug, h_aug
        )
        
        # Transform calibration from original/augmented image space to letterboxed space
        calib_transformed = self._transform_calib_for_letterbox(
            calib, scale_w_eff, scale_h_eff, pad_left_l, pad_top_l
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
        input_h, input_w = self.imgsz
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

        for i, (labels, calib, ori_shape) in enumerate(zip(labels_list, calibs, ori_shapes)):
            # Filter occluded objects if enabled (for training only)
            if self.filter_occluded:
                filtered_labels = []
                filtered_count = 0
                for lab in labels:
                    occlusion = lab.get("occluded", 0)
                    # KITTI occlusion levels: 0=fully visible, 1=partially occluded, 
                    # 2=heavily occluded, 3=unknown/completely blocked
                    if occlusion <= self.max_occlusion_level:
                        filtered_labels.append(lab)
                    else:
                        filtered_count += 1
                
                # Log filtered objects for the first few batches
                if i < 5 and filtered_count > 0:
                    LOGGER.debug(
                        f"Image {i}: filtered {filtered_count}/{len(labels)} occluded objects "
                        f"(max_occlusion_level={self.max_occlusion_level})"
                    )
                
                labels = filtered_labels
            
            n = len(labels)
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

            for lab in labels:
                cls_i = int(lab["class_id"])
                lb = lab["left_box"]
                rb = lab["right_box"]

                # Normalized xywh (letterboxed input space) for detection loss.
                cx = float(lb["center_x"])
                cy = float(lb["center_y"])
                bw = float(lb["width"])
                bh = float(lb["height"])

                all_batch_idx.append(i)
                all_cls.append(cls_i)
                all_bboxes.append([cx, cy, bw, bh])

                # -------------------------
                # Stereo 2D aux targets (feature-map units)
                # -------------------------
                center_x_px = cx * input_w
                right_center_x_px = float(rb["center_x"]) * input_w
                disparity_px = center_x_px - right_center_x_px
                lr_feat = disparity_px / stride  # feature units
                lr_list.append(torch.tensor([lr_feat], dtype=torch.float32))

                right_w_px = float(rb["width"]) * input_w
                right_w_feat = right_w_px / stride
                rw_list.append(torch.tensor([right_w_feat], dtype=torch.float32))

                # -------------------------
                # 3D aux targets (object-level, reused from TargetGenerator logic)
                # -------------------------
                dims = lab["dimensions"]  # meters
                # mean_dims now uses integer keys (class_id) instead of string keys (class_name)
                mean_dim = mean_dims.get(cls_i, [1.0, 1.0, 1.0])
                std_dim = std_dims.get(cls_i, [0.2, 0.2, 0.5])
                # [ΔH, ΔW, ΔL]
                dim_offset = torch.tensor(
                    [
                        float(dims["height"] - mean_dim[2]) / std_dim[2],
                        float(dims["width"] - mean_dim[1]) / std_dim[1],
                        float(dims["length"] - mean_dim[0]) / std_dim[0],
                    ],
                    dtype=torch.float32,
                )
                dim_list.append(dim_offset)

                # Orientation encoding (8-d) using same encoding as target generator
                rotation_y = float(lab["rotation_y"])
                loc = lab.get("location_3d", None) or {"x": 0.0, "z": 1.0}
                x_3d = float(loc.get("x", 0.0))
                z_3d = float(loc.get("z", 1.0))
                ray_angle = math.atan2(x_3d, z_3d)
                alpha = rotation_y - ray_angle
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
                num_labels = 1
                calib_list = [calib] * num_labels
                ori_shape_list = [ori_shape] * num_labels
                tmp = self.target_generator.generate_targets(
                    [lab],
                    input_size=self.imgsz,
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


# Backward-compatible alias (older training/val code imported the adapter name)
Stereo3DDetAdapterDataset = Stereo3DDetDataset
