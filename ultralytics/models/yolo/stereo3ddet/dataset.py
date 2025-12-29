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
        from ultralytics.data.stereo.target_improved import TargetGenerator
        self.target_generator = TargetGenerator(
            output_size=output_size,
            num_classes=num_classes,
            mean_dims=mean_dims,
        )

    def _get_image_ids(self) -> list[str]:
        """Return image ids that exist in both left and right directories."""
        left_files = sorted(self.left_dir.glob("*.png"))
        if not left_files:
            left_files = sorted(self.left_dir.glob("*.jpg"))

        image_ids: list[str] = []
        for lf in left_files:
            rf = self.right_dir / lf.name
            if rf.exists():
                image_ids.append(lf.stem)
            else:
                LOGGER.warning(f"Missing right image for {lf.stem}, skipping")
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
                assert (
                    0.1 < label_dict["dimensions"]["height"] < 5
                    and 0.1 < label_dict["dimensions"]["width"] < 3
                    and 0.1 < label_dict["dimensions"]["length"] < 20
                ), f"dimensions are out of range: {label_dict['dimensions']}"

                labels.append(label_dict)
        return labels

    def __len__(self) -> int:
        return len(self.image_ids)

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
            
            input_h, input_w = self.imgsz

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

                # Apply letterbox scale
                rx_px = rx_px * scale
                ry_px = ry_px * scale
                rw_px = rw_px * scale
                rh_px = rh_px * scale

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
                        
                        # Apply letterbox scale
                        vx_px = vx_px * scale
                        vy_px = vy_px * scale
                        
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
        
        # Generate targets for each sample in the batch.
        # Targets are generated in letterboxed input space (H, W).
        targets_list = []
        for labels, calib, ori_shape in zip(labels_list, calibs, ori_shapes):
            # TargetGenerator expects calib and original_size to match length of labels
            # All labels in a single image share the same calib and original_size
            num_labels = len(labels)
            calib_list = [calib] * num_labels 
            ori_shape_list = [ori_shape] * num_labels
            
            target = self.target_generator.generate_targets(
                labels,
                input_size=self.imgsz,  # Letterboxed input size (H, W)
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


# Backward-compatible alias (older training/val code imported the adapter name)
Stereo3DDetAdapterDataset = Stereo3DDetDataset
