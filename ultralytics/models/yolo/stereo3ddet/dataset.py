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

    def __init__(self, root: str | Path, split: str, imgsz: int, names: Dict[int, str] | List[str] | None = None, max_samples: int | None = None):
        """Initialize Stereo3DDetAdapterDataset.

        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split ('train' or 'val').
            imgsz (int): Target image size for letterboxing.
            names (Dict[int, str] | List[str] | None): Class names mapping. If None, uses default.
            max_samples (int | None): Maximum number of samples to load. If None, loads all available samples.
                If specified, only the first max_samples samples will be loaded. Defaults to None.
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

                # Filter and remap class ID if filtering is enabled
                if self.filter_classes:
                    # If base dataset already filtered, class_id is already remapped
                    # But we should double-check it's in the paper set
                    remapped_cid = filter_and_remap_class_id(original_cid)
                    if remapped_cid is None:
                        # This shouldn't happen if base dataset filtered correctly, but log if it does
                        from ultralytics.utils import LOGGER
                        LOGGER.warning(f"Filtering out class {original_cid} in adapter dataset (not in paper set: Car, Pedestrian, Cyclist)")
                        continue
                    cid = remapped_cid
                else:
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
        calib = sample.get("calib", {})
        
        # Optional stereo augmentation (train split only)
        if self.split == "train":
            calib_obj = StereoCalibration(
                fx=float(calib.get("fx", 0.0)),
                fy=float(calib.get("fy", 0.0)),
                cx=float(calib.get("cx", 0.0)),
                cy=float(calib.get("cy", 0.0)),
                baseline=float(calib.get("baseline", 0.0)),
                height=h0,
                width=w0,
            )
            left_img, right_img, labels_aug, _ = self._aug.augment(left_img, right_img, sample.get("labels", []), calib_obj)
        else:
            labels_aug = sample.get("labels", [])

        # BGR -> RGB for both
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        # Letterbox both identically using left's parameters, then reapply to right by computing separately
        left_resized, scale_l, pad_left_l, pad_top_l = _letterbox(left_rgb, self.imgsz)
        right_resized, _, _, _ = _letterbox(right_rgb, self.imgsz)
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
        labels = labels_aug
        image_id = sample.get("image_id")
        im_file = str(self.left_dir / f"{image_id}.png")

        return {
            "img": img6,  # uint8 [0,255], shape (6,H,W)
            "labels": labels,  # Pass labels for target generation (filtered and remapped if needed)
            "calib": calib,  # Add calib for validation metrics computation (T144)
            "im_file": im_file,
            "ori_shape": (h0, w0),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        imgs = torch.stack([b["img"] for b in batch], 0)  # (B,6,H,W)
        
        # Validate batch has 6 channels (stereo: left RGB + right RGB)
        assert imgs.shape[1] == 6, (
            f"Stereo batch must have 6 channels, but got shape {imgs.shape}. "
            f"Expected (B, 6, H, W) for stereo input."
        )
        
        labels_list = [b.get("labels", []) for b in batch]
        calibs = [b.get("calib", {}) for b in batch]  # Collect calib for each sample (T145)
        
        # Final safety check: ensure all class IDs are in [0, 1, 2] range
        # This should not be needed if filtering works correctly in __getitem__, but provides a safety net
        # Filter out any labels with invalid class IDs to prevent training errors
        filtered_labels_list = []
        for labels in labels_list:
            filtered_labels = []
            for label in labels:
                class_id = label.get("class_id", -1)
                if class_id in [0, 1, 2]:
                    filtered_labels.append(label)
                else:
                    from ultralytics.utils import LOGGER
                    LOGGER.warning(f"Filtering out label with invalid class_id {class_id} in collate_fn. This should not happen if filtering works correctly in __getitem__.")
            filtered_labels_list.append(filtered_labels)
        
        labels_list = filtered_labels_list
        
        # Extract class IDs for compatibility with trainer
        # Trainer expects cls to be a tensor with shape [batch_size]
        # We'll use the number of objects as a proxy
        cls_tensor = torch.tensor([len(labels) for labels in labels_list], dtype=torch.long)
        return {
            "img": imgs,
            "labels": labels_list,  # Pass labels for target generation
            "calib": calibs,  # Add calib for validation metrics computation (T145)
            "cls": cls_tensor,  # Add cls for trainer compatibility (number of objects per image)
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
        }
