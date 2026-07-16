# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import Compose, Format
from ultralytics.data.base import BaseDataset
from ultralytics.data.stereo.calib import load_kitti_calibration
from ultralytics.data.utils import load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.s3d.augment import (
    StereoCrop,
    StereoHFlip,
    StereoHSV,
    StereoLetterBox,
    StereoScale,
)
from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.checks import check_imgsz


def compute_dimension_offset(
    dims: tuple[float, float, float],
    class_id: int,
    mean_dims: dict,
    std_dims: dict,
) -> torch.Tensor:
    """Compute normalized dimension offset [ΔH, ΔW, ΔL] for 3D detection.

    The offset is computed as (dim - mean) / std for each dimension, following the paper's approach for stable training.

    Args:
        dims: Object dimensions as (length, width, height) in meters.
        class_id: Integer class ID for looking up mean/std values.
        mean_dims: Dict mapping class_id -> [mean_L, mean_W, mean_H].
        std_dims: Dict mapping class_id -> [std_L, std_W, std_H].

    Returns:
        Tensor of shape [3] with normalized offsets [ΔH, ΔW, ΔL].
    """
    mean_dim = mean_dims.get(class_id, [1.0, 1.0, 1.0])
    std_dim = std_dims.get(class_id, [0.2, 0.2, 0.5])
    return torch.tensor(
        [
            (dims[2] - mean_dim[2]) / std_dim[2],  # height
            (dims[1] - mean_dim[1]) / std_dim[1],  # width
            (dims[0] - mean_dim[0]) / std_dim[0],  # length
        ],
        dtype=torch.float32,
    )


def encode_proj_offset(
    location_3d: tuple[float, float, float],
    height: float,
    calib: dict,
    box_center_norm: tuple[float, float],
    ratio_pad: tuple[float, float, float],
    input_wh: tuple[int, int],
) -> tuple[float, float]:
    """Encode the offset from the 2D box center to the projected 3D centroid (letterbox-normalized).

    Args:
        location_3d (tuple): (X, Y, Z) bottom-center in camera frame (meters).
        height (float): 3D box height (meters); centroid Y = Y - height/2 (camera y-down).
        calib (dict): fx, fy, cx, cy (original-image pixels).
        box_center_norm (tuple): (u, v) 2D box center, letterbox-normalized [0,1].
        ratio_pad (tuple): (scale, pad_left, pad_top) of the letterbox applied to this sample.
        input_wh (tuple): (input_w, input_h) letterbox canvas size.

    Returns:
        (du, dv): projected-centroid minus box-center, in letterbox-normalized units.
    """
    X, Y, Z = location_3d
    Yc = Y - height / 2.0
    scale, pad_left, pad_top = ratio_pad
    input_w, input_h = input_wh
    u_orig = calib["fx"] * X / Z + calib["cx"]
    v_orig = calib["fy"] * Yc / Z + calib["cy"]
    u_lb = u_orig * scale + pad_left
    v_lb = v_orig * scale + pad_top
    u_norm = u_lb / input_w
    v_norm = v_lb / input_h
    return u_norm - box_center_norm[0], v_norm - box_center_norm[1]


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
        names: dict[int, str] | list[str] | None = None,
        mean_dims: dict[str, list[float]] | None = None,
        std_dims: dict[str, list[float]] | None = None,
        filter_occluded: bool = False,
        max_occlusion_level: int = 1,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        data: dict | None = None,
        fraction: float = 1.0,
    ):
        """Initialize Stereo3DDetDataset.

        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split ('train' or 'val').
            imgsz (int): Target image size for letterboxing.
            names (dict[int, str] | list[str] | None): Class names mapping. If None, uses default.
            mean_dims (dict[str, list[float]] | None): Mean dimensions per class [L, W, H] in meters. If None, uses
                default KITTI values.
            std_dims (dict[str, list[float]] | None): Standard deviation of dimensions per class [L, W, H] in meters.
                Used for normalized offset prediction. If None, defaults to reasonable estimates.
            filter_occluded (bool): Whether to filter out heavily occluded objects during training. If True, objects
                with occlusion level > max_occlusion_level are excluded from training. Defaults to False.
            max_occlusion_level (int): Maximum occlusion level to include when filter_occluded is True.
                KITTI occlusion levels: 0=fully visible, 1=partially occluded, 2=heavily occluded, 3=unknown.
                Default is 1 (exclude heavily occluded and unknown objects).
            cache: Cache images to RAM or disk.
            augment: Whether to apply augmentation.
            hyp: Hyperparameters dict.
            prefix: Prefix for log messages.
            data: Dataset configuration dict.
            fraction (float): Fraction of dataset to utilize (0.0 to 1.0). Defaults to 1.0 (use all samples).
        """
        self.root = Path(root)
        self.split = split
        self.imgsz_tuple = check_imgsz(imgsz, min_dim=2)  # (H, W) - stored separately for transforms
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
            fraction=fraction,
        )

        # Get number of classes
        assert mean_dims is not None, "mean_dims must be provided"
        assert std_dims is not None, "std_dims must be provided"
        # Dimension priors arrive from the dataset YAML keyed by class NAME (e.g. {"Car": [L,W,H]}),
        # but compute_dimension_offset() looks them up by integer class_id during target encoding.
        # Without rekeying, every lookup misses and falls back to the generic default mean/std,
        # producing badly-scaled dimension targets (the model then learns garbage that the decode
        # side — which DOES rekey to int ids — mis-expands). Normalize to int-keyed here so the
        # training target uses the same priors the decoder uses. Order stays [L, W, H].
        self.mean_dims = self._rekey_dims_to_int(mean_dims)
        self.std_dims = self._rekey_dims_to_int(std_dims)

    def _rekey_dims_to_int(self, dims: dict) -> dict:
        """Return dimension priors keyed by integer class id, matching string keys via names.

        Accepts dicts keyed by class name (e.g. "Car") or already by int id, and returns a dict
        keyed by integer class id while preserving the per-class [L, W, H] value order.
        """
        if not dims:
            return dims
        # Already int-keyed: leave as-is.
        if all(isinstance(k, int) for k in dims):
            return dims
        names = self.names if isinstance(self.names, dict) else {i: n for i, n in enumerate(self.names or [])}
        result: dict[int, list[float]] = {}
        for cid, cname in names.items():
            if cname in dims:
                result[cid] = dims[cname]
            elif cid in dims:
                result[cid] = dims[cid]
        return result if result else dims

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Override to return left image files that have matching right images, labels, and calibration.

        This method performs the validation check that all required files exist for each stereo pair,
        eliminating the need for a separate image_ids list.
        """
        # Get all left image files from parent
        left_files = super().get_img_files(img_path)

        # Filter to only include files that have all required matching files
        valid_files = []
        for f in left_files:
            path = Path(f)
            image_id = path.stem

            # Check for matching right image
            right_path = self.right_dir / path.name
            if not right_path.exists():
                LOGGER.warning(f"Missing right image for {image_id}, skipping")
                continue

            # Check for calibration file
            calib_file = self.calib_dir / f"{image_id}.txt"
            if not calib_file.exists():
                LOGGER.warning(f"Missing calibration for {image_id}, skipping")
                continue

            # Check for label file
            label_file = self.label_dir / f"{image_id}.txt"
            if not label_file.exists():
                LOGGER.warning(f"Missing label file for {image_id}, skipping")
                continue

            valid_files.append(f)

        if not valid_files:
            raise ValueError(f"No valid stereo pairs found in {self.left_dir} (all missing right/label/calib)")

        return valid_files

    def _read_image(self, im_file: str) -> tuple[np.ndarray, tuple[int, int]]:
        """Read stereo image pair (left + right) from disk for index 'i'.

        Overrides BaseDataset._read_image() to load both left and right images,
        convert them to RGB, and concatenate into a 6-channel stereo image.

        Args:
            im_file (str): Path to the left image file.

        Returns:
            im (np.ndarray): 6-channel stereo image [H, W, 6] (left RGB + right RGB).
            hw_original (tuple[int, int]): Original image dimensions (height, width) from left image.

        Raises:
            FileNotFoundError: If either image file is not found.
        """
        from ultralytics.utils.patches import imread

        # Load left image
        left_img = imread(im_file, flags=self.cv2_flag)  # BGR
        if left_img is None:
            raise FileNotFoundError(f"Image Not Found {im_file}")

        # Load right image
        image_id = Path(im_file).stem
        right_img_path = self.right_dir / Path(im_file).name
        if not right_img_path.exists():
            # Try different extension
            for ext in [".png", ".jpg", ".jpeg"]:
                right_img_path = self.right_dir / f"{image_id}{ext}"
                if right_img_path.exists():
                    break

        right_img = imread(str(right_img_path), flags=self.cv2_flag)  # BGR
        if right_img is None:
            raise FileNotFoundError(f"Right image not found: {right_img_path}")

        # Get original shape from left image
        h0, w0 = left_img.shape[:2]

        # Handle grayscale
        if left_img.ndim == 2:
            left_img = left_img[..., None]
        if right_img.ndim == 2:
            right_img = right_img[..., None]

        # Convert BGR to RGB and concatenate to 6-channel stereo image
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        im = np.concatenate([left_rgb, right_rgb], axis=2)  # [H, W, 6]

        return im, (h0, w0)

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
            except (OSError, ValueError) as e:
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
        except OSError as e:
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

        transforms = []

        if self.augment:
            transforms.extend(
                [
                    StereoHFlip(p=float(hyp.get("fliplr", 0.5))),
                    StereoScale(scale_range=(0.8, 1.2), p=float(hyp.get("scale", 0.5))),
                    StereoCrop(crop_h=0.9, crop_w=0.9, p=float(hyp.get("crop_fraction", 0.3))),
                    StereoHSV(
                        hgain=float(hyp.get("hsv_h", 0.015)),
                        sgain=float(hyp.get("hsv_s", 0.7)),
                        vgain=float(hyp.get("hsv_v", 0.4)),
                        p=0.5,
                    ),
                ]
            )

        transforms.append(StereoLetterBox(new_shape=self.imgsz_tuple, scaleup=self.augment, stride=32))
        transforms.append(Format(normalize=True, return_stereo=True))

        return Compose(transforms)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Convert stereo labels to Instances format with 3D data included.

        Args:
            label: Label dict from BaseDataset.get_image_and_label(). The 'img' key already contains the 6-channel
                stereo image from load_image().

        Returns:
            Updated label dict with Instances containing 3D data.
        """
        from ultralytics.utils.instance import Instances

        stereo_img = label["img"]  # Already 6-channel from load_image()

        n_ch = stereo_img.shape[-1]
        if n_ch != 6:
            raise ValueError(f"Expected 6 channel stereo image, got {n_ch}. Shape: {stereo_img.shape}")

        # Get labels and calibration from cached data (label IS self.labels[index] via deepcopy)
        label_list = label["labels"]  # List of dict objects

        # Filter by configured classes (e.g., Car-only dataset skips Ped/Cyc)
        if self.names:
            label_list = [obj for obj in label_list if obj["class_id"] in self.names]
        # Copy calibration so we don't mutate the cached original.
        calib = dict(label["calibration"]) if label["calibration"] else {}

        # Scale calibration by the pre-resize ratio from load_image().
        # load_image() resizes the image but does NOT update calibration,
        # so we must apply the same scale here to keep them in sync.
        # Without this, StereoLetterBox (and downstream reversal in decode/visualization)
        # operates on calibration that doesn't match the actual image dimensions.
        ratio_h, ratio_w = label.get("ratio_pad", (1.0, 1.0))
        if calib and (ratio_w != 1.0 or ratio_h != 1.0):
            calib["fx"] = calib.get("fx", 0.0) * ratio_w
            calib["fy"] = calib.get("fy", 0.0) * ratio_h
            calib["cx"] = calib.get("cx", 0.0) * ratio_w
            calib["cy"] = calib.get("cy", 0.0) * ratio_h
        label["calibration"] = calib

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
        else:
            # Extract left boxes (normalized xywh) and class IDs
            bboxes = np.array(
                [
                    [
                        obj["left_box"]["center_x"],
                        obj["left_box"]["center_y"],
                        obj["left_box"]["width"],
                        obj["left_box"]["height"],
                    ]
                    for obj in label_list
                ],
                dtype=np.float32,
            )
            cls = np.array([obj["class_id"] for obj in label_list], dtype=np.int64)

            # Extract right boxes
            right_bboxes = np.array(
                [
                    [
                        obj["right_box"]["center_x"],
                        obj["right_box"]["center_y"],
                        obj["right_box"]["width"],
                        obj["right_box"]["height"],
                    ]
                    for obj in label_list
                ],
                dtype=np.float32,
            )

            # Extract 3D data
            dimensions_3d = np.array(
                [
                    [obj["dimensions"]["length"], obj["dimensions"]["width"], obj["dimensions"]["height"]]
                    for obj in label_list
                ],
                dtype=np.float32,
            )

            location_3d = np.array(
                [[obj["location_3d"]["x"], obj["location_3d"]["y"], obj["location_3d"]["z"]] for obj in label_list],
                dtype=np.float32,
            )

            rotation_y = np.array([obj["rotation_y"] for obj in label_list], dtype=np.float32)

            # Extract additional metadata
            truncated = np.array([obj.get("truncated", 0.0) for obj in label_list], dtype=np.float32)
            occluded = np.array([obj.get("occluded", 0) for obj in label_list], dtype=np.int32)

        # Create Instances with 3D data included (normalized xywh format)
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
        # Store additional metadata that's not part of Instances
        label["truncated"] = truncated
        label["occluded"] = occluded

        return label

    def _parse_calibration(self, calib_file: Path) -> dict[str, Any]:
        """Parse a stereo calibration file into fx/fy/cx/cy/baseline/image_width/image_height.

        Delegates to the shared dual-format loader (raw KITTI P-matrices or simplified fx/fy keys).
        """
        return load_kitti_calibration(calib_file).to_dict()

    def _parse_labels(self, label_file: Path) -> list[dict[str, Any]]:
        """Parse YOLO 3D label file (18-value format, with backward-compat for legacy 26-value)."""
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels: list[dict[str, Any]] = []
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                n = len(parts)
                if n == 18:
                    # New 18-value format: no vertices
                    trunc_idx, occ_idx = 16, 17
                elif n == 26:
                    # Legacy 26-value format: skip vertex indices 16-23
                    trunc_idx, occ_idx = 24, 25
                else:
                    LOGGER.warning(f"Invalid label format in {label_file}: expected 18 or 26 values, got {n}")
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
                    "location_3d": {"x": values[12], "y": values[13], "z": values[14]},
                    "truncated": float(values[trunc_idx]),
                    "occluded": int(values[occ_idx]),
                }

                # Keep existing assertions for early data-quality feedback
                assert -np.pi <= label_dict["rotation_y"] <= np.pi, (
                    f"rotation_y is out of range: {label_dict['rotation_y']}"
                )
                labels.append(label_dict)
        return labels

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
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

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function that generates targets for training/validation.

        Args:
            batch: List of samples from __getitem__.

        Returns:
            Dictionary with batched images, targets, and metadata.
        """
        imgs = torch.stack([b["img"] for b in batch], 0)  # (B,6,H,W)

        assert imgs.shape[1] == 6, f"Stereo batch must have 6 channels, got shape {imgs.shape}."

        # Extract from instances and stereo-specific keys
        calibs = [b.get("calibration", b.get("calib", {})) for b in batch]
        ori_shapes = [b.get("ori_shape", b.get("resized_shape", (0, 0))) for b in batch]
        input_h, input_w = self.imgsz_tuple  # letterbox canvas (H, W), constant across the batch

        # ---------------------------------------------------------------------
        # YOLO26-style detection targets (P3-only first)
        # ---------------------------------------------------------------------
        # Produce standard keys used by TaskAlignedAssigner-based losses:
        # - batch_idx: [N, 1]
        # - cls: [N] (per-object class ids)
        # - bboxes: [N, 4] normalized xywh in input image space (letterboxed)
        #
        # Also produce per-object stereo/3D targets, padded per image:
        # aux_targets[name]: [B, max_n, C] in feature-map units for P3 (stride=8).

        all_batch_idx: list[int] = []
        all_cls: list[int] = []
        all_bboxes: list[list[float]] = []

        # Per-image lists (will be padded)
        per_image_aux: dict[str, list[torch.Tensor]] = {
            "lr_distance": [],
            "depth": [],
            "dimensions": [],
            "orientation": [],
            "is_pseudo": [],  # 0=real, 1=stereo-pseudo (occ>=10), 2=mono-pseudo (occ>=20)
            "proj_offset": [],  # (du, dv): box-center -> projected-3D-centroid offset
        }

        per_image_counts: list[int] = []
        labels_list: list[list[dict[str, Any]]] = []

        for i, b in enumerate(batch):
            # Extract from Format output tensors (instances was already popped by Format)
            bboxes_tensor = b.get("bboxes")  # torch.Tensor (N, 4) normalized xywh
            cls_tensor = b.get("cls")  # torch.Tensor (N,)
            right_bboxes_tensor = b.get("right_bboxes")  # torch.Tensor (N, 4)
            dimensions_3d_tensor = b.get("dimensions_3d")  # torch.Tensor (N, 3)
            location_3d_tensor = b.get("location_3d")  # torch.Tensor (N, 3)
            rotation_y_tensor = b.get("rotation_y")  # torch.Tensor (N,)
            occluded_tensor = b.get("occluded")  # torch.Tensor (N,) or None
            truncated_tensor = b.get("truncated")  # torch.Tensor (N,) or None

            # Convert tensors to numpy for processing
            if bboxes_tensor is not None and len(bboxes_tensor) > 0:
                n = len(bboxes_tensor)
                bboxes_xywh = bboxes_tensor.numpy()
                cls_array = cls_tensor.numpy() if cls_tensor is not None else np.zeros((n,), dtype=np.int64)
                right_bboxes = (
                    right_bboxes_tensor.numpy()
                    if right_bboxes_tensor is not None
                    else np.zeros((n, 4), dtype=np.float32)
                )
                dimensions_3d = (
                    dimensions_3d_tensor.numpy()
                    if dimensions_3d_tensor is not None
                    else np.zeros((n, 3), dtype=np.float32)
                )
                location_3d = (
                    location_3d_tensor.numpy() if location_3d_tensor is not None else np.zeros((n, 3), dtype=np.float32)
                )
                rotation_y = (
                    rotation_y_tensor.numpy() if rotation_y_tensor is not None else np.zeros((n,), dtype=np.float32)
                )
                occluded = (
                    occluded_tensor.numpy().astype(np.int32)
                    if occluded_tensor is not None
                    else np.zeros((n,), dtype=np.int32)
                )
                truncated = (
                    truncated_tensor.numpy().astype(np.float32)
                    if truncated_tensor is not None
                    else np.zeros((n,), dtype=np.float32)
                )
            else:
                n = 0
                bboxes_xywh = np.zeros((0, 4), dtype=np.float32)
                cls_array = np.zeros((0,), dtype=np.int64)
                right_bboxes = np.zeros((0, 4), dtype=np.float32)
                dimensions_3d = np.zeros((0, 3), dtype=np.float32)
                location_3d = np.zeros((0, 3), dtype=np.float32)
                rotation_y = np.zeros((0,), dtype=np.float32)
                occluded = np.zeros((0,), dtype=np.int32)
                truncated = np.zeros((0,), dtype=np.float32)

            # Filter occluded objects if enabled (for training only)
            if self.filter_occluded and n > 0:
                valid_mask = occluded <= self.max_occlusion_level
                filtered_count = n - valid_mask.sum()

                if filtered_count > 0:
                    bboxes_xywh = bboxes_xywh[valid_mask]
                    cls_array = cls_array[valid_mask]
                    right_bboxes = right_bboxes[valid_mask]
                    dimensions_3d = dimensions_3d[valid_mask]
                    location_3d = location_3d[valid_mask]
                    rotation_y = rotation_y[valid_mask]
                    occluded = occluded[valid_mask]
                    truncated = truncated[valid_mask]
                    n = valid_mask.sum()

                    # Log filtered objects for the first few batches
                    if i < 5:
                        LOGGER.debug(
                            f"Image {i}: filtered {filtered_count}/{n + filtered_count} occluded objects "
                            f"(max_occlusion_level={self.max_occlusion_level})"
                        )

            per_image_counts.append(n)

            # Build this sample's list-of-dicts for batch["labels"] from tensors (single source of truth)
            sample_labels = []
            for j in range(n):
                cx, cy, bw, bh = (
                    float(bboxes_xywh[j, 0]),
                    float(bboxes_xywh[j, 1]),
                    float(bboxes_xywh[j, 2]),
                    float(bboxes_xywh[j, 3]),
                )
                right_cx = float(right_bboxes[j, 0])
                right_cy = float(right_bboxes[j, 1])
                right_bw = float(right_bboxes[j, 2])
                right_bh = float(right_bboxes[j, 3])
                dims = dimensions_3d[j]
                loc_3d = location_3d[j]
                sample_labels.append(
                    {
                        "class_id": int(cls_array[j]),
                        "left_box": {"center_x": cx, "center_y": cy, "width": bw, "height": bh},
                        "right_box": {
                            "center_x": right_cx,
                            "center_y": right_cy,
                            "width": right_bw,
                            "height": right_bh,
                        },
                        "dimensions": {"length": float(dims[0]), "width": float(dims[1]), "height": float(dims[2])},
                        "location_3d": {"x": float(loc_3d[0]), "y": float(loc_3d[1]), "z": float(loc_3d[2])},
                        "rotation_y": float(rotation_y[j]),
                        "truncated": float(truncated[j]),
                        "occluded": int(occluded[j]),
                    }
                )
            labels_list.append(sample_labels)

            if n == 0:
                # keep empty placeholders
                continue

            # Build per-object aux targets for this image
            calib_i = calibs[i]
            lr_list = []
            depth_list = []
            dim_list = []
            ori_list = []
            pseudo_list = []
            proj_list = []

            for j in range(n):
                cls_i = int(cls_array[j])
                z_3d = float(location_3d[j][2])

                # Normalized xywh (letterboxed input space) for detection loss.
                cx, cy, bw, bh = bboxes_xywh[j]
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

                all_batch_idx.append(i)
                all_cls.append(cls_i)
                all_bboxes.append([cx, cy, bw, bh])

                # Disparity in log-space (log of normalized disparity)
                right_cx = float(right_bboxes[j, 0])
                disparity_norm = cx - right_cx
                lr_list.append(torch.tensor([math.log(max(disparity_norm, 1e-6))], dtype=torch.float32))
                depth_list.append(torch.tensor([math.log(max(z_3d, 1e-6))], dtype=torch.float32))

                # Dimension offset: normalized (dim - mean) / std, shape [3]
                dims = dimensions_3d[j]  # [length, width, height] in meters
                dim_list.append(compute_dimension_offset(tuple(dims), cls_i, self.mean_dims, self.std_dims))

                # Orientation: MultiBin encoding of alpha (rotation_y - ray_angle).
                # ray_angle = atan2(x_3d, z_3d). See orientation.encode_orientation.
                loc = location_3d[j]  # [x, y, z] in camera frame
                ray_angle = math.atan2(float(loc[0]), float(loc[2]))
                alpha = float(rotation_y[j]) - ray_angle
                ori_list.append(torch.tensor(encode_orientation(alpha), dtype=torch.float32))

                # Pseudo-label flag: 0=real, 1=stereo-pseudo (occ>=10), 2=mono-pseudo (occ>=20)
                occ_j = int(occluded[j])
                pseudo_list.append(torch.tensor([min(occ_j // 10, 2)], dtype=torch.float32))

                # Projected-3D-centroid offset (du, dv), letterbox-normalized. calib_i's fx/fy/cx/cy
                # are already letterbox-transformed (scaled+padded, see StereoLetterBox/
                # update_labels_info), so an identity ratio_pad is passed to encode_proj_offset.
                du, dv = encode_proj_offset(
                    (float(loc[0]), float(loc[1]), max(z_3d, 1e-6)),
                    float(dims[2]),
                    calib_i,
                    (cx, cy),
                    (1.0, 0.0, 0.0),
                    (input_w, input_h),
                )
                proj_list.append(torch.tensor([du, dv], dtype=torch.float32))

            per_image_aux["lr_distance"].append(torch.stack(lr_list, 0))
            per_image_aux["depth"].append(torch.stack(depth_list, 0))
            per_image_aux["dimensions"].append(torch.stack(dim_list, 0))
            per_image_aux["orientation"].append(torch.stack(ori_list, 0))
            per_image_aux["is_pseudo"].append(torch.stack(pseudo_list, 0))
            per_image_aux["proj_offset"].append(torch.stack(proj_list, 0))

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
            c = {
                "lr_distance": 1,
                "depth": 1,
                "dimensions": 3,
                "orientation": ORIENT_CHANNELS,
                "is_pseudo": 1,
                "proj_offset": 2,
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

        result = {
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

        return result
