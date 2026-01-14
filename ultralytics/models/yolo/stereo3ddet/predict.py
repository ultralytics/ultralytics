# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.data.stereo.calib import CalibrationParameters, load_kitti_calibration
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.stereo3ddet.val import decode_stereo3d_outputs_yolo11_p3
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_imgsz


def load_stereo_pair(
    left_path: str | Path,
    right_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load stereo image pair (left and right images).

    Args:
        left_path: Path to left image file.
        right_path: Path to right image file.

    Returns:
        tuple: (left_image, right_image) as numpy arrays in BGR format.

    Raises:
        FileNotFoundError: If either image file does not exist.
        ValueError: If images cannot be loaded or have different sizes.
    """
    left_path = Path(left_path)
    right_path = Path(right_path)

    if not left_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_path}")

    # Load images using OpenCV (BGR format)
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None:
        raise ValueError(f"Failed to load left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Failed to load right image: {right_path}")

    # Verify images have the same size
    if left_img.shape != right_img.shape:
        raise ValueError(
            f"Image size mismatch: left {left_img.shape} vs right {right_img.shape}"
        )

    return left_img, right_img


class Stereo3DDetPredictor(DetectionPredictor):
    """Stereo 3D Detection predictor.

    Extends DetectionPredictor to handle stereo image pairs (6-channel input)
    and decode 3D bounding boxes from 10-branch model outputs.
    """

    def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize Stereo3DDetPredictor.

        Args:
            cfg: Configuration dictionary or path. Defaults to DEFAULT_CFG if None.
            overrides: Configuration overrides.
            _callbacks: Callback functions.
        """
        # Ensure task is set in overrides
        if overrides is None:
            overrides = {}
        if "task" not in overrides:
            overrides["task"] = "stereo3ddet"
        
        # Use DEFAULT_CFG if cfg is None (BasePredictor expects this)
        from ultralytics.utils import DEFAULT_CFG
        if cfg is None:
            cfg = DEFAULT_CFG
        
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "stereo3ddet"
        # Store calibration parameters for each image
        self.calib_params: dict[str, CalibrationParameters] = {}
        
        # Initialize letterbox transformer (same as dataset)
        # Will be updated in setup_source when imgsz is checked
        self._letterbox = None
        
        # Mean and std dimensions for decoding (from dataset config)
        if isinstance(self.data, (str, Path)):
            data_cfg = YAML.load(str(self.data))
        self.mean_dims = data_cfg.get("mean_dims")
        self.std_dims = data_cfg.get("std_dims")

    def setup_source(self, source=None):
        """Set up input source for stereo prediction.

        For stereo3ddet, source can be:
        - A tuple/list of (left_path, right_path) for a single stereo pair
        - A list of tuples/lists for multiple stereo pairs
        - A single path (raises ValueError - stereo requires both left and right)

        Args:
            source: Input source(s) for prediction.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        
        # Initialize letterbox transformer (same as dataset)
        # Use the same parameters: auto=False, scale_fill=False, scaleup=True, stride=32
        self._letterbox = LetterBox(new_shape=self.imgsz, auto=False, scale_fill=False, scaleup=True, stride=32)
        
        if source is None:
            source = self.args.source

        # Check if source is a single path (not allowed for stereo)
        if isinstance(source, (str, Path)):
            try:
                left_path, right_path = source.split(",")
                stereo_pairs = [(left_path, right_path)]
            except ValueError:
                raise ValueError(
                    "Stereo3DDetPredictor requires both left and right images. "
                    "Provide source as a tuple/list: (left_path, right_path) or "
                    "[(left1, right1), (left2, right2), ...] for batch prediction."
                )

        # Convert to list of stereo pairs
        elif isinstance(source, (tuple, list)) and len(source) > 0:
            # Check if first element is a tuple/list (batch of pairs)
            if isinstance(source[0], (tuple, list)) and len(source[0]) == 2:
                # Batch of stereo pairs: [(left1, right1), (left2, right2), ...]
                stereo_pairs = source
            elif len(source) == 2:
                # Single stereo pair: (left, right)
                stereo_pairs = [source]
            else:
                raise ValueError(
                    f"Invalid source format. Expected (left_path, right_path) or "
                    f"[(left1, right1), ...], got: {source}"
                )
        else:
            raise ValueError(f"Invalid source type: {type(source)}")

        # Load stereo pairs and create a dataset-like structure
        self.stereo_pairs = []
        for left_path, right_path in stereo_pairs:
            left_img, right_img = load_stereo_pair(left_path, right_path)
            # Stack left and right to create 6-channel image
            stereo_img = np.concatenate([left_img, right_img], axis=2)  # [H, W, 6]
            self.stereo_pairs.append((stereo_img, str(left_path)))

            # Try to load calibration from KITTI format
            # Look for calib file in same directory as left image
            left_path_obj = Path(left_path)
            calib_path = left_path_obj.parent / f"{left_path_obj.stem}.txt"
            if not calib_path.exists():
                # Try alternative: calib.txt in parent directory
                calib_path = left_path_obj.parent / "calib.txt"
            if calib_path.exists():
                try:
                    calib = load_kitti_calibration(calib_path)
                    self.calib_params[str(left_path)] = calib
                except Exception as e:
                    LOGGER.warning(f"Failed to load calibration from {calib_path}: {e}")
                    # Use default calibration
                    self.calib_params[str(left_path)] = CalibrationParameters(
                        fx=721.5377, fy=721.5377, cx=609.5593, cy=172.8540,
                        baseline=0.54, image_width=stereo_img.shape[1], image_height=stereo_img.shape[0]
                    )
            else:
                # Use default calibration
                self.calib_params[str(left_path)] = CalibrationParameters(
                    fx=721.5377, fy=721.5377, cx=609.5593, cy=172.8540,
                    baseline=0.54, image_width=stereo_img.shape[1], image_height=stereo_img.shape[0]
                )

        # Set up dataset-like structure for BasePredictor
        # BasePredictor expects dataset to yield (paths, im0s, s) tuples
        class StereoDataset:
            def __init__(self, stereo_pairs):
                self.stereo_pairs = stereo_pairs
                self.bs = len(stereo_pairs)  # batch size
                self.mode = "image"  # Stereo pairs are image files
                self.source_type = type("SourceType", (), {
                    "stream": False,
                    "tensor": False,
                    "screenshot": False,
                    "from_img": False,
                })()

            def __iter__(self):
                # Yield batch as (paths, im0s, s) tuple
                paths = [path for _, path in self.stereo_pairs]
                im0s = [img for img, _ in self.stereo_pairs]
                s = [""] * len(self.stereo_pairs)  # empty strings for now
                yield (paths, im0s, s)

            def __len__(self):
                return 1  # Single batch

        self.dataset = StereoDataset(self.stereo_pairs)
        self.source_type = self.dataset.source_type

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Preprocess stereo image pairs for inference.

        Args:
            im: List of stereo images (6-channel) or tensor.

        Returns:
            Preprocessed tensor of shape (N, 6, H, W).
        """
        if isinstance(im, torch.Tensor):
            # Already a tensor, just move to device and normalize
            im = im.to(self.device)
            im = im.half() if self.model.fp16 else im.float()
            if im.dtype == torch.uint8:
                im /= 255
            return im

        # Apply letterbox to each stereo image (same as dataset)
        # Each image is [H, W, 6] (stereo pair)
        letterboxed = []
        for stereo_img in im:
            # Apply letterbox to resize to target imgsz
            if self._letterbox is not None:
                letterboxed_img = self._letterbox(image=stereo_img)
            else:
                # Fallback: no letterbox (shouldn't happen if setup_source called)
                letterboxed_img = stereo_img
            letterboxed.append(letterboxed_img)
        
        # Convert list of letterboxed numpy arrays to tensor
        im = np.stack(letterboxed)  # [N, H, W, 6]
        im = im[..., ::-1]  # BGR to RGB for all 6 channels
        im = im.transpose((0, 3, 1, 2))  # [N, H, W, 6] -> [N, 6, H, W]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255  # 0-255 to 0.0-1.0

        return im

    def postprocess(self, preds: dict[str, torch.Tensor], img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs) -> list[Results]:
        """Post-process model predictions to Results objects with 3D boxes.

        Args:
            preds: Dictionary of YOLO11-mapped model outputs.
            img: Preprocessed input tensor.
            orig_imgs: List of original stereo images (6-channel).
            **kwargs: Additional arguments.

        Returns:
            List of Results objects with boxes3d attribute.
        """
        # Get calibration for batch
        calib = None
        if self.batch and self.batch[0]:
            # Try to get calibration from first image
            img_path = self.batch[0][0]
            calib = self.calib_params.get(img_path)
            if calib is not None:
                if isinstance(calib, CalibrationParameters):
                    # TODO: Refactor to use the CalibrationParameters object instead of dict.
                    calib = {
                        "fx": calib.fx,
                        "fy": calib.fy,
                        "cx": calib.cx,
                        "cy": calib.cy,
                        "baseline": calib.baseline,
                    }
        
        # Get original shapes
        ori_shapes = None
        if orig_imgs and len(orig_imgs) > 0:
            orig_img = orig_imgs[0]
            # orig_img is 6-channel stereo image, get height and width
            ori_shapes = [(orig_img.shape[0], orig_img.shape[1])]  # (H, W)
        
        # Get imgsz
        imgsz = getattr(self.args, 'imgsz', 384)
        
        # Get class names from model
        class_names = self.model.names if hasattr(self.model, "names") else None
        if class_names is None:
            raise ValueError("Model must have names attribute for prediction")
        
        # Decode 3D boxes using YOLO11 decoder
        # Use default mean_dims and std_dims for KITTI (will use dataset config if loaded)
        mean_dims = self.mean_dims
        std_dims = self.std_dims

        # TODO: ugly fix for single calibration parameter.
        if isinstance(calib, dict):
            calib = [calib]
        
        results_boxes3d = decode_stereo3d_outputs_yolo11_p3(
            preds,
            conf_threshold=self.args.conf,
            top_k=self.args.max_det,
            calib=calib,
            imgsz=imgsz,
            ori_shapes=ori_shapes,
            iou_thres=getattr(self.args, "iou", 0.45),
            mean_dims=mean_dims,
            std_dims=std_dims,
            class_names=class_names,
        )
        
        # Handle result format (list[Box3D] for batch_size=1, list[list[Box3D]] for batch_size>1)
        if isinstance(results_boxes3d, list) and len(results_boxes3d) > 0 and isinstance(results_boxes3d[0], Box3D):
            # Single batch result - wrap in list
            results_boxes3d = [results_boxes3d]
        
        # Create Results objects
        results = []
        for i, boxes3d in enumerate(results_boxes3d):
            img_path = self.batch[0][i] if self.batch and len(self.batch[0]) > i else f"image_{i}.jpg"
            orig_img = orig_imgs[i][:, :, :3] if len(orig_imgs) > i else orig_imgs[0][:, :, :3]
            result = Results(
                orig_img=orig_img,
                path=img_path,
                names=class_names,
                boxes3d=boxes3d if isinstance(boxes3d, list) else [],
            )
            results.append(result)

        return results