# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.data.stereo.calib import CalibrationParameters, load_kitti_calibration
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor

from ultralytics.models.yolo.s3d.head import DEPTH_MAX, DEPTH_MIN
from ultralytics.models.yolo.s3d.preprocess import decode_and_refine_predictions, preprocess_stereo_images
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import unwrap_model


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
            overrides["task"] = "s3d"

        # Use DEFAULT_CFG if cfg is None (BasePredictor expects this)
        from ultralytics.utils import DEFAULT_CFG

        if cfg is None:
            cfg = DEFAULT_CFG

        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "s3d"
        # Store calibration parameters for each image
        self.calib_params: dict[str, CalibrationParameters] = {}

        # Initialize letterbox transformer (same as dataset)
        # Will be updated in setup_source when imgsz is checked
        self._letterbox = None

        # Will be populated from model attributes in setup_model
        self.mean_dims = None
        self.std_dims = None

    def setup_model(self, model, verbose=True):
        """Set up model and read mean/std dims stored during training (like classify's transforms)."""
        super().setup_model(model, verbose)
        # Prefer dims stored on the model (saved during training); fall back to data YAML
        self.mean_dims = getattr(self.model.model, "mean_dims", None)
        self.std_dims = getattr(self.model.model, "std_dims", None)
        # Depth range travels with the checkpoint via the head's DepthDFL bin buffer.
        head = getattr(unwrap_model(self.model.model), "model", [None])[-1]
        self.depth_min = getattr(head, "depth_min", DEPTH_MIN)
        self.depth_max = getattr(head, "depth_max", DEPTH_MAX)

    def setup_source(self, source=None):
        """Set up input source for stereo prediction.

        For s3d, source can be:
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
                source_str = str(source).strip().strip("()[]")
                left_path, right_path = source_str.split(",")
                stereo_pairs = [(left_path.strip(), right_path.strip())]
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
                    f"Invalid source format. Expected (left_path, right_path) or [(left1, right1), ...], got: {source}"
                )
        else:
            raise ValueError(f"Invalid source type: {type(source)}")

        # Load stereo pairs and calibrations
        self.stereo_pairs = []
        for left_path, right_path in stereo_pairs:
            # Load images
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            if left_img is None or right_img is None:
                raise ValueError(f"Failed to load images: {left_path}, {right_path}")
            stereo_img = np.concatenate([left_img, right_img], axis=2)
            self.stereo_pairs.append((stereo_img, str(left_path)))

            # Resolve the calibration file from the usual locations, including the dataset layout
            # <root>/images/<split>/left/<stem>.png -> <root>/calib/<split>/<stem>.txt.
            left = Path(left_path)
            candidates = [left.with_suffix(".txt"), left.parent / "calib.txt"]
            if left.parent.name == "left" and len(left.parents) >= 4:
                candidates.append(left.parents[3] / "calib" / left.parents[1].name / f"{left.stem}.txt")
            calib = None
            for calib_path in candidates:
                if calib_path.exists():
                    try:
                        calib = load_kitti_calibration(calib_path)
                        break
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(f"Failed to parse calibration {calib_path}: {exc}")
            if calib is None:
                LOGGER.warning(
                    f"No calibration found for {left_path}; using default KITTI calibration — "
                    f"3D predictions will be inaccurate for cameras with different intrinsics."
                )
                calib = CalibrationParameters(
                    fx=721.5377,
                    fy=721.5377,
                    cx=609.5593,
                    cy=172.8540,
                    baseline=0.54,
                    image_width=stereo_img.shape[1],
                    image_height=stereo_img.shape[0],
                )
            self.calib_params[str(left_path)] = calib

        # Set up dataset-like structure for BasePredictor
        # BasePredictor expects dataset to yield (paths, im0s, s) tuples
        class StereoDataset:
            def __init__(self, stereo_pairs):
                self.stereo_pairs = stereo_pairs
                self.bs = len(stereo_pairs)  # batch size
                self.mode = "image"  # Stereo pairs are image files
                self.source_type = type(
                    "SourceType",
                    (),
                    {
                        "stream": False,
                        "tensor": False,
                        "screenshot": False,
                        "from_img": False,
                    },
                )()

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

        Uses shared preprocessing from preprocess.py for consistency with validator.

        Args:
            im: List of stereo images (6-channel) or tensor.

        Returns:
            Preprocessed tensor of shape (N, 6, H, W).
        """
        return preprocess_stereo_images(
            images=im,
            imgsz=self.imgsz,
            device=self.device,
            half=self.model.fp16,
            letterbox=self._letterbox,
        )

    def postprocess(self, preds, img: torch.Tensor, orig_imgs: list[np.ndarray], **kwargs) -> list[Results]:
        """Post-process model predictions to Results objects with 3D boxes.

        Uses shared decode_and_refine_predictions from preprocess.py which handles
        decoding, geometric construction, and dense alignment refinement.

        Args:
            preds: Tuple of (inference_output, preds_dict) from model forward.
            img: Preprocessed input tensor.
            orig_imgs: List of original stereo images (6-channel).
            **kwargs: Additional arguments.

        Returns:
            List of Results objects with boxes3d attribute.
        """
        # Unpack (y, preds_dict) from Detect.forward — AutoBackend may convert tuple to list
        if isinstance(preds, (tuple, list)) and len(preds) == 2 and isinstance(preds[1], dict):
            y, preds_dict = preds
            preds_dict = {**preds_dict, "det": y}
        else:
            preds_dict = preds

        # Build calibration list from stored parameters
        calibs = []
        if self.batch and self.batch[0]:
            for img_path in self.batch[0]:
                calib = self.calib_params.get(img_path)
                if calib is not None:
                    if isinstance(calib, CalibrationParameters):
                        calibs.append(
                            {
                                "fx": calib.fx,
                                "fy": calib.fy,
                                "cx": calib.cx,
                                "cy": calib.cy,
                                "baseline": calib.baseline,
                            }
                        )
                    else:
                        calibs.append(calib)

        # Get original shapes from all images
        ori_shapes = []
        if orig_imgs:
            for orig_img in orig_imgs:
                ori_shapes.append((orig_img.shape[0], orig_img.shape[1]))  # (H, W)

        # Get class names from model
        class_names = self.model.names if hasattr(self.model, "names") else None
        if class_names is None:
            raise ValueError("Model must have names attribute for prediction")

        # Build batch dict for decode_and_refine_predictions
        batch = {
            "calib": calibs,
            "ori_shape": ori_shapes,
            "img": img,  # Include preprocessed images for dense alignment
        }

        # Use shared decode and refine pipeline (includes geometric + dense alignment)
        results_boxes3d = decode_and_refine_predictions(
            preds=preds_dict,
            batch=batch,
            args=self.args,
            use_geometric=getattr(self.args, "use_geometric", None),
            use_dense_alignment=getattr(self.args, "use_dense_alignment", None),
            conf_threshold=self.args.conf,
            top_k=self.args.max_det,
            iou_thres=getattr(self.args, "iou", 0.45),
            imgsz=getattr(self.args, "imgsz", 384),
            mean_dims=self.mean_dims,
            std_dims=self.std_dims,
            class_names=class_names,
            depth_min=getattr(self, "depth_min", DEPTH_MIN),
            depth_max=getattr(self, "depth_max", DEPTH_MAX),
        )

        # Create Results objects (one per input image; decode may return fewer when no detections)
        num_imgs = len(orig_imgs)
        results = []
        for i in range(num_imgs):
            boxes3d = results_boxes3d[i] if i < len(results_boxes3d) else []
            img_path = self.batch[0][i] if self.batch and len(self.batch[0]) > i else f"image_{i}.jpg"
            orig_img = orig_imgs[i][:, :, :3] if i < len(orig_imgs) else orig_imgs[0][:, :, :3]
            result = Results(
                orig_img=orig_img,
                path=img_path,
                names=class_names,
                boxes3d=boxes3d if isinstance(boxes3d, list) else [],
            )
            # Attach calibration for 3D box projection in plot()
            calib = self.calib_params.get(img_path)
            if calib is not None:
                if isinstance(calib, CalibrationParameters):
                    calib = {"fx": calib.fx, "fy": calib.fy, "cx": calib.cx, "cy": calib.cy, "baseline": calib.baseline}
                result._calib = calib
            results.append(result)

        return results
