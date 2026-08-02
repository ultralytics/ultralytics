# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.dataset import parse_calib_p2
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops

from .utils import set_detect3d_quality_power


class Detection3DPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a 3D detection model.

    This predictor handles inference for YOLO models trained for 3D detection, processing output with 3D parameters
    (depth, position, dimensions, rotation).

    Attributes:
        args (dict): Configuration arguments for the predictor.

    Methods:
        postprocess: Post-process predictions for 3D detection.
        construct_result: Construct a single Results object with extra 3D parameters attached.

    Examples:
        >>> from ultralytics.models.yolo.detect3d import Detection3DPredictor
        >>> args = dict(model="yolo11n-3d.pt", source="image.jpg", calib="calib/image.txt")
        >>> predictor = Detection3DPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize Detection3DPredictor."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect3d"
        super().__init__(cfg if cfg is not None else DEFAULT_CFG, overrides, _callbacks)

    def setup_model(self, model, verbose: bool = True) -> None:
        """Load the inference backend and apply configurable q3d score calibration to native PyTorch models."""
        super().setup_model(model, verbose)
        set_detect3d_quality_power(self.model, self.args.quality3d_power)

    def _load_p2(self, img_path):
        """Load and cache the calibration for an image from a configured file or stem-matched directory."""
        calib = getattr(getattr(self, "args", None), "calib", None)
        if not calib:
            return None

        calib_root = Path(calib).expanduser()
        calib_path = calib_root / f"{Path(img_path).stem}.txt" if calib_root.is_dir() else calib_root
        if not calib_path.is_file():
            if calib_root.is_dir():
                raise FileNotFoundError(
                    f"Detect3D calibration file not found for image '{img_path}': expected '{calib_path}'"
                )
            raise FileNotFoundError(f"Detect3D calibration file not found: '{calib_path}'")

        cache = getattr(self, "_p2_cache", None)
        if cache is None:
            cache = self._p2_cache = {}
        cache_key = str(calib_path.resolve())
        if cache_key not in cache:
            try:
                cache[cache_key] = parse_calib_p2(str(calib_path))
            except (OSError, ValueError) as e:
                raise ValueError(f"Invalid Detect3D calibration file '{calib_path}': {e}") from e
        return cache[cache_key]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction, keeping the extra 3D parameters.

        The NMS output `pred` has shape (N, 6 + nd) where the last nd=8 columns are the decoded 3D parameters
        (center_x, center_y, depth, sin_alpha, cos_alpha, h_3d, w_3d, l_3d).
        The base class drops them via `pred[:, :6]`, so here they are mapped to native-image coordinates and stored in
        the formal `Results.d3_params` container (aligned row-by-row with `result.boxes`).
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        d3_params = pred[:, 6:].clone()
        if len(d3_params):
            # Invert predictor letterboxing without clipping: a valid projected 3D center can lie outside the image for
            # a truncated object even when part of its 2D box remains visible.
            img_h, img_w = img.shape[2:]
            orig_h, orig_w = orig_img.shape[:2]
            gain = min(img_h / orig_h, img_w / orig_w)
            pad_x = round((img_w - round(orig_w * gain)) / 2 - 0.1)
            pad_y = round((img_h - round(orig_h * gain)) / 2 - 0.1)
            d3_params[:, 0] = (d3_params[:, 0] - pad_x) / gain
            d3_params[:, 1] = (d3_params[:, 1] - pad_y) / gain
        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            d3_params=d3_params,
            p2=self._load_p2(img_path),
        )
