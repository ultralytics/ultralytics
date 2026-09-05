# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import model_info

from .predict import Predictor, SAM2Predictor, SAM3Predictor, SAM3SemanticPredictor


class SAM(Model):
    """SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

    This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for promptable
    segmentation with versatility in image analysis. It supports various prompts such as bounding boxes, points, or
    labels, and features zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM model.
        is_sam2 (bool): Indicates whether the model is SAM2 variant.
        task (str): The task type, set to "segment" for SAM models.

    Methods:
        predict: Perform segmentation prediction on the given image or video source.
        info: Log information about the SAM model.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """Initialize the SAM (Segment Anything Model) instance.

        Args:
            model (str): Path to a pre-trained ``.pt`` or ``.pth`` checkpoint, or to a SAM3 export directory whose name
                ends in ``_onnx`` or ``_engine``.

        Raises:
            NotImplementedError: If the path is neither such a checkpoint nor such a directory.
        """
        path = Path(model)
        # A SAM3 export is a directory of modules rather than a single file, and the trailing
        # _onnx / _engine is what selects the backend that serves it.
        self.is_exported_dir = bool(model) and path.is_dir() and path.name.endswith(("_onnx", "_engine"))
        if model and not self.is_exported_dir and path.suffix not in {".pt", ".pth"}:
            raise NotImplementedError(
                "SAM prediction requires a pre-trained *.pt or *.pth model, or a SAM3 export directory "
                "whose name ends in _onnx or _engine."
            )
        self.is_sam2 = "sam2" in path.stem
        self.is_sam3 = "sam3" in path.stem or self.is_exported_dir
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """Load the specified weights into the SAM model.

        Args:
            weights (str): Path to a ``.pt`` or ``.pth`` checkpoint, or to a SAM3 export directory.
            task (str | None): Task name. If provided, it specifies the particular task the model is being loaded for.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")
        """
        self.ckpt_path = weights
        if self.is_exported_dir:
            from ultralytics.nn.backends.sam3 import SAM3Backend

            self.model = SAM3Backend(weights)
        elif self.is_sam3:
            from .build_sam3 import build_interactive_sam3

            self.model = build_interactive_sam3(weights)
        else:
            from .build import build_sam  # slow import

            self.model = build_sam(weights)

    def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, text=None, **kwargs):
        """Perform segmentation prediction on the given image or video source.

        Args:
            source (str | PIL.Image | np.ndarray): Path to the image or video file, or a PIL.Image object, or a
                np.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (list[list[float]] | None): List of points for prompted segmentation.
            labels (list[int] | None): List of labels for prompted segmentation.
            text (list[str] | None): Class names or phrases for SAM3 text prompted segmentation.
            **kwargs (Any): Additional keyword arguments for prediction.

        Returns:
            (list): The model predictions.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"Detected {len(r.masks)} masks")
        """
        # An export states the size it was traced at, so honor that over the checkpoint default.
        imgsz = getattr(self.model, "imgsz", None) if self.is_exported_dir else None
        overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": imgsz or 1024}
        kwargs = {**overrides, **kwargs, "retina_masks": True}
        prompts = {"bboxes": bboxes, "points": points, "labels": labels}
        if text is not None:
            prompts["text"] = text
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, text=None, **kwargs):
        """Perform segmentation prediction on the given image or video source.

        This method is an alias for the 'predict' method, providing a convenient way to call the SAM model for
        segmentation tasks.

        Args:
            source (str | PIL.Image | np.ndarray | None): Path to the image or video file, or a PIL.Image object, or a
                np.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (list[list[float]] | None): List of points for prompted segmentation.
            labels (list[int] | None): List of labels for prompted segmentation.
            text (list[str] | None): Class names or phrases for SAM3 text prompted segmentation.
            **kwargs (Any): Additional keyword arguments to be passed to the predict method.

        Returns:
            (list): The model predictions, typically containing segmentation masks and other relevant information.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"Detected {len(results[0].masks)} masks")
        """
        return self.predict(source, stream, bboxes, points, labels, text, **kwargs)

    def export(self, **kwargs):
        """Export SAM3 model to ONNX or TensorRT format.

        Writes four to six module files into a directory next to the checkpoint, named
        ``<stem>_onnx`` or ``<stem>_engine``: vision encoder, text encoder, decoder, and a
        geometry free text decoder, plus a prompt encoder and mask decoder when the checkpoint
        carries interactive weights. The directory suffix is what selects the inference backend,
        so keep it when moving the directory. Other SAM variants are not supported for export.

        An exported directory serves text, box, and point prompts on images only. Video tracking
        needs the memory-bank tracker, which is not exported, so keep the checkpoint for it.

        ``format="engine"`` goes all the way from the checkpoint on its own: it exports the ONNX
        modules first and then builds the engines from them, leaving both directories behind. There
        is no need to call ``format="onnx"`` beforehand, and doing so only exports it twice. The
        engine path always writes FP32 ONNX regardless of ``half``, because TensorRT applies FP16
        itself through mixed precision.

        Args:
            **kwargs (Any): Export arguments. Key options:
            format (str): ``"onnx"`` or ``"engine"`` (TensorRT).
            imgsz (int): Image size (must be divisible by 14). Default 1008. With ``dynamic`` this is the largest size
                the export accepts.
            dynamic (bool): Accept any image size from ``min_imgsz`` to ``imgsz`` instead of only ``imgsz``, so one
                export serves several sizes. Default False.
            min_imgsz (int): Smallest size accepted when ``dynamic``. Defaults to half of ``imgsz``.
            quantize (int): 16 for FP16. ONNX stores FP16 weights behind FP32 inputs and outputs, TensorRT instead picks
                a precision per node.
            half (bool): Deprecated alias for ``quantize=16``.
            device (str): Export device. Default ``"cpu"``.
            opset (int): ONNX opset version. Default 20.

        Returns:
            (str): Path to the output directory.

        Examples:
            One call takes the checkpoint all the way to TensorRT, writing the ONNX modules on the
            way and returning the engine directory::

                model = SAM("sam3.pt")
                model.export(format="engine", imgsz=1008, quantize=16)

            Export only the ONNX modules::

                model.export(format="onnx", imgsz=1008)
        """
        if not self.is_sam3:
            raise NotImplementedError("Export is only supported for SAM3 models.")
        if self.is_exported_dir:
            raise NotImplementedError(f"{self.ckpt_path} is already exported. Export from the .pt checkpoint instead.")

        from ultralytics.utils.export.sam3_onnx import export_sam3_engine, export_sam3_onnx

        fmt = kwargs.pop("format", "onnx")
        assert fmt in {"onnx", "engine"}, f"SAM3 export supports format='onnx' or 'engine', got '{fmt}'"

        imgsz = kwargs.pop("imgsz", 1008)
        dynamic = kwargs.pop("dynamic", False)
        min_imgsz = kwargs.pop("min_imgsz", None)
        # The CLI rewrites half into quantize, so accept both spellings or FP16 is silently dropped.
        quantize = kwargs.pop("quantize", None)
        half = kwargs.pop("half", quantize == 16)
        device = kwargs.pop("device", "cpu")
        opset = kwargs.pop("opset", 20)
        workspace = kwargs.pop("workspace", None)

        # For TRT: always export FP32 ONNX (TRT handles FP16 internally via mixed precision)
        onnx_half = half if fmt == "onnx" else False

        onnx_files = export_sam3_onnx(
            checkpoint_path=str(self.ckpt_path),
            device=device,
            opset=opset,
            half=onnx_half,
            imgsz=imgsz if isinstance(imgsz, int) else imgsz[0],
            dynamic=dynamic,
            min_imgsz=min_imgsz,
        )
        onnx_dir = str(Path(onnx_files[0]).parent)

        if fmt == "engine":
            engine_files = export_sam3_engine(
                onnx_dir=onnx_dir,
                half=half,
                workspace=workspace,
            )
            return str(Path(engine_files[0]).parent)

        return onnx_dir

    def info(self, detailed: bool = False, verbose: bool = True):
        """Log information about the SAM model.

        Args:
            detailed (bool): If True, displays detailed information about the model layers and operations.
            verbose (bool): If True, prints the information to the console.

        Returns:
            (tuple): A tuple containing the model's information (string representations of the model).

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # Print summary information
        """
        if self.is_exported_dir:
            LOGGER.info(repr(self.model))  # exported modules are graphs, not torch layers to summarize
            return None
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
        """Provide a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (dict[str, dict[str, type[Predictor]]]): A dictionary mapping the 'segment' task to its corresponding
                Predictor class. For SAM2 models, it maps to SAM2Predictor, otherwise to the standard Predictor.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}
        """
        # An export directory serves text, boxes and points from one backend, which is what
        # SAM3SemanticPredictor drives; a SAM3 checkpoint keeps the interactive predictor.
        if self.is_exported_dir:
            predictor = SAM3SemanticPredictor
        elif self.is_sam2:
            predictor = SAM2Predictor
        elif self.is_sam3:
            predictor = SAM3Predictor
        else:
            predictor = Predictor
        return {"segment": {"predictor": predictor}}
