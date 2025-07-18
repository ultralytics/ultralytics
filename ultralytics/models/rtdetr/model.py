# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware
    query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """
        Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        # Smart version detection and config selection for .pt files
        if str(model).endswith(".pt"):
            model_path = model
            # Auto-select config based on weight version
            config_path = self._auto_select_config(model_path)
            LOGGER.warning(f"RT-DETR: Auto-selected config '{config_path}' for weights '{model_path}'")
            super().__init__(model=config_path, task="detect")
            # Load weights after model initialization
            if config_path != model_path:  # Only load if we changed the config
                LOGGER.warning(f"RT-DETR: Loading weights from '{model_path}' with auto-detected configuration")
                self.load(model_path)
        else:
            super().__init__(model=model, task="detect")

    def _auto_select_config(self, weight_path):
        """
        Automatically select appropriate config based on weight file version.

        Args:
            weight_path (str): Path to the weight file.

        Returns:
            str: Path to the appropriate config file.
        """
        import torch

        try:
            # Load checkpoint to detect version
            checkpoint = torch.load(weight_path, map_location="cpu")
            model_state = checkpoint.get("model", checkpoint)

            # Detect version based on sampling_offsets dimensions
            version = self._detect_model_version(model_state)

            if version == "v2":
                # Use v2 config
                LOGGER.warning(f"RT-DETR: Detected v2 weights in '{weight_path}', using rtdetrv2-l.yaml config")
                return "rtdetrv2-l.yaml"
            else:
                # Use v1 config (default)
                LOGGER.warning(f"RT-DETR: Detected v1 weights in '{weight_path}', using rtdetr-l.yaml config")
                return "rtdetr-l.yaml"

        except Exception as e:
            LOGGER.warning(f"RT-DETR: Could not auto-detect version from {weight_path}: {e}")
            LOGGER.warning("RT-DETR: Using default v1 configuration (rtdetr-l.yaml)")
            return "rtdetr-l.yaml"

    def _detect_model_version(self, state_dict):
        """
        Detect RT-DETR model version from state dict.

        Args:
            state_dict (dict): Model state dictionary.

        Returns:
            str: Detected version ('v1' or 'v2').
        """
        # Look for decoder layer sampling_offsets to determine version
        for name, param in state_dict.items():
            if "decoder.layers" in name and "sampling_offsets" in name:
                # Check last dimension to determine version
                if param.shape[-1] == 64:  # v1: 8 * 4 * 2
                    return "v1"
                elif param.shape[-1] == 192:  # v2: 8 * 12 * 2
                    return "v2"

        # Default to v1 if no clear indicators
        return "v1"

    @property
    def task_map(self) -> dict:
        """
        Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
