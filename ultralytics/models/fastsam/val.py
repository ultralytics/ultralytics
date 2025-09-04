# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import types
from pathlib import Path

import torch

from ultralytics.models.yolo.segment import SegmentationValidator


class FastSAMValidator(SegmentationValidator):
    """
    Custom validation class for Fast SAM (Segment Anything Model) segmentation in Ultralytics YOLO framework.

    Extends the SegmentationValidator class, customizing the validation process specifically for Fast SAM. This class
    sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled
    to avoid errors during validation.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The data loader object used for validation.
        save_dir (Path): The directory where validation results will be saved.
        args (SimpleNamespace): Additional arguments for customization of the validation process.
        _callbacks (list): List of callback functions to be invoked during validation.
        metrics (SegmentMetrics): Segmentation metrics calculator for evaluation.

    Methods:
        __init__: Initialize the FastSAMValidator with custom settings for Fast SAM.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader | None = None,
        save_dir: Path | None = None,
        args: types.SimpleNamespace | None = None,
        _callbacks: list | None = None,
    ) -> None:
        """
        Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader | None): Dataloader to be used for validation.
            save_dir (Path | None): Directory to save results.
            args (types.SimpleNamespace | None): Configuration for the validator.
            _callbacks (list | None): List of callback functions to be invoked during validation.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
