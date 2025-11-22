# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.segment import SegmentationValidator


class FastSAMValidator(SegmentationValidator):
    """Custom validation class for Fast SAM (Segment Anything Model) segmentation in Ultralytics YOLO framework.

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

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (list, optional): List of callback functions to be invoked during validation.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
