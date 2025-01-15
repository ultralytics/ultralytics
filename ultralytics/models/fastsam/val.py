# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):
    """
    Custom validation class for fast SAM (Segment Anything Model) segmentation in Ultralytics YOLO framework.

    Extends the SegmentationValidator class, customizing the validation process specifically for fast SAM. This class
    sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled
    to avoid errors during validation.

    Attributes:
        dataloader: The data loader object used for validation.
        save_dir (str): The directory where validation results will be saved.
        pbar: A progress bar object.
        args: Additional arguments for customization.
        _callbacks: List of callback functions to be invoked during validation.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
