# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import emojis


class HUBModelError(Exception):
    """
    Custom exception class for handling errors related to model fetching in Ultralytics YOLO.

    This exception is raised when a requested model is not found or cannot be retrieved.
    The message is also processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
    """

    def __init__(self, message="Model not found. Please check model URL and try again."):
        """Create an exception for when a model is not found."""
        super().__init__(emojis(message))


class DatasetError(Exception):
    """
    Custom exception class for handling errors related to dataset.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
    """

    def __init__(self, message="Dataset not found. Please check dataset and try again."):
        """Create an exception for when a dataset is not found or corrupted."""
        message = f"{message}\n \n If you're looking for assistance in building your dataset, be sure to explore the official documentation at: https://docs.ultralytics.com/datasets."
        super().__init__(emojis(message))


DatasetError.__module__ = "Ultralytics"  # To have Ultralytics.DatasetError
