# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis


class HUBModelError(Exception):
<<<<<<< HEAD
    """
    Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

    This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO.
    The error message is processed to include emojis for better user experience.
=======
    """Custom exception class for handling errors related to model fetching in Ultralytics YOLO.

    This exception is raised when a requested model is not found or cannot be retrieved. The message is also processed
    to include emojis for better user experience.
>>>>>>> 02121a52dd0a636899376093a514e43cc27a4435

    Attributes:
        message (str): The error message displayed when the exception is raised.

<<<<<<< HEAD
    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        >>> # Code that might fail to find a model
        >>>     raise HUBModelError("Custom model not found message")
        >>> except HUBModelError as e:
        >>>     print(e)  # Displays the emoji-enhanced error message
=======
    Notes:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
>>>>>>> 02121a52dd0a636899376093a514e43cc27a4435
    """

    def __init__(self, message="Model not found. Please check model URL and try again."):
        """
        Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB.
        The message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.

        Examples:
            >>> try:
            ...     raise HUBModelError("Custom model error message")
            ... except HUBModelError as e:
            ...     print(e)
        """
        super().__init__(emojis(message))
