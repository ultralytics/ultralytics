---
description: Explore error handling for Ultralytics YOLO. Learn about custom exceptions like HUBModelError to manage model fetching issues effectively.
keywords: Ultralytics, YOLO, error handling, HUBModelError, model fetching, custom exceptions, Python
---

# Reference for `ultralytics/utils/errors.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/errors.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/errors.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`HUBModelError`](#ultralytics.utils.errors.HUBModelError)


## Class `ultralytics.utils.errors.HUBModelError` {#ultralytics.utils.errors.HUBModelError}

```python
HUBModelError(self, message: str = "Model not found. Please check model URL and try again.")
```

**Bases:** `Exception`

Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO. The error message is processed to include emojis for better user experience.

This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB. The message is processed to include emojis for better user experience.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `message` | `str, optional` | The error message to display when the exception is raised. | `"Model not found. Please check model URL and try again."` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `message` | `str` | The error message displayed when the exception is raised. |

**Examples**

```python
>>> try:
...     # Code that might fail to find a model
...     raise HUBModelError("Custom model not found message")
... except HUBModelError as e:
...     print(e)  # Displays the emoji-enhanced error message
```

<details>
<summary>Source code in <code>ultralytics/utils/errors.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/errors.py#L6-L35"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class HUBModelError(Exception):
    """Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

    This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO. The
    error message is processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        ...     # Code that might fail to find a model
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # Displays the emoji-enhanced error message
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB. The
        message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.
        """
        super().__init__(emojis(message))
```
</details>

<br><br>
