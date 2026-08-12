---
title: models.__init__ API Reference
description: Reference for `ultralytics.models.__init__` in the Ultralytics package.
keywords: Ultralytics, ultralytics.models.__init__, API reference, YOLO, Python
---

# Reference for `ultralytics/models/__init__.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/\_\_init\_\_.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/__init__.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`__getattr__`](#ultralytics.models.__init__.__getattr__)


## Function `ultralytics.models.__getattr__` {#ultralytics.models.\_\_init\_\_.\_\_getattr\_\_}

```python
def __getattr__(name)
```

Lazy-import SAM so standard YOLO imports don't load optional torchvision internals.

<details>
<summary>Source code in <code>ultralytics/models/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/__init__.py#L12-L19">View on GitHub</a>
```python
def __getattr__(name):
    """Lazy-import SAM so standard YOLO imports don't load optional torchvision internals."""
    if name == "SAM":
        # Scoped for import ultralytics speed: SAM pulls optional torchvision-heavy modules.
        from .sam import SAM

        return SAM
    raise AttributeError(f"module {__name__} has no attribute {name}")
```
</details>

<br><br>
