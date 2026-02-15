---
description: Learn how Ultralytics uses lazy imports with __getattr__ to speed up package startup and load model classes like YOLO, NAS, RTDETR, and SAM only when needed.
keywords: Ultralytics, YOLO, lazy import, __getattr__, NAS, RTDETR, SAM, FastSAM, YOLOWorld, performance, startup time, optimization
---

# Reference for `ultralytics/__init__.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/\_\_init\_\_.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/__init__.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`__getattr__`](#ultralytics.__init__.__getattr__)
        - [`__dir__`](#ultralytics.__init__.__dir__)


## Function `ultralytics.__getattr__` {#ultralytics.\_\_init\_\_.\_\_getattr\_\_}

```python
def __getattr__(name: str)
```

Lazy-import model classes on first access.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/__init__.py#L35-L39"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
```
</details>


<br><br><hr><br>

## Function `ultralytics.__dir__` {#ultralytics.\_\_init\_\_.\_\_dir\_\_}

```python
def __dir__()
```

Extend dir() to include lazily available model names for IDE autocompletion.

<details>
<summary>Source code in <code>ultralytics/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/__init__.py#L42-L44"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))
```
</details>

<br><br>
