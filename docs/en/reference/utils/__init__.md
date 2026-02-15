---
description: Explore the comprehensive reference for ultralytics.utils in the Ultralytics library. Enhance your ML workflow with these utility functions.
keywords: Ultralytics, utils, TQDM, Python, ML, Machine Learning utilities, YOLO, threading, logging, yaml, settings
---

# Reference for `ultralytics/utils/__init__.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/\_\_init\_\_.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DataExportMixin`](#ultralytics.utils.__init__.DataExportMixin)
        - [`SimpleClass`](#ultralytics.utils.__init__.SimpleClass)
        - [`IterableSimpleNamespace`](#ultralytics.utils.__init__.IterableSimpleNamespace)
        - [`ThreadingLocked`](#ultralytics.utils.__init__.ThreadingLocked)
        - [`YAML`](#ultralytics.utils.__init__.YAML)
        - [`TryExcept`](#ultralytics.utils.__init__.TryExcept)
        - [`Retry`](#ultralytics.utils.__init__.Retry)
        - [`JSONDict`](#ultralytics.utils.__init__.JSONDict)
        - [`SettingsManager`](#ultralytics.utils.__init__.SettingsManager)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DataExportMixin.to_df`](#ultralytics.utils.__init__.DataExportMixin.to_df)
        - [`DataExportMixin.to_csv`](#ultralytics.utils.__init__.DataExportMixin.to_csv)
        - [`DataExportMixin.to_json`](#ultralytics.utils.__init__.DataExportMixin.to_json)
        - [`SimpleClass.__str__`](#ultralytics.utils.__init__.SimpleClass.__str__)
        - [`SimpleClass.__repr__`](#ultralytics.utils.__init__.SimpleClass.__repr__)
        - [`SimpleClass.__getattr__`](#ultralytics.utils.__init__.SimpleClass.__getattr__)
        - [`IterableSimpleNamespace.__iter__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__iter__)
        - [`IterableSimpleNamespace.__str__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__str__)
        - [`IterableSimpleNamespace.__getattr__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__getattr__)
        - [`IterableSimpleNamespace.get`](#ultralytics.utils.__init__.IterableSimpleNamespace.get)
        - [`ThreadingLocked.__call__`](#ultralytics.utils.__init__.ThreadingLocked.__call__)
        - [`YAML._get_instance`](#ultralytics.utils.__init__.YAML._get_instance)
        - [`YAML.save`](#ultralytics.utils.__init__.YAML.save)
        - [`YAML.load`](#ultralytics.utils.__init__.YAML.load)
        - [`YAML.print`](#ultralytics.utils.__init__.YAML.print)
        - [`TryExcept.__enter__`](#ultralytics.utils.__init__.TryExcept.__enter__)
        - [`TryExcept.__exit__`](#ultralytics.utils.__init__.TryExcept.__exit__)
        - [`Retry.__call__`](#ultralytics.utils.__init__.Retry.__call__)
        - [`JSONDict._load`](#ultralytics.utils.__init__.JSONDict._load)
        - [`JSONDict._save`](#ultralytics.utils.__init__.JSONDict._save)
        - [`JSONDict._json_default`](#ultralytics.utils.__init__.JSONDict._json_default)
        - [`JSONDict.__setitem__`](#ultralytics.utils.__init__.JSONDict.__setitem__)
        - [`JSONDict.__delitem__`](#ultralytics.utils.__init__.JSONDict.__delitem__)
        - [`JSONDict.__str__`](#ultralytics.utils.__init__.JSONDict.__str__)
        - [`JSONDict.update`](#ultralytics.utils.__init__.JSONDict.update)
        - [`JSONDict.clear`](#ultralytics.utils.__init__.JSONDict.clear)
        - [`SettingsManager._validate_settings`](#ultralytics.utils.__init__.SettingsManager._validate_settings)
        - [`SettingsManager.__setitem__`](#ultralytics.utils.__init__.SettingsManager.__setitem__)
        - [`SettingsManager.update`](#ultralytics.utils.__init__.SettingsManager.update)
        - [`SettingsManager.reset`](#ultralytics.utils.__init__.SettingsManager.reset)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`plt_settings`](#ultralytics.utils.__init__.plt_settings)
        - [`set_logging`](#ultralytics.utils.__init__.set_logging)
        - [`emojis`](#ultralytics.utils.__init__.emojis)
        - [`read_device_model`](#ultralytics.utils.__init__.read_device_model)
        - [`is_ubuntu`](#ultralytics.utils.__init__.is_ubuntu)
        - [`is_debian`](#ultralytics.utils.__init__.is_debian)
        - [`is_colab`](#ultralytics.utils.__init__.is_colab)
        - [`is_kaggle`](#ultralytics.utils.__init__.is_kaggle)
        - [`is_jupyter`](#ultralytics.utils.__init__.is_jupyter)
        - [`is_runpod`](#ultralytics.utils.__init__.is_runpod)
        - [`is_docker`](#ultralytics.utils.__init__.is_docker)
        - [`is_raspberrypi`](#ultralytics.utils.__init__.is_raspberrypi)
        - [`is_jetson`](#ultralytics.utils.__init__.is_jetson)
        - [`is_dgx`](#ultralytics.utils.__init__.is_dgx)
        - [`is_online`](#ultralytics.utils.__init__.is_online)
        - [`is_pip_package`](#ultralytics.utils.__init__.is_pip_package)
        - [`is_dir_writeable`](#ultralytics.utils.__init__.is_dir_writeable)
        - [`is_pytest_running`](#ultralytics.utils.__init__.is_pytest_running)
        - [`is_github_action_running`](#ultralytics.utils.__init__.is_github_action_running)
        - [`get_default_args`](#ultralytics.utils.__init__.get_default_args)
        - [`get_ubuntu_version`](#ultralytics.utils.__init__.get_ubuntu_version)
        - [`get_user_config_dir`](#ultralytics.utils.__init__.get_user_config_dir)
        - [`colorstr`](#ultralytics.utils.__init__.colorstr)
        - [`remove_colorstr`](#ultralytics.utils.__init__.remove_colorstr)
        - [`threaded`](#ultralytics.utils.__init__.threaded)
        - [`set_sentry`](#ultralytics.utils.__init__.set_sentry)
        - [`deprecation_warn`](#ultralytics.utils.__init__.deprecation_warn)
        - [`clean_url`](#ultralytics.utils.__init__.clean_url)
        - [`url2file`](#ultralytics.utils.__init__.url2file)
        - [`vscode_msg`](#ultralytics.utils.__init__.vscode_msg)


## Class `ultralytics.utils.DataExportMixin` {#ultralytics.utils.\_\_init\_\_.DataExportMixin}

```python
DataExportMixin()
```

Mixin class for exporting validation metrics or prediction results in various formats.

This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results from classification, object detection, segmentation, or pose estimation tasks into various formats: Polars DataFrame, CSV, and JSON.

**Methods**

| Name | Description |
| --- | --- |
| [`to_csv`](#ultralytics.utils.__init__.DataExportMixin.to_csv) | Export results or metrics to CSV string format. |
| [`to_df`](#ultralytics.utils.__init__.DataExportMixin.to_df) | Create a Polars DataFrame from the prediction results summary or validation metrics. |
| [`to_json`](#ultralytics.utils.__init__.DataExportMixin.to_json) | Export results to JSON format. |

**Examples**

```python
>>> model = YOLO("yolo26n.pt")
>>> results = model("image.jpg")
>>> df = results.to_df()
>>> print(df)
>>> csv_data = results.to_csv()
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L150-L226"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DataExportMixin:
```
</details>

<br>

### Method `ultralytics.utils.DataExportMixin.to_csv` {#ultralytics.utils.\_\_init\_\_.DataExportMixin.to\_csv}

```python
def to_csv(self, normalize = False, decimals = 5)
```

Export results or metrics to CSV string format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool, optional` | Normalize numeric values. | `False` |
| `decimals` | `int, optional` | Decimal precision. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | CSV content as string. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L185-L214"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def to_csv(self, normalize=False, decimals=5):
    """Export results or metrics to CSV string format.

    Args:
        normalize (bool, optional): Normalize numeric values.
        decimals (int, optional): Decimal precision.

    Returns:
        (str): CSV content as string.
    """
    import polars as pl

    df = self.to_df(normalize=normalize, decimals=decimals)

    try:
        return df.write_csv()
    except Exception:
        # Minimal string conversion for any remaining complex types
        def _to_str_simple(v):
            if v is None:
                return ""
            elif isinstance(v, (dict, list, tuple, set)):
                return repr(v)
            else:
                return str(v)

        df_str = df.select(
            [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
        )
        return df_str.write_csv()
```
</details>

<br>

### Method `ultralytics.utils.DataExportMixin.to_df` {#ultralytics.utils.\_\_init\_\_.DataExportMixin.to\_df}

```python
def to_df(self, normalize = False, decimals = 5)
```

Create a Polars DataFrame from the prediction results summary or validation metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool, optional` | Normalize numerical values for easier comparison. | `False` |
| `decimals` | `int, optional` | Decimal places to round floats. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `polars.DataFrame` | Polars DataFrame containing the summary data. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L171-L183"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def to_df(self, normalize=False, decimals=5):
    """Create a Polars DataFrame from the prediction results summary or validation metrics.

    Args:
        normalize (bool, optional): Normalize numerical values for easier comparison.
        decimals (int, optional): Decimal places to round floats.

    Returns:
        (polars.DataFrame): Polars DataFrame containing the summary data.
    """
    import polars as pl  # scope for faster 'import ultralytics'

    return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))
```
</details>

<br>

### Method `ultralytics.utils.DataExportMixin.to_json` {#ultralytics.utils.\_\_init\_\_.DataExportMixin.to\_json}

```python
def to_json(self, normalize = False, decimals = 5)
```

Export results to JSON format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool, optional` | Normalize numeric values. | `False` |
| `decimals` | `int, optional` | Decimal precision. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `str` | JSON-formatted string of the results. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L216-L226"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def to_json(self, normalize=False, decimals=5):
    """Export results to JSON format.

    Args:
        normalize (bool, optional): Normalize numeric values.
        decimals (int, optional): Decimal precision.

    Returns:
        (str): JSON-formatted string of the results.
    """
    return self.to_df(normalize=normalize, decimals=decimals).write_json()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.SimpleClass` {#ultralytics.utils.\_\_init\_\_.SimpleClass}

```python
SimpleClass()
```

A simple base class for creating objects with string representations of their attributes.

This class provides a foundation for creating objects that can be easily printed or represented as strings, showing all their non-callable attributes. It's useful for debugging and introspection of object states.

**Methods**

| Name | Description |
| --- | --- |
| [`__getattr__`](#ultralytics.utils.__init__.SimpleClass.__getattr__) | Provide a custom attribute access error message with helpful information. |
| [`__repr__`](#ultralytics.utils.__init__.SimpleClass.__repr__) | Return a machine-readable string representation of the object. |
| [`__str__`](#ultralytics.utils.__init__.SimpleClass.__str__) | Return a human-readable string representation of the object. |

**Examples**

```python
>>> class MyClass(SimpleClass):
...     def __init__(self):
...         self.x = 10
...         self.y = "hello"
>>> obj = MyClass()
>>> print(obj)
__main__.MyClass object with attributes:

x: 10
y: 'hello'
```

!!! note "Notes"

    - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
    - The string representation includes the module and class name of the object.
    - Callable attributes and attributes starting with an underscore are excluded from the string representation.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L229-L279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SimpleClass:
```
</details>

<br>

### Method `ultralytics.utils.SimpleClass.__getattr__` {#ultralytics.utils.\_\_init\_\_.SimpleClass.\_\_getattr\_\_}

```python
def __getattr__(self, attr)
```

Provide a custom attribute access error message with helpful information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `attr` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L276-L279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getattr__(self, attr):
    """Provide a custom attribute access error message with helpful information."""
    name = self.__class__.__name__
    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
```
</details>

<br>

### Method `ultralytics.utils.SimpleClass.__repr__` {#ultralytics.utils.\_\_init\_\_.SimpleClass.\_\_repr\_\_}

```python
def __repr__(self)
```

Return a machine-readable string representation of the object.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L272-L274"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __repr__(self):
    """Return a machine-readable string representation of the object."""
    return self.__str__()
```
</details>

<br>

### Method `ultralytics.utils.SimpleClass.__str__` {#ultralytics.utils.\_\_init\_\_.SimpleClass.\_\_str\_\_}

```python
def __str__(self)
```

Return a human-readable string representation of the object.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L258-L270"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __str__(self):
    """Return a human-readable string representation of the object."""
    attr = []
    for a in dir(self):
        v = getattr(self, a)
        if not callable(v) and not a.startswith("_"):
            if isinstance(v, SimpleClass):
                # Display only the module and class name for subclasses
                s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
            else:
                s = f"{a}: {v!r}"
            attr.append(s)
    return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.IterableSimpleNamespace` {#ultralytics.utils.\_\_init\_\_.IterableSimpleNamespace}

```python
IterableSimpleNamespace()
```

**Bases:** `SimpleNamespace`

An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

This class extends the SimpleNamespace class with additional methods for iteration, string representation, and attribute access. It is designed to be used as a convenient container for storing and accessing configuration parameters.

**Methods**

| Name | Description |
| --- | --- |
| [`__getattr__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__getattr__) | Provide a custom attribute access error message with helpful information. |
| [`__iter__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__iter__) | Return an iterator of key-value pairs from the namespace's attributes. |
| [`__str__`](#ultralytics.utils.__init__.IterableSimpleNamespace.__str__) | Return a human-readable string representation of the object. |
| [`get`](#ultralytics.utils.__init__.IterableSimpleNamespace.get) | Return the value of the specified key if it exists; otherwise, return the default value. |

**Examples**

```python
>>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
>>> for k, v in cfg:
...     print(f"{k}: {v}")
a: 1
b: 2
c: 3
>>> print(cfg)
a=1
b=2
c=3
>>> cfg.get("b")
2
>>> cfg.get("d", "default")
'default'
```

!!! note "Notes"

    This class is particularly useful for storing configuration parameters in a more accessible
    and iterable format compared to a standard dictionary.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L282-L338"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class IterableSimpleNamespace(SimpleNamespace):
```
</details>

<br>

### Method `ultralytics.utils.IterableSimpleNamespace.__getattr__` {#ultralytics.utils.\_\_init\_\_.IterableSimpleNamespace.\_\_getattr\_\_}

```python
def __getattr__(self, attr)
```

Provide a custom attribute access error message with helpful information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `attr` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L324-L334"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __getattr__(self, attr):
    """Provide a custom attribute access error message with helpful information."""
    name = self.__class__.__name__
    raise AttributeError(
        f"""
        '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
        'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
        {DEFAULT_CFG_PATH} with the latest version from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
        """
    )
```
</details>

<br>

### Method `ultralytics.utils.IterableSimpleNamespace.__iter__` {#ultralytics.utils.\_\_init\_\_.IterableSimpleNamespace.\_\_iter\_\_}

```python
def __iter__(self)
```

Return an iterator of key-value pairs from the namespace's attributes.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L316-L318"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __iter__(self):
    """Return an iterator of key-value pairs from the namespace's attributes."""
    return iter(vars(self).items())
```
</details>

<br>

### Method `ultralytics.utils.IterableSimpleNamespace.__str__` {#ultralytics.utils.\_\_init\_\_.IterableSimpleNamespace.\_\_str\_\_}

```python
def __str__(self)
```

Return a human-readable string representation of the object.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L320-L322"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __str__(self):
    """Return a human-readable string representation of the object."""
    return "\n".join(f"{k}={v}" for k, v in vars(self).items())
```
</details>

<br>

### Method `ultralytics.utils.IterableSimpleNamespace.get` {#ultralytics.utils.\_\_init\_\_.IterableSimpleNamespace.get}

```python
def get(self, key, default = None)
```

Return the value of the specified key if it exists; otherwise, return the default value.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` |  |  | *required* |
| `default` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L336-L338"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get(self, key, default=None):
    """Return the value of the specified key if it exists; otherwise, return the default value."""
    return getattr(self, key, default)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.ThreadingLocked` {#ultralytics.utils.\_\_init\_\_.ThreadingLocked}

```python
ThreadingLocked(self)
```

A decorator class for ensuring thread-safe execution of a function or method.

This class can be used as a decorator to make sure that if the decorated function is called from multiple threads, only one thread at a time will be able to execute the function.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `lock` | `threading.Lock` | A lock object used to manage access to the decorated function. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.__init__.ThreadingLocked.__call__) | Run thread-safe execution of function or method. |

**Examples**

```python
>>> from ultralytics.utils import ThreadingLocked
>>> @ThreadingLocked()
... def my_function():
...    # Your code here
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L473-L503"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ThreadingLocked:
    """A decorator class for ensuring thread-safe execution of a function or method.

    This class can be used as a decorator to make sure that if the decorated function is called from multiple threads,
    only one thread at a time will be able to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Examples:
        >>> from ultralytics.utils import ThreadingLocked
        >>> @ThreadingLocked()
        ... def my_function():
        ...    # Your code here
    """

    def __init__(self):
        """Initialize the decorator class with a threading lock."""
        self.lock = threading.Lock()
```
</details>

<br>

### Method `ultralytics.utils.ThreadingLocked.__call__` {#ultralytics.utils.\_\_init\_\_.ThreadingLocked.\_\_call\_\_}

```python
def __call__(self, f)
```

Run thread-safe execution of function or method.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `f` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L493-L503"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, f):
    """Run thread-safe execution of function or method."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        """Apply thread-safety to the decorated function or method."""
        with self.lock:
            return f(*args, **kwargs)

    return decorated
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.YAML` {#ultralytics.utils.\_\_init\_\_.YAML}

```python
YAML(self)
```

YAML utility class for efficient file operations with automatic C-implementation detection.

This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
(C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
usage without explicit instantiation. The class handles file path creation, validation, and character encoding
issues automatically.

The implementation prioritizes performance through:
    - Automatic C-based loader/dumper selection when available
    - Singleton pattern to reuse the same instance
    - Lazy initialization to defer import costs until needed
    - Fallback mechanisms for handling problematic YAML content

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `_instance` |  | Internal singleton instance storage. |
| `yaml` |  | Reference to the PyYAML module. |
| `SafeLoader` |  | Best available YAML loader (CSafeLoader if available). |
| `SafeDumper` |  | Best available YAML dumper (CSafeDumper if available). |

**Methods**

| Name | Description |
| --- | --- |
| [`_get_instance`](#ultralytics.utils.__init__.YAML._get_instance) | Initialize singleton instance on first use. |
| [`load`](#ultralytics.utils.__init__.YAML.load) | Load YAML file to Python object with robust error handling. |
| [`print`](#ultralytics.utils.__init__.YAML.print) | Pretty print YAML file or object to console. |
| [`save`](#ultralytics.utils.__init__.YAML.save) | Save Python object as YAML file. |

**Examples**

```python
>>> data = YAML.load("config.yaml")
>>> data["new_value"] = 123
>>> YAML.save("updated_config.yaml", data)
>>> YAML.print(data)
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L506-L633"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class YAML:
    """YAML utility class for efficient file operations with automatic C-implementation detection.

    This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
    (C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
    usage without explicit instantiation. The class handles file path creation, validation, and character encoding
    issues automatically.

    The implementation prioritizes performance through:
        - Automatic C-based loader/dumper selection when available
        - Singleton pattern to reuse the same instance
        - Lazy initialization to defer import costs until needed
        - Fallback mechanisms for handling problematic YAML content

    Attributes:
        _instance: Internal singleton instance storage.
        yaml: Reference to the PyYAML module.
        SafeLoader: Best available YAML loader (CSafeLoader if available).
        SafeDumper: Best available YAML dumper (CSafeDumper if available).

    Examples:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """Initialize singleton instance on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize with optimal YAML implementation (C-based when available)."""
        import yaml

        self.yaml = yaml
        # Use C-based implementation if available for better performance
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper
```
</details>

<br>

### Method `ultralytics.utils.YAML._get_instance` {#ultralytics.utils.\_\_init\_\_.YAML.\_get\_instance}

```python
def _get_instance(cls)
```

Initialize singleton instance on first use.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L536-L540"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def _get_instance(cls):
    """Initialize singleton instance on first use."""
    if cls._instance is None:
        cls._instance = cls()
    return cls._instance
```
</details>

<br>

### Method `ultralytics.utils.YAML.load` {#ultralytics.utils.\_\_init\_\_.YAML.load}

```python
def load(cls, file = "data.yaml", append_filename = False)
```

Load YAML file to Python object with robust error handling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `str | Path` | Path to YAML file. | `"data.yaml"` |
| `append_filename` | `bool` | Whether to add filename to returned dict. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Loaded YAML content. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L585-L616"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def load(cls, file="data.yaml", append_filename=False):
    """Load YAML file to Python object with robust error handling.

    Args:
        file (str | Path): Path to YAML file.
        append_filename (bool): Whether to add filename to returned dict.

    Returns:
        (dict): Loaded YAML content.
    """
    instance = cls._get_instance()
    assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

    # Read file content
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()

    # Try loading YAML with fallback for problematic characters
    try:
        data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
    except Exception:
        # Remove problematic characters and retry
        s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

    # Check for accidental user-error None strings (should be 'null' in YAML)
    if "None" in data.values():
        data = {k: None if v == "None" else v for k, v in data.items()}

    if append_filename:
        data["yaml_file"] = str(file)
    return data
```
</details>

<br>

### Method `ultralytics.utils.YAML.print` {#ultralytics.utils.\_\_init\_\_.YAML.print}

```python
def print(cls, yaml_file)
```

Pretty print YAML file or object to console.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `yaml_file` | `str | Path | dict` | Path to YAML file or dict to print. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L619-L633"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def print(cls, yaml_file):
    """Pretty print YAML file or object to console.

    Args:
        yaml_file (str | Path | dict): Path to YAML file or dict to print.
    """
    instance = cls._get_instance()

    # Load file if path provided
    yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

    # Use -1 for unlimited width in C implementation
    dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")
```
</details>

<br>

### Method `ultralytics.utils.YAML.save` {#ultralytics.utils.\_\_init\_\_.YAML.save}

```python
def save(cls, file = "data.yaml", data = None, header = "")
```

Save Python object as YAML file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `str | Path` | Path to save YAML file. | `"data.yaml"` |
| `data` | `dict | None` | Dict or compatible object to save. | `None` |
| `header` | `str` | Optional string to add at file beginning. | `""` |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L556-L582"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@classmethod
def save(cls, file="data.yaml", data=None, header=""):
    """Save Python object as YAML file.

    Args:
        file (str | Path): Path to save YAML file.
        data (dict | None): Dict or compatible object to save.
        header (str): Optional string to add at file beginning.
    """
    instance = cls._get_instance()
    if data is None:
        data = {}

    # Create parent directories if needed
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Write YAML file
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.TryExcept` {#ultralytics.utils.\_\_init\_\_.TryExcept}

```python
TryExcept(self, msg = "", verbose = True)
```

**Bases:** `contextlib.ContextDecorator`

Ultralytics TryExcept class for handling exceptions gracefully.

This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages. It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `msg` |  |  | `""` |
| `verbose` |  |  | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `msg` | `str` | Optional message to display when an exception occurs. |
| `verbose` | `bool` | Whether to print the exception message. |

**Methods**

| Name | Description |
| --- | --- |
| [`__enter__`](#ultralytics.utils.__init__.TryExcept.__enter__) | Execute when entering TryExcept context, initialize instance. |
| [`__exit__`](#ultralytics.utils.__init__.TryExcept.__exit__) | Define behavior when exiting a 'with' block, print error message if necessary. |

**Examples**

```python
As a decorator:
>>> @TryExcept(msg="Error occurred in func", verbose=True)
... def func():
...     # Function logic here
...     pass

As a context manager:
>>> with TryExcept(msg="Error occurred in block", verbose=True):
...     # Code block here
...     pass
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1010-L1046"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TryExcept(contextlib.ContextDecorator):
    """Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        ... def func():
        ...     # Function logic here
        ...     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        ...     # Code block here
        ...     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose
```
</details>

<br>

### Method `ultralytics.utils.TryExcept.__enter__` {#ultralytics.utils.\_\_init\_\_.TryExcept.\_\_enter\_\_}

```python
def __enter__(self)
```

Execute when entering TryExcept context, initialize instance.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1038-L1040"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __enter__(self):
    """Execute when entering TryExcept context, initialize instance."""
    pass
```
</details>

<br>

### Method `ultralytics.utils.TryExcept.__exit__` {#ultralytics.utils.\_\_init\_\_.TryExcept.\_\_exit\_\_}

```python
def __exit__(self, exc_type, value, traceback)
```

Define behavior when exiting a 'with' block, print error message if necessary.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `exc_type` |  |  | *required* |
| `value` |  |  | *required* |
| `traceback` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1042-L1046"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __exit__(self, exc_type, value, traceback):
    """Define behavior when exiting a 'with' block, print error message if necessary."""
    if self.verbose and value:
        LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
    return True
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.Retry` {#ultralytics.utils.\_\_init\_\_.Retry}

```python
Retry(self, times = 3, delay = 2)
```

**Bases:** `contextlib.ContextDecorator`

Retry class for function execution with exponential backoff.

This decorator can be used to retry a function on exceptions, up to a specified number of times with an exponentially increasing delay between retries. It's useful for handling transient failures in network operations or other unreliable processes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `times` |  |  | `3` |
| `delay` |  |  | `2` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `times` | `int` | Maximum number of retry attempts. |
| `delay` | `int` | Initial delay between retries in seconds. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.__init__.Retry.__call__) | Decorator implementation for Retry with exponential backoff. |

**Examples**

```python
Example usage as a decorator:
>>> @Retry(times=3, delay=2)
... def test_func():
...     # Replace with function logic that may raise exceptions
...     return True
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1049-L1090"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Retry(contextlib.ContextDecorator):
    """Retry class for function execution with exponential backoff.

    This decorator can be used to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries. It's useful for handling transient failures in network operations or
    other unreliable processes.

    Attributes:
        times (int): Maximum number of retry attempts.
        delay (int): Initial delay between retries in seconds.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        ... def test_func():
        ...     # Replace with function logic that may raise exceptions
        ...     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0
```
</details>

<br>

### Method `ultralytics.utils.Retry.__call__` {#ultralytics.utils.\_\_init\_\_.Retry.\_\_call\_\_}

```python
def __call__(self, func)
```

Decorator implementation for Retry with exponential backoff.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1074-L1090"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, func):
    """Decorator implementation for Retry with exponential backoff."""

    def wrapped_func(*args, **kwargs):
        """Apply retries to the decorated function or method."""
        self._attempts = 0
        while self._attempts < self.times:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self._attempts += 1
                LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                if self._attempts >= self.times:
                    raise e
                time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

    return wrapped_func
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.JSONDict` {#ultralytics.utils.\_\_init\_\_.JSONDict}

```python
JSONDict(self, file_path: str | Path = "data.json")
```

**Bases:** `dict`

A dictionary-like class that provides JSON persistence for its contents.

This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are modified. It ensures thread-safe operations using a lock and handles JSON serialization of Path objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str | Path` |  | `"data.json"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `file_path` | `Path` | The path to the JSON file used for persistence. |
| `lock` | `threading.Lock` | A lock object to ensure thread-safe operations. |

**Methods**

| Name | Description |
| --- | --- |
| [`__delitem__`](#ultralytics.utils.__init__.JSONDict.__delitem__) | Remove an item and update the persistent storage. |
| [`__setitem__`](#ultralytics.utils.__init__.JSONDict.__setitem__) | Store a key-value pair and persist to disk. |
| [`__str__`](#ultralytics.utils.__init__.JSONDict.__str__) | Return a pretty-printed JSON string representation of the dictionary. |
| [`_json_default`](#ultralytics.utils.__init__.JSONDict._json_default) | Handle JSON serialization of Path objects. |
| [`_load`](#ultralytics.utils.__init__.JSONDict._load) | Load the data from the JSON file into the dictionary. |
| [`_save`](#ultralytics.utils.__init__.JSONDict._save) | Save the current state of the dictionary to the JSON file. |
| [`clear`](#ultralytics.utils.__init__.JSONDict.clear) | Clear all entries and update the persistent storage. |
| [`update`](#ultralytics.utils.__init__.JSONDict.update) | Update the dictionary and persist changes. |

**Examples**

```python
>>> json_dict = JSONDict("data.json")
>>> json_dict["key"] = "value"
>>> print(json_dict["key"])
value
>>> del json_dict["key"]
>>> json_dict.update({"new_key": "new_value"})
>>> json_dict.clear()
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1195-L1285"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class JSONDict(dict):
    """A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock and handles JSON serialization of Path objects.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Load the data from the JSON file into the dictionary.
        _save: Save the current state of the dictionary to the JSON file.
        __setitem__: Store a key-value pair and persist it to disk.
        __delitem__: Remove an item and update the persistent storage.
        update: Update the dictionary and persist changes.
        clear: Clear all entries and update the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: str | Path = "data.json"):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()
```
</details>

<br>

### Method `ultralytics.utils.JSONDict.__delitem__` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_\_delitem\_\_}

```python
def __delitem__(self, key)
```

Remove an item and update the persistent storage.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1264-L1268"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __delitem__(self, key):
    """Remove an item and update the persistent storage."""
    with self.lock:
        super().__delitem__(key)
        self._save()
```
</details>

<br>

### Method `ultralytics.utils.JSONDict.__setitem__` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_\_setitem\_\_}

```python
def __setitem__(self, key, value)
```

Store a key-value pair and persist to disk.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` |  |  | *required* |
| `value` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1258-L1262"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __setitem__(self, key, value):
    """Store a key-value pair and persist to disk."""
    with self.lock:
        super().__setitem__(key, value)
        self._save()
```
</details>

<br>

### Method `ultralytics.utils.JSONDict.__str__` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_\_str\_\_}

```python
def __str__(self)
```

Return a pretty-printed JSON string representation of the dictionary.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1270-L1273"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __str__(self):
    """Return a pretty-printed JSON string representation of the dictionary."""
    contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
    return f'JSONDict("{self.file_path}"):\n{contents}'
```
</details>

<br>

### Method `ultralytics.utils.JSONDict._json_default` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_json\_default}

```python
def _json_default(obj)
```

Handle JSON serialization of Path objects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `obj` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1252-L1256"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _json_default(obj):
    """Handle JSON serialization of Path objects."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
```
</details>

<br>

### Method `ultralytics.utils.JSONDict._load` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_load}

```python
def _load(self)
```

Load the data from the JSON file into the dictionary.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1230-L1240"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _load(self):
    """Load the data from the JSON file into the dictionary."""
    try:
        if self.file_path.exists():
            with open(self.file_path) as f:
                # Use the base dict update to avoid persisting during reads
                super().update(json.load(f))
    except json.JSONDecodeError:
        LOGGER.warning(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
    except Exception as e:
        LOGGER.error(f"Error reading from {self.file_path}: {e}")
```
</details>

<br>

### Method `ultralytics.utils.JSONDict._save` {#ultralytics.utils.\_\_init\_\_.JSONDict.\_save}

```python
def _save(self)
```

Save the current state of the dictionary to the JSON file.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1242-L1249"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _save(self):
    """Save the current state of the dictionary to the JSON file."""
    try:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(dict(self), f, indent=2, default=self._json_default)
    except Exception as e:
        LOGGER.error(f"Error writing to {self.file_path}: {e}")
```
</details>

<br>

### Method `ultralytics.utils.JSONDict.clear` {#ultralytics.utils.\_\_init\_\_.JSONDict.clear}

```python
def clear(self)
```

Clear all entries and update the persistent storage.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1281-L1285"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clear(self):
    """Clear all entries and update the persistent storage."""
    with self.lock:
        super().clear()
        self._save()
```
</details>

<br>

### Method `ultralytics.utils.JSONDict.update` {#ultralytics.utils.\_\_init\_\_.JSONDict.update}

```python
def update(self, *args, **kwargs)
```

Update the dictionary and persist changes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` |  |  | *required* |
| `**kwargs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1275-L1279"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, *args, **kwargs):
    """Update the dictionary and persist changes."""
    with self.lock:
        super().update(*args, **kwargs)
        self._save()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.SettingsManager` {#ultralytics.utils.\_\_init\_\_.SettingsManager}

```python
SettingsManager(self, file = SETTINGS_FILE, version = "0.0.6")
```

**Bases:** `JSONDict`

SettingsManager class for managing and persisting Ultralytics settings.

This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default values. It validates settings on initialization and provides methods to update or reset settings. The settings include directories for datasets, weights, and runs, as well as various integration flags.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` |  |  | `SETTINGS_FILE` |
| `version` |  |  | `"0.0.6"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `file` | `Path` | The path to the JSON file used for persistence. |
| `version` | `str` | The version of the settings schema. |
| `defaults` | `dict` | A dictionary containing default settings. |
| `help_msg` | `str` | A help message for users on how to view and update settings. |

**Methods**

| Name | Description |
| --- | --- |
| [`__setitem__`](#ultralytics.utils.__init__.SettingsManager.__setitem__) | Update one key: value pair. |
| [`_validate_settings`](#ultralytics.utils.__init__.SettingsManager._validate_settings) | Validate the current settings and reset if necessary. |
| [`reset`](#ultralytics.utils.__init__.SettingsManager.reset) | Reset the settings to default and save them. |
| [`update`](#ultralytics.utils.__init__.SettingsManager.update) | Update settings, validating keys and types. |

**Examples**

```python
Initialize and update settings:
>>> settings = SettingsManager()
>>> settings.update(runs_dir="/new/runs/dir")
>>> print(settings["runs_dir"])
/new/runs/dir
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1288-L1405"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SettingsManager(JSONDict):
    """SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings. The settings
    include directories for datasets, weights, and runs, as well as various integration flags.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validate the current settings and reset if necessary.
        update: Update settings, validating keys and types.
        reset: Reset the settings to default and save them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """Initialize the SettingsManager with default settings and load user settings."""
        import hashlib
        import uuid

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (root.parent if GIT.root and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # Settings schema version
            "datasets_dir": str(datasets_root / "datasets"),  # Datasets directory
            "weights_dir": str(root / "weights"),  # Model weights directory
            "runs_dir": str(root / "runs"),  # Experiment runs directory
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
            "sync": True,  # Enable synchronization
            "api_key": "",  # Ultralytics API Key
            "openai_api_key": "",  # OpenAI API Key
            "clearml": True,  # ClearML integration
            "comet": True,  # Comet integration
            "dvc": True,  # DVC integration
            "hub": True,  # Ultralytics HUB integration
            "mlflow": True,  # MLflow integration
            "neptune": True,  # Neptune integration
            "raytune": True,  # Ray Tune integration
            "tensorboard": False,  # TensorBoard logging
            "wandb": False,  # Weights & Biases logging
            "vscode_msg": True,  # VSCode message
            "openvino_msg": True,  # OpenVINO export on Intel CPU message
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # Check if file doesn't exist or is empty
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()
```
</details>

<br>

### Method `ultralytics.utils.SettingsManager.__setitem__` {#ultralytics.utils.\_\_init\_\_.SettingsManager.\_\_setitem\_\_}

```python
def __setitem__(self, key, value)
```

Update one key: value pair.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` |  |  | *required* |
| `value` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1383-L1385"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __setitem__(self, key, value):
    """Update one key: value pair."""
    self.update({key: value})
```
</details>

<br>

### Method `ultralytics.utils.SettingsManager._validate_settings` {#ultralytics.utils.\_\_init\_\_.SettingsManager.\_validate\_settings}

```python
def _validate_settings(self)
```

Validate the current settings and reset if necessary.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1363-L1381"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _validate_settings(self):
    """Validate the current settings and reset if necessary."""
    correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
    correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
    correct_version = self.get("settings_version", "") == self.version

    if not (correct_keys and correct_types and correct_version):
        LOGGER.warning(
            "Ultralytics settings reset to default values. This may be due to a possible problem "
            f"with your settings or a recent ultralytics package update. {self.help_msg}"
        )
        self.reset()

    if self.get("datasets_dir") == self.get("runs_dir"):
        LOGGER.warning(
            f"Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
            f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
            f"Please change one to avoid possible issues during training. {self.help_msg}"
        )
```
</details>

<br>

### Method `ultralytics.utils.SettingsManager.reset` {#ultralytics.utils.\_\_init\_\_.SettingsManager.reset}

```python
def reset(self)
```

Reset the settings to default and save them.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1402-L1405"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def reset(self):
    """Reset the settings to default and save them."""
    self.clear()
    self.update(self.defaults)
```
</details>

<br>

### Method `ultralytics.utils.SettingsManager.update` {#ultralytics.utils.\_\_init\_\_.SettingsManager.update}

```python
def update(self, *args, **kwargs)
```

Update settings, validating keys and types.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` |  |  | *required* |
| `**kwargs` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1387-L1400"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, *args, **kwargs):
    """Update settings, validating keys and types."""
    for arg in args:
        if isinstance(arg, dict):
            kwargs.update(arg)
    for k, v in kwargs.items():
        if k not in self.defaults:
            raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
        t = type(self.defaults[k])
        if not isinstance(v, t):
            raise TypeError(
                f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
            )
    super().update(*args, **kwargs)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.plt_settings` {#ultralytics.utils.\_\_init\_\_.plt\_settings}

```python
def plt_settings(rcparams = None, backend = "Agg")
```

Decorator to temporarily set rc parameters and the backend for a plotting function.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `rcparams` | `dict, optional` | Dictionary of rc parameters to set. | `None` |
| `backend` | `str, optional` | Name of the backend to use. | `"Agg"` |

**Returns**

| Type | Description |
| --- | --- |
| `Callable` | Decorated function with temporarily set rc parameters and backend. |

**Examples**

```python
>>> @plt_settings({"font.size": 12})
... def plot_function():
...     plt.figure()
...     plt.plot([1, 2, 3])
...     plt.show()

>>> with plt_settings({"font.size": 12}):
...     plt.figure()
...     plt.plot([1, 2, 3])
...     plt.show()
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L341-L391"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plt_settings(rcparams=None, backend="Agg"):
    """Decorator to temporarily set rc parameters and the backend for a plotting function.

    Args:
        rcparams (dict, optional): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend.

    Examples:
        >>> @plt_settings({"font.size": 12})
        ... def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Set rc parameters and backend, call the original function, and restore the settings."""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.set_logging` {#ultralytics.utils.\_\_init\_\_.set\_logging}

```python
def set_logging(name = "LOGGING_NAME", verbose = True)
```

Set up logging with UTF-8 encoding and configurable verbosity.

This function configures logging for the Ultralytics library, setting the appropriate logging level and formatter based on the verbosity flag and the current process rank. It handles special cases for Windows environments where UTF-8 encoding might not be the default.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | Name of the logger. | `"LOGGING_NAME"` |
| `verbose` | `bool` | Flag to set logging level to INFO if True, ERROR otherwise. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `logging.Logger` | Configured logger object. |

**Examples**

```python
>>> set_logging(name="ultralytics", verbose=True)
>>> logger = logging.getLogger("ultralytics")
>>> logger.info("This is an info message")
```

!!! note "Notes"

    - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
    - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
    - The function sets up a StreamHandler with the appropriate formatter and level.
    - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L394-L460"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_logging(name="LOGGING_NAME", verbose=True):
    """Set up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and formatter
    based on the verbosity flag and the current process rank. It handles special cases for Windows environments where
    UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """Format log records with prefixes based on level."""
            # Apply prefixes based on log level
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"

            # Handle emojis in message based on platform
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # Handle Windows UTF-8 encoding issues
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.emojis` {#ultralytics.utils.\_\_init\_\_.emojis}

```python
def emojis(string = "")
```

Return platform-dependent emoji-safe version of string.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `string` |  |  | `""` |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L468-L470"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.read_device_model` {#ultralytics.utils.\_\_init\_\_.read\_device\_model}

```python
def read_device_model() -> str
```

Read the device model information from the system.

**Returns**

| Type | Description |
| --- | --- |
| `str` | Platform release string in lowercase, used to identify device models like Jetson or Raspberry Pi. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L642-L648"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def read_device_model() -> str:
    """Read the device model information from the system.

    Returns:
        (str): Platform release string in lowercase, used to identify device models like Jetson or Raspberry Pi.
    """
    return platform.release().lower()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_ubuntu` {#ultralytics.utils.\_\_init\_\_.is\_ubuntu}

```python
def is_ubuntu() -> bool
```

Check if the OS is Ubuntu.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if OS is Ubuntu, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L651-L661"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_ubuntu() -> bool:
    """Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_debian` {#ultralytics.utils.\_\_init\_\_.is\_debian}

```python
def is_debian(codenames: list[str] | None | str = None) -> list[bool] | bool
```

Check if the OS is Debian.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `codenames` | `list[str] | None | str` | Specific Debian codename to check for (e.g., 'buster', 'bullseye'). If None,<br>    only checks for Debian. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `list[bool] | bool` | List of booleans indicating if OS matches each Debian codename, or a single boolean if no |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L664-L687"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_debian(codenames: list[str] | None | str = None) -> list[bool] | bool:
    """Check if the OS is Debian.

    Args:
        codenames (list[str] | None | str): Specific Debian codename to check for (e.g., 'buster', 'bullseye'). If None,
            only checks for Debian.

    Returns:
        (list[bool] | bool): List of booleans indicating if OS matches each Debian codename, or a single boolean if no
            codenames provided.
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_colab` {#ultralytics.utils.\_\_init\_\_.is\_colab}

```python
def is_colab()
```

Check if the current script is running inside a Google Colab notebook.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running inside a Colab notebook, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L690-L696"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_colab():
    """Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_kaggle` {#ultralytics.utils.\_\_init\_\_.is\_kaggle}

```python
def is_kaggle()
```

Check if the current script is running inside a Kaggle kernel.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running inside a Kaggle kernel, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L699-L705"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_kaggle():
    """Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_jupyter` {#ultralytics.utils.\_\_init\_\_.is\_jupyter}

```python
def is_jupyter()
```

Check if the current script is running inside a Jupyter Notebook.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running inside a Jupyter Notebook, False otherwise. |

!!! note "Notes"

    - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
    - "get_ipython" in globals() method suffers false positives when IPython package installed manually.

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L708-L718"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_jupyter():
    """Check if the current script is running inside a Jupyter Notebook.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.

    Notes:
        - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
        - "get_ipython" in globals() method suffers false positives when IPython package installed manually.
    """
    return IS_COLAB or IS_KAGGLE
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_runpod` {#ultralytics.utils.\_\_init\_\_.is\_runpod}

```python
def is_runpod()
```

Check if the current script is running inside a RunPod container.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running in RunPod, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L721-L727"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_runpod():
    """Check if the current script is running inside a RunPod container.

    Returns:
        (bool): True if running in RunPod, False otherwise.
    """
    return "RUNPOD_POD_ID" in os.environ
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_docker` {#ultralytics.utils.\_\_init\_\_.is\_docker}

```python
def is_docker() -> bool
```

Determine if the script is running inside a Docker container.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if the script is running inside a Docker container, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L730-L739"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_docker() -> bool:
    """Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_raspberrypi` {#ultralytics.utils.\_\_init\_\_.is\_raspberrypi}

```python
def is_raspberrypi() -> bool
```

Determine if the Python environment is running on a Raspberry Pi.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running on a Raspberry Pi, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L742-L748"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_raspberrypi() -> bool:
    """Determine if the Python environment is running on a Raspberry Pi.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return "rpi" in DEVICE_MODEL
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_jetson` {#ultralytics.utils.\_\_init\_\_.is\_jetson}

```python
def is_jetson(jetpack = None) -> bool
```

Determine if the Python environment is running on an NVIDIA Jetson device.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `jetpack` | `int | None` | If specified, check for specific JetPack version (4, 5, 6). | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running on an NVIDIA Jetson device, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L752-L769"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """Determine if the Python environment is running on an NVIDIA Jetson device.

    Args:
        jetpack (int | None): If specified, check for specific JetPack version (4, 5, 6).

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = open("/etc/nv_tegra_release").read()
            version_map = {4: "R32", 5: "R35", 6: "R36", 7: "R38"}  # JetPack to L4T major version mapping
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_dgx` {#ultralytics.utils.\_\_init\_\_.is\_dgx}

```python
def is_dgx() -> bool
```

Check if the current script is running inside a DGX (NVIDIA Data Center GPU), DGX-Ready or DGX Spark system.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if running in a DGX or DGX-Ready or DGX Spark system, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L772-L782"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_dgx() -> bool:
    """Check if the current script is running inside a DGX (NVIDIA Data Center GPU), DGX-Ready or DGX Spark system.

    Returns:
        (bool): True if running in a DGX or DGX-Ready or DGX Spark system, False otherwise.
    """
    try:
        with open("/etc/dgx-release") as f:
            return "DGX" in f.read()
    except FileNotFoundError:
        return False
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_online` {#ultralytics.utils.\_\_init\_\_.is\_online}

```python
def is_online() -> bool
```

Fast online check using DNS (v4/v6) resolution (Cloudflare + Google).

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if connection is successful, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L785-L800"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_online() -> bool:
    """Fast online check using DNS (v4/v6) resolution (Cloudflare + Google).

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_pip_package` {#ultralytics.utils.\_\_init\_\_.is\_pip\_package}

```python
def is_pip_package(filepath: str = __name__) -> bool
```

Determine if the file at the given filepath is part of a pip package.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `filepath` | `str` | The filepath to check. | `__name__` |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if the file is part of a pip package, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L803-L818"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_pip_package(filepath: str = __name__) -> bool:
    """Determine if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_dir_writeable` {#ultralytics.utils.\_\_init\_\_.is\_dir\_writeable}

```python
def is_dir_writeable(dir_path: str | Path) -> bool
```

Check if a directory is writable.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dir_path` | `str | Path` | The path to the directory. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if the directory is writable, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L821-L830"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_dir_writeable(dir_path: str | Path) -> bool:
    """Check if a directory is writable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_pytest_running` {#ultralytics.utils.\_\_init\_\_.is\_pytest\_running}

```python
def is_pytest_running()
```

Determine whether pytest is currently running or not.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if pytest is running, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L833-L839"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_pytest_running():
    """Determine whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.is_github_action_running` {#ultralytics.utils.\_\_init\_\_.is\_github\_action\_running}

```python
def is_github_action_running() -> bool
```

Determine if the current environment is a GitHub Actions runner.

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if the current environment is a GitHub Actions runner, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L842-L848"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_github_action_running() -> bool:
    """Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.get_default_args` {#ultralytics.utils.\_\_init\_\_.get\_default\_args}

```python
def get_default_args(func)
```

Return a dictionary of default arguments for a function.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` | `callable` | The function to inspect. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | A dictionary where each key is a parameter name, and each value is the default value of that parameter. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L851-L861"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_default_args(func):
    """Return a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.get_ubuntu_version` {#ultralytics.utils.\_\_init\_\_.get\_ubuntu\_version}

```python
def get_ubuntu_version()
```

Retrieve the Ubuntu version if the OS is Ubuntu.

**Returns**

| Type | Description |
| --- | --- |
| `str` | Ubuntu version or None if not an Ubuntu OS. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L864-L875"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_ubuntu_version():
    """Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.get_user_config_dir` {#ultralytics.utils.\_\_init\_\_.get\_user\_config\_dir}

```python
def get_user_config_dir(sub_dir = "Ultralytics")
```

Return a writable config dir, preferring YOLO_CONFIG_DIR and being OS-aware.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `sub_dir` | `str` | The name of the subdirectory to create. | `"Ultralytics"` |

**Returns**

| Type | Description |
| --- | --- |
| `Path` | The path to the user config directory. |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L878-L918"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_user_config_dir(sub_dir="Ultralytics"):
    """Return a writable config dir, preferring YOLO_CONFIG_DIR and being OS-aware.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    if p.exists():  # already created → trust it
        return p
    if is_dir_writeable(p.parent):  # create if possible
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Fallbacks for Docker, GCP/AWS functions where only /tmp is writable
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt

    # Last fallback → CWD
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.colorstr` {#ultralytics.utils.\_\_init\_\_.colorstr}

```python
def colorstr(*input)
```

Color a string based on the provided color and style arguments using ANSI escape codes.

This function can be called in two ways:
    - colorstr('color', 'style', 'your string')
    - colorstr('your string')

In the second form, 'blue' and 'bold' will be applied by default.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*input` | `str | Path` | A sequence of strings where the first n-1 strings are color and style arguments, and the<br>    last string is the one to be colored. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | The input string wrapped with ANSI escape codes for the specified color and style. |

**Examples**

```python
>>> colorstr("blue", "bold", "hello world")
"\033[34m\033[1mhello world\033[0m"
```

!!! note "Notes"

    Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L938-L990"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def colorstr(*input):
    r"""Color a string based on the provided color and style arguments using ANSI escape codes.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments, and the
            last string is the one to be colored.

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        "\033[34m\033[1mhello world\033[0m"

    Notes:
        Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.remove_colorstr` {#ultralytics.utils.\_\_init\_\_.remove\_colorstr}

```python
def remove_colorstr(input_string)
```

Remove ANSI escape codes from a string, effectively un-coloring it.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input_string` | `str` | The string to remove color and style from. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | A new string with all ANSI escape codes removed. |

**Examples**

```python
>>> remove_colorstr(colorstr("blue", "bold", "hello world"))
"hello world"
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L993-L1007"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def remove_colorstr(input_string):
    """Remove ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        "hello world"
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.threaded` {#ultralytics.utils.\_\_init\_\_.threaded}

```python
def threaded(func)
```

Multi-thread a target function by default and return the thread or function result.

This decorator provides flexible execution of the target function, either in a separate thread or synchronously. By default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument which is removed from kwargs before calling the function.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` | `callable` | The function to be potentially executed in a separate thread. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `callable` | A wrapper function that either returns a daemon thread or the direct function result. |

**Examples**

```python
>>> @threaded
... def process_data(data):
...     return data
>>>
>>> thread = process_data(my_data)  # Runs in background thread
>>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
```

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1093-L1124"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def threaded(func):
    """Multi-thread a target function by default and return the thread or function result.

    This decorator provides flexible execution of the target function, either in a separate thread or synchronously. By
    default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument which
    is removed from kwargs before calling the function.

    Args:
        func (callable): The function to be potentially executed in a separate thread.

    Returns:
        (callable): A wrapper function that either returns a daemon thread or the direct function result.

    Examples:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # Runs in background thread
        >>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
    """

    def wrapper(*args, **kwargs):
        """Multi-thread a given function based on 'threaded' kwarg and return the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.set_sentry` {#ultralytics.utils.\_\_init\_\_.set\_sentry}

```python
def set_sentry()
```

Initialize the Sentry SDK for error tracking and reporting.

Only used if sentry_sdk package is installed and sync=True in settings. Run 'yolo settings' to see and update
settings.

Conditions required to send errors (ALL conditions must be met or no errors will be reported):
    - sentry_sdk package is installed
    - sync=True in YOLO settings
    - pytest is not running
    - running in a pip package installation
    - running in a non-git directory
    - running with rank -1 or 0
    - online environment
    - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1127-L1192"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_sentry():
    """Initialize the Sentry SDK for error tracking and reporting.

    Only used if sentry_sdk package is installed and sync=True in settings. Run 'yolo settings' to see and update
    settings.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)
    """
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or not ONLINE
        or not IS_PIP_PACKAGE
        or GIT.is_repo
    ):
        return
    # If sentry_sdk package is not installed then return and do not use Sentry
    try:
        import sentry_sdk
    except ImportError:
        return

    def before_send(event, hint):
        """Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            (dict | None): The modified event or None if the event should not be sent to Sentry.
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # do not send event

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 anonymized UUID hash
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.deprecation_warn` {#ultralytics.utils.\_\_init\_\_.deprecation\_warn}

```python
def deprecation_warn(arg, new_arg = None)
```

Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `arg` |  |  | *required* |
| `new_arg` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1408-L1413"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def deprecation_warn(arg, new_arg=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    msg = f"'{arg}' is deprecated and will be removed in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.clean_url` {#ultralytics.utils.\_\_init\_\_.clean\_url}

```python
def clean_url(url)
```

Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `url` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1416-L1419"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return unquote(url).split("?", 1)[0]  # '%2F' to '/', split https://url.com/file.txt?auth
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.url2file` {#ultralytics.utils.\_\_init\_\_.url2file}

```python
def url2file(url)
```

Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `url` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1422-L1424"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.vscode_msg` {#ultralytics.utils.\_\_init\_\_.vscode\_msg}

```python
def vscode_msg(ext = "ultralytics.ultralytics-snippets") -> str
```

Display a message to install Ultralytics-Snippets for VS Code if not already installed.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ext` |  |  | `"ultralytics.ultralytics-snippets"` |

<details>
<summary>Source code in <code>ultralytics/utils/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py#L1427-L1433"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """Display a message to install Ultralytics-Snippets for VS Code if not already installed."""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # file tracks uninstalled extensions, while source directory remains
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"
```
</details>

<br><br>
