---
description: Explore the utility functions and context managers in Ultralytics like WorkingDirectory, increment_path, file_size, and more. Enhance your file handling in Python.
keywords: Ultralytics, file utilities, Python, WorkingDirectory, increment_path, file_size, file_age, contexts, file handling, file management
---

# Reference for `ultralytics/utils/files.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`WorkingDirectory`](#ultralytics.utils.files.WorkingDirectory)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`WorkingDirectory.__enter__`](#ultralytics.utils.files.WorkingDirectory.__enter__)
        - [`WorkingDirectory.__exit__`](#ultralytics.utils.files.WorkingDirectory.__exit__)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`spaces_in_path`](#ultralytics.utils.files.spaces_in_path)
        - [`increment_path`](#ultralytics.utils.files.increment_path)
        - [`file_age`](#ultralytics.utils.files.file_age)
        - [`file_date`](#ultralytics.utils.files.file_date)
        - [`file_size`](#ultralytics.utils.files.file_size)
        - [`get_latest_run`](#ultralytics.utils.files.get_latest_run)
        - [`update_models`](#ultralytics.utils.files.update_models)


## Class `ultralytics.utils.files.WorkingDirectory` {#ultralytics.utils.files.WorkingDirectory}

```python
WorkingDirectory(self, new_dir: str | Path)
```

**Bases:** `contextlib.ContextDecorator`

A context manager and decorator for temporarily changing the working directory.

This class allows for the temporary change of the working directory using a context manager or decorator. It ensures that the original working directory is restored after the context or decorated function completes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `new_dir` | `str | Path` |  | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dir` | `Path | str` | The new directory to switch to. |
| `cwd` | `Path` | The original current working directory before the switch. |

**Methods**

| Name | Description |
| --- | --- |
| [`__enter__`](#ultralytics.utils.files.WorkingDirectory.__enter__) | Change the current working directory to the specified directory upon entering the context. |
| [`__exit__`](#ultralytics.utils.files.WorkingDirectory.__exit__) | Restore the original working directory when exiting the context. |

**Examples**

```python
Using as a context manager:
>>> with WorkingDirectory("/path/to/new/dir"):
...     # Perform operations in the new directory
...     pass

Using as a decorator:
>>> @WorkingDirectory("/path/to/new/dir")
... def some_function():
...     # Perform operations in the new directory
...     pass
```

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L15-L53"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class WorkingDirectory(contextlib.ContextDecorator):
    """A context manager and decorator for temporarily changing the working directory.

    This class allows for the temporary change of the working directory using a context manager or decorator. It ensures
    that the original working directory is restored after the context or decorated function completes.

    Attributes:
        dir (Path | str): The new directory to switch to.
        cwd (Path): The original current working directory before the switch.

    Methods:
        __enter__: Changes the current directory to the specified directory.
        __exit__: Restores the original working directory on context exit.

    Examples:
        Using as a context manager:
        >>> with WorkingDirectory("/path/to/new/dir"):
        ...     # Perform operations in the new directory
        ...     pass

        Using as a decorator:
        >>> @WorkingDirectory("/path/to/new/dir")
        ... def some_function():
        ...     # Perform operations in the new directory
        ...     pass
    """

    def __init__(self, new_dir: str | Path):
        """Initialize the WorkingDirectory context manager with the target directory."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir
```
</details>

<br>

### Method `ultralytics.utils.files.WorkingDirectory.__enter__` {#ultralytics.utils.files.WorkingDirectory.\_\_enter\_\_}

```python
def __enter__(self)
```

Change the current working directory to the specified directory upon entering the context.

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L47-L49"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __enter__(self):
    """Change the current working directory to the specified directory upon entering the context."""
    os.chdir(self.dir)
```
</details>

<br>

### Method `ultralytics.utils.files.WorkingDirectory.__exit__` {#ultralytics.utils.files.WorkingDirectory.\_\_exit\_\_}

```python
def __exit__(self, exc_type, exc_val, exc_tb)
```

Restore the original working directory when exiting the context.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `exc_type` |  |  | *required* |
| `exc_val` |  |  | *required* |
| `exc_tb` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L51-L53"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    """Restore the original working directory when exiting the context."""
    os.chdir(self.cwd)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.spaces_in_path` {#ultralytics.utils.files.spaces\_in\_path}

```python
def spaces_in_path(path: str | Path)
```

Context manager to handle paths with spaces in their names.

If a path contains spaces, it replaces them with underscores, copies the file/directory to the new path, executes the context code block, then copies the file/directory back to its original location.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` | The original path that may contain spaces. | *required* |

**Examples**

```python
>>> with spaces_in_path("/path/with spaces") as new_path:
...     # Your code here
...     pass
```

**Yields**

| Type | Description |
| --- | --- |
| `Path | str` | Temporary path with any spaces replaced by underscores. |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L57-L103"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@contextmanager
def spaces_in_path(path: str | Path):
    """Context manager to handle paths with spaces in their names.

    If a path contains spaces, it replaces them with underscores, copies the file/directory to the new path, executes
    the context code block, then copies the file/directory back to its original location.

    Args:
        path (str | Path): The original path that may contain spaces.

    Yields:
        (Path | str): Temporary path with any spaces replaced by underscores.

    Examples:
        >>> with spaces_in_path("/path/with spaces") as new_path:
        ...     # Your code here
        ...     pass
    """
    # If path has spaces, replace them with underscores
    if " " in str(path):
        string = isinstance(path, str)  # input type
        path = Path(path)

        # Create a temporary directory and construct the new path
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")

            # Copy file/directory
            if path.is_dir():
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)

            try:
                # Yield the temporary path
                yield str(tmp_path) if string else tmp_path

            finally:
                # Copy file/directory back
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)  # Copy back the file

    else:
        # If there are no spaces, just yield the original path
        yield path
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.increment_path` {#ultralytics.utils.files.increment\_path}

```python
def increment_path(path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path
```

Increment a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the number will be appended directly to the end of the path.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` | Path to increment. | *required* |
| `exist_ok` | `bool, optional` | If True, the path will not be incremented and returned as-is. | `False` |
| `sep` | `str, optional` | Separator to use between the path and the incrementation number. | `""` |
| `mkdir` | `bool, optional` | Create a directory if it does not exist. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `Path` | Incremented path. |

**Examples**

```python
Increment a directory path:
>>> from pathlib import Path
>>> path = Path("runs/exp")
>>> new_path = increment_path(path)
>>> print(new_path)
runs/exp2

Increment a file path:
>>> path = Path("runs/exp/results.txt")
>>> new_path = increment_path(path)
>>> print(new_path)
runs/exp/results2.txt
```

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L106-L150"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def increment_path(path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path:
    """Increment a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to the
    end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the number
    will be appended directly to the end of the path.

    Args:
        path (str | Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is.
        sep (str, optional): Separator to use between the path and the incrementation number.
        mkdir (bool, optional): Create a directory if it does not exist.

    Returns:
        (Path): Incremented path.

    Examples:
        Increment a directory path:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        Increment a file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.file_age` {#ultralytics.utils.files.file\_age}

```python
def file_age(path: str | Path = __file__) -> int
```

Return days since the last modification of the specified file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` |  | `__file__` |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L153-L156"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def file_age(path: str | Path = __file__) -> int:
    """Return days since the last modification of the specified file."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.file_date` {#ultralytics.utils.files.file\_date}

```python
def file_date(path: str | Path = __file__) -> str
```

Return the file modification date in 'YYYY-M-D' format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` |  | `__file__` |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L159-L162"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def file_date(path: str | Path = __file__) -> str:
    """Return the file modification date in 'YYYY-M-D' format."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.file_size` {#ultralytics.utils.files.file\_size}

```python
def file_size(path: str | Path) -> float
```

Return the size of a file or directory in mebibytes (MiB).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str | Path` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L165-L174"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def file_size(path: str | Path) -> float:
    """Return the size of a file or directory in mebibytes (MiB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.get_latest_run` {#ultralytics.utils.files.get\_latest\_run}

```python
def get_latest_run(search_dir: str = ".") -> str
```

Return the path to the most recent 'last.pt' file in the specified directory for resuming training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `search_dir` | `str` |  | `"."` |

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L177-L180"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_latest_run(search_dir: str = ".") -> str:
    """Return the path to the most recent 'last.pt' file in the specified directory for resuming training."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.files.update_models` {#ultralytics.utils.files.update\_models}

```python
def update_models(model_names: tuple = ("yolo26n.pt",), source_dir: Path = Path("."), update_names: bool = False)
```

Update and re-save specified YOLO models in an 'updated_models' subdirectory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_names` | `tuple, optional` | Model filenames to update. | `("yolo26n.pt",)` |
| `source_dir` | `Path, optional` | Directory containing models and target subdirectory. | `Path(".")` |
| `update_names` | `bool, optional` | Update model names from a data YAML. | `False` |

**Examples**

```python
Update specified YOLO models and save them in 'updated_models' subdirectory:
>>> from ultralytics.utils.files import update_models
>>> model_names = ("yolo26n.pt", "yolo11s.pt")
>>> update_models(model_names, source_dir=Path("/models"), update_names=True)
```

<details>
<summary>Source code in <code>ultralytics/utils/files.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py#L183-L219"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update_models(model_names: tuple = ("yolo26n.pt",), source_dir: Path = Path("."), update_names: bool = False):
    """Update and re-save specified YOLO models in an 'updated_models' subdirectory.

    Args:
        model_names (tuple, optional): Model filenames to update.
        source_dir (Path, optional): Directory containing models and target subdirectory.
        update_names (bool, optional): Update model names from a data YAML.

    Examples:
        Update specified YOLO models and save them in 'updated_models' subdirectory:
        >>> from ultralytics.utils.files import update_models
        >>> model_names = ("yolo26n.pt", "yolo11s.pt")
        >>> update_models(model_names, source_dir=Path("/models"), update_names=True)
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names
    from ultralytics.utils import LOGGER

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists

    for model_name in model_names:
        model_path = source_dir / model_name
        LOGGER.info(f"Loading model from {model_path}")

        # Load model
        model = YOLO(model_path)
        model.half()
        if update_names:  # update model names from a dataset YAML
            model.model.names = default_class_names("coco8.yaml")

        # Define new save path
        save_path = target_dir / model_name

        # Save model using model.save()
        LOGGER.info(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path)
```
</details>

<br><br>
