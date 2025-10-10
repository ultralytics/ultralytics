# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    """
    A context manager and decorator for temporarily changing the working directory.

    This class allows for the temporary change of the working directory using a context manager or decorator.
    It ensures that the original working directory is restored after the context or decorated function completes.

    Attributes:
        dir (Path | str): The new directory to switch to.
        cwd (Path): The original current working directory before the switch.

    Methods:
        __enter__: Changes the current directory to the specified directory.
        __exit__: Restores the original working directory on context exit.

    Examples:
        Using as a context manager:
        >>> with WorkingDirectory('/path/to/new/dir'):
        >>> # Perform operations in the new directory
        >>>     pass

        Using as a decorator:
        >>> @WorkingDirectory('/path/to/new/dir')
        >>> def some_function():
        >>> # Perform operations in the new directory
        >>>     pass
    """

    def __init__(self, new_dir: str | Path):
        """Initialize the WorkingDirectory context manager with the target directory."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """Change the current working directory to the specified directory upon entering the context."""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """Restore the original working directory when exiting the context."""
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path: str | Path):
    """
    Context manager to handle paths with spaces in their names.

    If a path contains spaces, it replaces them with underscores, copies the file/directory to the new path, executes
    the context code block, then copies the file/directory back to its original location.

    Args:
        path (str | Path): The original path that may contain spaces.

    Yields:
        (Path | str): Temporary path with spaces replaced by underscores if spaces were present, otherwise the
            original path.

    Examples:
        >>> with spaces_in_path('/path/with spaces') as new_path:
        >>> # Your code here
        >>>     pass
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


def increment_path(path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path:
    """
    Increment a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path.

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


def file_age(path: str | Path = __file__) -> int:
    """Return days since the last modification of the specified file."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path: str | Path = __file__) -> str:
    """Return the file modification date in 'YYYY-M-D' format."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path: str | Path) -> float:
    """Return the size of a file or directory in megabytes (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir: str = ".") -> str:
    """Return the path to the most recent 'last.pt' file in the specified directory for resuming training."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(model_names: tuple = ("yolo11n.pt",), source_dir: Path = Path("."), update_names: bool = False):
    """
    Update and re-save specified YOLO models in an 'updated_models' subdirectory.

    Args:
        model_names (tuple, optional): Model filenames to update.
        source_dir (Path, optional): Directory containing models and target subdirectory.
        update_names (bool, optional): Update model names from a data YAML.

    Examples:
        Update specified YOLO models and save them in 'updated_models' subdirectory:
        >>> from ultralytics.utils.files import update_models
        >>> model_names = ("yolo11n.pt", "yolov8s.pt")
        >>> update_models(model_names, source_dir=Path("/models"), update_names=True)
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists

    for model_name in model_names:
        model_path = source_dir / model_name
        print(f"Loading model from {model_path}")

        # Load model
        model = YOLO(model_path)
        model.half()
        if update_names:  # update model names from a dataset YAML
            model.model.names = default_class_names("coco8.yaml")

        # Define new save path
        save_path = target_dir / model_name

        # Save model using model.save()
        print(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path)
