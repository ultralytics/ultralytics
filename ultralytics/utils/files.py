# Ultralytics YOLO 🚀, AGPL-3.0 license

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
        dir (Path): The new directory to switch to.
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

    def __init__(self, new_dir):
        """Sets the working directory to 'new_dir' upon instantiation for use with context managers or decorators."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """Changes the current working directory to the specified directory upon entering the context."""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """Restores the original working directory when exiting the context."""
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path):
    """
    Context manager to handle paths with spaces in their names. If a path contains spaces, it replaces them with
    underscores, copies the file/directory to the new path, executes the context code block, then copies the
    file/directory back to its original location.

    Args:
        path (str | Path): The original path that may contain spaces.

    Yields:
        (Path): Temporary path with spaces replaced by underscores if spaces were present, otherwise the original path.

    Examples:
        Use the context manager to handle paths with spaces:
        >>> from ultralytics.utils.files import spaces_in_path
        >>> with spaces_in_path('/path/with spaces') as new_path:
        >>> # Your code here
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
                # tmp_path.mkdir(parents=True, exist_ok=True)
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


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str | pathlib.Path): Path to increment.
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        sep (str): Separator to use between the path and the incrementation number.
        mkdir (bool): Create a directory if it does not exist.

    Returns:
        (pathlib.Path): Incremented path.

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


def file_age(path=__file__):
    """Return days since the last modification of the specified file."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """Returns the file modification date in 'YYYY-M-D' format."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """Returns the size of a file or directory in megabytes (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir="."):
    """Returns the path to the most recent 'last.pt' file in the specified directory for resuming training."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(model_names=("yolo11n.pt",), source_dir=Path("."), update_names=False):
    """
    Updates and re-saves specified YOLO models in an 'updated_models' subdirectory.

    Args:
        model_names (Tuple[str, ...]): Model filenames to update.
        source_dir (Path): Directory containing models and target subdirectory.
        update_names (bool): Update model names from a data YAML.

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
