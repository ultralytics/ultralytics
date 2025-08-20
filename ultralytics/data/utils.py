# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    MACOS,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    YAML,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", not MACOS)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def img2label_paths(img_paths: List[str]) -> List[str]:
    """Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def check_file_speeds(
    files: List[str], threshold_ms: float = 10, threshold_mb: float = 50, max_files: int = 5, prefix: str = ""
):
    """
    Check dataset file access speed and provide performance feedback.

    This function tests the access speed of dataset files by measuring ping (stat call) time and read speed.
    It samples up to 5 files from the provided list and warns if access times exceed the threshold.

    Args:
        files (List[str]): List of file paths to check for access speed.
        threshold_ms (float, optional): Threshold in milliseconds for ping time warnings.
        threshold_mb (float, optional): Threshold in megabytes per second for read speed warnings.
        max_files (int, optional): The maximum number of files to check.
        prefix (str, optional): Prefix string to add to log messages.

    Examples:
        >>> from pathlib import Path
        >>> image_files = list(Path("dataset/images").glob("*.jpg"))
        >>> check_file_speeds(image_files, threshold_ms=15)
    """
    if not files or len(files) == 0:
        LOGGER.warning(f"{prefix}Image speed checks: No files to check")
        return

    # Sample files (max 5)
    files = random.sample(files, min(max_files, len(files)))

    # Test ping (stat time)
    ping_times = []
    file_sizes = []
    read_speeds = []

    for f in files:
        try:
            # Measure ping (stat call)
            start = time.perf_counter()
            file_size = os.stat(f).st_size
            ping_times.append((time.perf_counter() - start) * 1000)  # ms
            file_sizes.append(file_size)

            # Measure read speed
            start = time.perf_counter()
            with open(f, "rb") as file_obj:
                _ = file_obj.read()
            read_time = time.perf_counter() - start
            if read_time > 0:  # Avoid division by zero
                read_speeds.append(file_size / (1 << 20) / read_time)  # MB/s
        except Exception:
            pass

    if not ping_times:
        LOGGER.warning(f"{prefix}Image speed checks: failed to access files")
        return

    # Calculate stats with uncertainties
    avg_ping = np.mean(ping_times)
    std_ping = np.std(ping_times, ddof=1) if len(ping_times) > 1 else 0
    size_msg = f", size: {np.mean(file_sizes) / (1 << 10):.1f} KB"
    ping_msg = f"ping: {avg_ping:.1f}±{std_ping:.1f} ms"

    if read_speeds:
        avg_speed = np.mean(read_speeds)
        std_speed = np.std(read_speeds, ddof=1) if len(read_speeds) > 1 else 0
        speed_msg = f", read: {avg_speed:.1f}±{std_speed:.1f} MB/s"
    else:
        speed_msg = ""

    if avg_ping < threshold_ms or avg_speed < threshold_mb:
        LOGGER.info(f"{prefix}Fast image access ✅ ({ping_msg}{speed_msg}{size_msg})")
    else:
        LOGGER.warning(
            f"{prefix}Slow image access detected ({ping_msg}{speed_msg}{size_msg}). "
            f"Use local storage instead of remote/mounted storage for better performance. "
            f"See https://docs.ultralytics.com/guides/model-training-tips/"
        )


def get_hash(paths: List[str]) -> str:
    """Return a single hash value of a list of paths (files or dirs)."""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image) -> Tuple[int, int]:
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def verify_image(args: Tuple) -> Tuple:
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args: Tuple) -> List:
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

                # All labels
                max_cls = 0 if single_cls else lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def visualize_image_annotations(image_path: str, txt_path: str, label_map: Dict[int, str]):
    """
    Visualize YOLO annotations (bounding boxes and class labels) on an image.

    This function reads an image and its corresponding annotation file in YOLO format, then
    draws bounding boxes around detected objects and labels them with their respective class names.
    The bounding box colors are assigned based on the class ID, and the text color is dynamically
    adjusted for readability, depending on the background color's luminance.

    Args:
        image_path (str): The path to the image file to annotate, and it can be in formats supported by PIL.
        txt_path (str): The path to the annotation file in YOLO format, that should contain one line per object.
        label_map (Dict[int, str]): A dictionary that maps class IDs (integers) to class labels (strings).

    Examples:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # It should include all annotated classes details
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    _, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # Get and normalize the RGB color
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()


def polygon2mask(
    imgsz: Tuple[int, int], polygons: List[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (Tuple[int, int]): The size of the image as (height, width).
        polygons (List[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(
    imgsz: Tuple[int, int], polygons: List[np.ndarray], color: int, downsample_ratio: int = 1
) -> np.ndarray:
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (Tuple[int, int]): The size of the image as (height, width).
        polygons (List[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(
    imgsz: Tuple[int, int], segments: List[np.ndarray], downsample_ratio: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset: str, autodownload: bool = True) -> Dict[str, Any]:
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found.

    Returns:
        (Dict[str, Any]): Parsed dataset information and paths.
    """
    file = check_file(dataset)

    # Download (optional)
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # Read YAML
    data = YAML.load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.warning("renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])
    data["channels"] = data.get("channels", 3)  # get image channels, default to 3

    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.exists() and not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()  # path relative to DATASETS_DIR

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            LOGGER.info("")
            m = f"Dataset '{name}' images not found, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: Union[str, Path], split: str = "") -> Dict[str, Any]:
    """
    Check a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''.

    Returns:
        (Dict[str, Any]): A dictionary containing the following keys:

            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (Dict[int, str]): A dictionary of class names in the dataset.
    """
    # Download (optional if dataset=https://file.zip is passed directly)
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif str(dataset).endswith((".zip", ".tar", ".gz")):
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        LOGGER.info("")
        LOGGER.warning(f"Dataset not found, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        LOGGER.info(f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n")
    train_set = data_dir / "train"
    if not train_set.is_dir():
        LOGGER.warning(f"Dataset 'split=train' not found at {train_set}")
        image_files = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
        if image_files:
            from ultralytics.data.split import split_classify_dataset

            LOGGER.info(f"Found {len(image_files)} images in subdirectories. Attempting to split...")
            data_dir = split_classify_dataset(data_dir, train_ratio=0.8)
            train_set = data_dir / "train"
        else:
            LOGGER.error(f"No images found in {data_dir} or its subdirectories.")
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else data_dir / "valid"
        if (data_dir / "valid").exists()
        else None
    )  # data/test or data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val or data/test
    if split == "val" and not val_set:
        LOGGER.warning("Dataset 'split=val' not found, using 'split=test' instead.")
        val_set = test_set
    elif split == "test" and not test_set:
        LOGGER.warning("Dataset 'split=test' not found, using 'split=val' instead.")
        test_set = val_set

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))

    # Print to console
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)  # number of files
            nd = len({file.parent for file in files})  # number of directories
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(f"{dataset} '{k}:' no training images found")
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes (no images found)")
            elif nd != nc:
                LOGGER.error(f"{prefix} found {nf} images in {nd} classes (requires {nc} classes, not {nd})")
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names, "channels": 3}


class HUBDatasetStats:
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'.
        autodownload (bool): Attempt to download dataset if not found locally.

    Attributes:
        task (str): Dataset task type.
        hub_dir (Path): Directory path for HUB dataset files.
        im_dir (Path): Directory path for compressed images.
        stats (Dict): Statistics dictionary containing dataset information.
        data (Dict): Dataset configuration data.

    Methods:
        get_json: Return dataset JSON for Ultralytics HUB.
        process_images: Compress images for Ultralytics HUB.

    Note:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
        i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.

    Examples:
        >>> from ultralytics.data.utils import HUBDatasetStats
        >>> stats = HUBDatasetStats("path/to/coco8.zip", task="detect")  # detect dataset
        >>> stats = HUBDatasetStats("path/to/coco8-seg.zip", task="segment")  # segment dataset
        >>> stats = HUBDatasetStats("path/to/coco8-pose.zip", task="pose")  # pose dataset
        >>> stats = HUBDatasetStats("path/to/dota8.zip", task="obb")  # OBB dataset
        >>> stats = HUBDatasetStats("path/to/imagenet10.zip", task="classify")  # classification dataset
        >>> stats.get_json(save=True)
        >>> stats.process_images()
    """

    def __init__(self, path: str = "coco8.yaml", task: str = "detect", autodownload: bool = False):
        """Initialize class."""
        path = Path(path).resolve()
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        self.task = task  # detect, segment, pose, classify, obb
        if self.task == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose, obb
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # Load YAML with checks
                data = YAML.load(yaml_path)
                data["path"] = ""  # strip path since YAML should be in dataset root for all HUB datasets
                YAML.save(yaml_path, data)
                data = check_det_dataset(yaml_path, autodownload)  # dict
                data["path"] = data_dir  # YAML path should be set to '' (relative) or parent (absolute)
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f"{data['path']}-hub")
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _unzip(path: Path) -> Tuple[bool, str, Path]:
        """Unzip data.zip."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. path/to/abc.zip MUST unzip to path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f: str):
        """Save a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save: bool = False, verbose: bool = False) -> Dict:
        """Return dataset JSON for Ultralytics HUB."""

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == "detect":
                coordinates = labels["bboxes"]
            elif self.task in {"segment", "obb"}:  # Segment and OBB use segments. OBB segments are normalized xyxyxyxy
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task == "pose":
                n, nk, nd = labels["keypoints"].shape
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
            else:
                raise ValueError(f"Undefined dataset task={self.task}.")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None  # predefine
            path = self.data.get(split)

            # Check split
            if path is None:  # no split
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # image files in split
            if not files:  # no images
                continue

            # Get dataset statistics
            if self.task == "classify":
                from torchvision.datasets import ImageFolder  # scope for faster 'import ultralytics'

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )  # shape(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }

        # Save, print and return
        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self) -> Path:
        """Compress images for Ultralytics HUB."""
        from ultralytics.data import YOLODataset  # ClassificationDataset

        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/images/
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


def compress_one_image(f: str, f_new: str = None, max_dim: int = 1920, quality: int = 50):
    """
    Compress a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image.
        quality (int, optional): The image compression quality as a percentage.

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.data.utils import compress_one_image
        >>> for f in Path("path/to/dataset").rglob("*.jpg"):
        >>>    compress_one_image(f)
    """
    try:  # use PIL
        Image.MAX_IMAGE_PIXELS = None  # Fix DecompressionBombError, allow optimization of image > ~178.9 million pixels
        im = Image.open(f)
        if im.mode in {"RGBA", "LA"}:  # Convert to RGB if needed (for JPEG)
            im = im.convert("RGB")
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.warning(f"HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def load_dataset_cache_file(path: Path) -> Dict:
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix: str, path: Path, x: Dict, version: str):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}Cache directory {path.parent} is not writeable, cache not saved.")
