# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM, YAML
from ultralytics.utils.checks import check_file, check_requirements
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class() -> list[int]:
    """
    Convert 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list[int]): A list of 91 class IDs where the index represents the 80-index class ID and the value
            is the corresponding 91-index class ID.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class() -> list[int]:
    r"""
    Convert 80-index (val2014) to 91-index (paper).

    Returns:
        (list[int]): A list of 80 class IDs where each value is the corresponding 91-index class ID.

    References:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

    Examples:
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
    lvis: bool = False,
):
    """
    Convert COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    Examples:
        >>> from ultralytics.data.converter import convert_coco

        Convert COCO annotations to YOLO format
        >>> convert_coco("coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)

        Convert LVIS annotations to YOLO format
        >>> convert_coco("lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
    """
    # Create dataset directory
    save_dir = increment_path(save_dir)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            # NOTE: create folders for both train and val in advance,
            # since LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Create image dict
        images = {f"{x['id']:d}": x for x in data["images"]}
        # Create image-annotations dict
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        image_txt = []
        # Write labels file
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            # Write
            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


def convert_segment_masks_to_yolo_seg(masks_dir: str, output_dir: str, classes: int):
    """
    Convert a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation
    format. The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total classes in the dataset i.e. for COCO classes=80

    Examples:
        >>> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        >>> convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")


def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Convert DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Examples:
        >>> from ultralytics.data.converter import convert_dota_to_yolo_obb
        >>> convert_dota_to_yolo_obb("path/to/DOTA")

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name: str, image_width: int, image_height: int, orig_label_dir: Path, save_dir: Path):
        """Convert a single image's DOTA annotation to YOLO OBB format and save it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in {"train", "val"}:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        idx1 (int): Index of the point in arr1 with the shortest distance.
        idx2 (int): Index of the point in arr2 with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_bbox2segment(im_dir: str | Path, save_dir: str | Path | None = None, sam_model: str = "sam_b.pt", device=None):
    """
    Convert existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB) in
    YOLO format. Generate segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path, optional): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None.
        sam_model (str): Segmentation model to use for intermediate segmentation data.
        device (int | str, optional): The specific device to run SAM models.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy

    # NOTE: add placeholder to pass class index check
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000)), channels=3))
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) == 0:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")


def create_synthetic_coco_dataset():
    """
    Create a synthetic COCO dataset with random images based on filenames from label lists.

    This function downloads COCO labels, reads image filenames from label list files,
    creates synthetic images for train2017 and val2017 subsets, and organizes
    them in the COCO dataset structure. It uses multithreading to generate images efficiently.

    Examples:
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    Notes:
        - Requires internet connection to download label files.
        - Generates random RGB images of varying sizes (480x480 to 640x640 pixels).
        - Existing test2017 directory is removed as it's not needed.
        - Reads image filenames from train2017.txt and val2017.txt files.
    """

    def create_synthetic_image(image_file: Path):
        """Generate synthetic images with random sizes and colors for dataset augmentation or testing purposes."""
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)

    # Download labels
    dir = DATASETS_DIR / "coco"
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    label_zip = "coco2017labels-segments.zip"
    download([url + label_zip], dir=dir.parent)

    # Create synthetic images
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)  # Remove test2017 directory as not needed
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in {"train2017", "val2017"}:
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            # Read image filenames from label list file
            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file, encoding="utf-8") as f:
                    image_files = [dir / line.strip() for line in f]

                # Submit all tasks
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"Generating images for {subset}"):
                    pass  # The actual work is done in the background
            else:
                LOGGER.warning(f"Labels file {label_list_file} does not exist. Skipping image creation for {subset}.")

    LOGGER.info("Synthetic COCO dataset created successfully.")


def convert_to_multispectral(path: str | Path, n_channels: int = 10, replace: bool = False, zip: bool = False):
    """
    Convert RGB images to multispectral images by interpolating across wavelength bands.

    This function takes RGB images and interpolates them to create multispectral images with a specified number
    of channels. It can process either a single image or a directory of images.

    Args:
        path (str | Path): Path to an image file or directory containing images to convert.
        n_channels (int): Number of spectral channels to generate in the output image.
        replace (bool): Whether to replace the original image file with the converted one.
        zip (bool): Whether to zip the converted images into a zip file.

    Examples:
        Convert a single image
        >>> convert_to_multispectral("path/to/image.jpg", n_channels=10)

        Convert a dataset
        >>> convert_to_multispectral("coco8", n_channels=10)
    """
    from scipy.interpolate import interp1d

    from ultralytics.data.utils import IMG_FORMATS

    path = Path(path)
    if path.is_dir():
        # Process directory
        im_files = sum((list(path.rglob(f"*.{ext}")) for ext in (IMG_FORMATS - {"tif", "tiff"})), [])
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f"Error converting {im_path}: {e}")

        if zip:
            zip_directory(path)
    else:
        # Process a single image
        output_path = path.with_suffix(".tiff")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        # Interpolate all pixels at once
        rgb_wavelengths = np.array([650, 510, 475])  # R, G, B wavelengths (nm)
        target_wavelengths = np.linspace(450, 700, n_channels)
        f = interp1d(rgb_wavelengths.T, img, kind="linear", bounds_error=False, fill_value="extrapolate")
        multispectral = f(target_wavelengths)
        cv2.imwritemulti(str(output_path), np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1))
        LOGGER.info(f"Converted {output_path}")


async def convert_ndjson_to_yolo(ndjson_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Convert NDJSON dataset format to Ultralytics YOLO11 dataset structure.

    This function converts datasets stored in NDJSON (Newline Delimited JSON) format to the standard YOLO
    format with separate directories for images and labels. It supports parallel processing for efficient
    conversion of large datasets and can download images from URLs if they don't exist locally.

    The NDJSON format consists of:
    - First line: Dataset metadata with class names and configuration
    - Subsequent lines: Individual image records with annotations and optional URLs

    Args:
        ndjson_path (Union[str, Path]): Path to the input NDJSON file containing dataset information.
        output_path (Optional[Union[str, Path]], optional): Directory where the converted YOLO dataset
            will be saved. If None, uses the parent directory of the NDJSON file. Defaults to None.

    Returns:
        (Path): Path to the generated data.yaml file that can be used for YOLO training.

    Examples:
        Convert a local NDJSON file:
        >>> yaml_path = convert_ndjson_to_yolo("dataset.ndjson")
        >>> print(f"Dataset converted to: {yaml_path}")

        Convert with custom output directory:
        >>> yaml_path = convert_ndjson_to_yolo("dataset.ndjson", output_path="./converted_datasets")

        Use with YOLO training
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.train(data="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-ndjson.ndjson")
    """
    check_requirements("aiohttp")
    import aiohttp

    ndjson_path = Path(check_file(ndjson_path))
    output_path = Path(output_path or DATASETS_DIR)
    with open(ndjson_path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    dataset_record, image_records = lines[0], lines[1:]
    dataset_dir = output_path / ndjson_path.stem
    splits = {record["split"] for record in image_records}

    # Create directories and prepare YAML structure
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = dict(dataset_record)
    data_yaml["names"] = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    data_yaml.pop("class_names")

    for split in sorted(splits):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        data_yaml[split] = f"images/{split}"

    async def process_record(session, semaphore, record):
        """Process single image record with async session."""
        async with semaphore:
            split, original_name = record["split"], record["file"]
            label_path = dataset_dir / "labels" / split / f"{Path(original_name).stem}.txt"
            image_path = dataset_dir / "images" / split / original_name

            annotations = record.get("annotations", {})
            lines_to_write = []
            for key in annotations.keys():
                lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
                break
            if "classification" in annotations:
                lines_to_write = [str(cls) for cls in annotations["classification"]]

            label_path.write_text("\n".join(lines_to_write) + "\n" if lines_to_write else "")

            if http_url := record.get("url"):
                if not image_path.exists():
                    try:
                        async with session.get(http_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            response.raise_for_status()
                            with open(image_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                        return True
                    except Exception as e:
                        LOGGER.warning(f"Failed to download {http_url}: {e}")
                        return False
            return True

    # Process all images with async downloads
    semaphore = asyncio.Semaphore(64)
    async with aiohttp.ClientSession() as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} → {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result

        await asyncio.gather(*[tracked_process(record) for record in image_records])
        pbar.close()

    # Write data.yaml
    yaml_path = dataset_dir / "data.yaml"
    YAML.save(yaml_path, data_yaml)

    return yaml_path
