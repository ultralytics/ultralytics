# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class():
    """
    Converts 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
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


def coco80_to_coco91_class():
    """
    Converts 80-index (val2014) to 91-index (paper).
    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.

    Example:
        ```python
        import numpy as np

        a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
        b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        ```
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
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        convert_coco('../datasets/lvis/annotations/', use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
        ```

    Output:
        Generates output files in the specified output directory.
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
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        image_txt = []
        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
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
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            with open((Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")), "a") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes):
    """
    Converts a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total classes in the dataset i.e. for COCO classes=80

    Example:
        ```python
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        # The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        convert_segment_masks_to_yolo_seg('path/to/masks_directory', 'path/to/output/directory', classes=80)
        ```

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
    import os

    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_filename in os.listdir(masks_dir):
        if mask_filename.endswith(".png"):
            mask_path = os.path.join(masks_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_filename}, skipping.")
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
            output_path = os.path.join(output_dir, os.path.splitext(mask_filename)[0] + ".txt")
            with open(output_path, "w") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")


def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb('path/to/DOTA')
        ```

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

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
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
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
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


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
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


def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt"):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in YOLO format. Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from tqdm import tqdm

    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils import LOGGER
    from ultralytics.utils.ops import xywh2xyxy

    # NOTE: add placeholder to pass class index check
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in tqdm(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        if texts:
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")
