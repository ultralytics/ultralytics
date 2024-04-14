# Ultralytics YOLO 🚀, AGPL-3.0 license

import getpass
from typing import List

import cv2
import numpy as np

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER as logger
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import plot_images


def get_table_schema(vector_size):
    """Extracts and returns the schema of a database table."""
    from lancedb.pydantic import LanceModel, Vector

    class Schema(LanceModel):
        im_file: str
        labels: List[str]
        cls: List[int]
        bboxes: List[List[float]]
        masks: List[List[List[int]]]
        keypoints: List[List[List[float]]]
        vector: Vector(vector_size)

    return Schema


def get_sim_index_schema():
    """Returns a LanceModel schema for a database table with specified vector size."""
    from lancedb.pydantic import LanceModel

    class Schema(LanceModel):
        idx: int
        im_file: str
        count: int
        sim_im_files: List[str]

    return Schema


def sanitize_batch(batch, dataset_info):
    """Sanitizes input batch for inference, ensuring correct format and dimensions."""
    batch["cls"] = batch["cls"].flatten().int().tolist()
    box_cls_pair = sorted(zip(batch["bboxes"].tolist(), batch["cls"]), key=lambda x: x[1])
    batch["bboxes"] = [box for box, _ in box_cls_pair]
    batch["cls"] = [cls for _, cls in box_cls_pair]
    batch["labels"] = [dataset_info["names"][i] for i in batch["cls"]]
    batch["masks"] = batch["masks"].tolist() if "masks" in batch else [[[]]]
    batch["keypoints"] = batch["keypoints"].tolist() if "keypoints" in batch else [[[]]]
    return batch


def plot_query_result(similar_set, plot_labels=True):
    """
    Plot images from the similar set.

    Args:
        similar_set (list): Pyarrow or pandas object containing the similar data points
        plot_labels (bool): Whether to plot labels or not
    """
    import pandas  # scope for faster 'import ultralytics'

    similar_set = (
        similar_set.to_dict(orient="list") if isinstance(similar_set, pandas.DataFrame) else similar_set.to_pydict()
    )
    empty_masks = [[[]]]
    empty_boxes = [[]]
    images = similar_set.get("im_file", [])
    bboxes = similar_set.get("bboxes", []) if similar_set.get("bboxes") is not empty_boxes else []
    masks = similar_set.get("masks") if similar_set.get("masks")[0] != empty_masks else []
    kpts = similar_set.get("keypoints") if similar_set.get("keypoints")[0] != empty_masks else []
    cls = similar_set.get("cls", [])

    plot_size = 640
    imgs, batch_idx, plot_boxes, plot_masks, plot_kpts = [], [], [], [], []
    for i, imf in enumerate(images):
        im = cv2.imread(imf)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        r = min(plot_size / h, plot_size / w)
        imgs.append(LetterBox(plot_size, center=False)(image=im).transpose(2, 0, 1))
        if plot_labels:
            if len(bboxes) > i and len(bboxes[i]) > 0:
                box = np.array(bboxes[i], dtype=np.float32)
                box[:, [0, 2]] *= r
                box[:, [1, 3]] *= r
                plot_boxes.append(box)
            if len(masks) > i and len(masks[i]) > 0:
                mask = np.array(masks[i], dtype=np.uint8)[0]
                plot_masks.append(LetterBox(plot_size, center=False)(image=mask))
            if len(kpts) > i and kpts[i] is not None:
                kpt = np.array(kpts[i], dtype=np.float32)
                kpt[:, :, :2] *= r
                plot_kpts.append(kpt)
        batch_idx.append(np.ones(len(np.array(bboxes[i], dtype=np.float32))) * i)
    imgs = np.stack(imgs, axis=0)
    masks = np.stack(plot_masks, axis=0) if plot_masks else np.zeros(0, dtype=np.uint8)
    kpts = np.concatenate(plot_kpts, axis=0) if plot_kpts else np.zeros((0, 51), dtype=np.float32)
    boxes = xyxy2xywh(np.concatenate(plot_boxes, axis=0)) if plot_boxes else np.zeros(0, dtype=np.float32)
    batch_idx = np.concatenate(batch_idx, axis=0)
    cls = np.concatenate([np.array(c, dtype=np.int32) for c in cls], axis=0)

    return plot_images(
        imgs, batch_idx, cls, bboxes=boxes, masks=masks, kpts=kpts, max_subplots=len(images), save=False, threaded=False
    )


def prompt_sql_query(query):
    """Plots images with optional labels from a similar data set."""
    check_requirements("openai>=1.6.1")
    from openai import OpenAI

    if not SETTINGS["openai_api_key"]:
        logger.warning("OpenAI API key not found in settings. Please enter your API key below.")
        openai_api_key = getpass.getpass("OpenAI API key: ")
        SETTINGS.update({"openai_api_key": openai_api_key})
    openai = OpenAI(api_key=SETTINGS["openai_api_key"])

    messages = [
        {
            "role": "system",
            "content": """
                You are a helpful data scientist proficient in SQL. You need to output exactly one SQL query based on
                the following schema and a user request. You only need to output the format with fixed selection
                statement that selects everything from "'table'", like `SELECT * from 'table'`

                Schema:
                im_file: string not null
                labels: list<item: string> not null
                child 0, item: string
                cls: list<item: int64> not null
                child 0, item: int64
                bboxes: list<item: list<item: double>> not null
                child 0, item: list<item: double>
                    child 0, item: double
                masks: list<item: list<item: list<item: int64>>> not null
                child 0, item: list<item: list<item: int64>>
                    child 0, item: list<item: int64>
                        child 0, item: int64
                keypoints: list<item: list<item: list<item: double>>> not null
                child 0, item: list<item: list<item: double>>
                    child 0, item: list<item: double>
                        child 0, item: double
                vector: fixed_size_list<item: float>[256] not null
                child 0, item: float

                Some details about the schema:
                - the "labels" column contains the string values like 'person' and 'dog' for the respective objects
                    in each image
                - the "cls" column contains the integer values on these classes that map them the labels

                Example of a correct query:
                request - Get all data points that contain 2 or more people and at least one dog
                correct query-
                SELECT * FROM 'table' WHERE  ARRAY_LENGTH(cls) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'person')) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'dog')) >= 1;
             """,
        },
        {"role": "user", "content": f"{query}"},
    ]

    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content
