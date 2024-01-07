from typing import List

import cv2
import numpy as np

from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import plot_images


def get_table_schema(vector_size):
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
    from lancedb.pydantic import LanceModel

    class Schema(LanceModel):
        idx: int
        im_file: str
        count: int
        sim_im_files: List[str]

    return Schema


def sanitize_batch(batch, dataset_info):
    batch['cls'] = batch['cls'].flatten().int().tolist()
    box_cls_pair = sorted(zip(batch['bboxes'].tolist(), batch['cls']), key=lambda x: x[1])
    batch['bboxes'] = [box for box, _ in box_cls_pair]
    batch['cls'] = [cls for _, cls in box_cls_pair]
    batch['labels'] = [dataset_info['names'][i] for i in batch['cls']]
    batch['masks'] = batch['masks'].tolist() if 'masks' in batch else [[[]]]
    batch['keypoints'] = batch['keypoints'].tolist() if 'keypoints' in batch else [[[]]]

    return batch


def plot_similar_images(similar_set, plot_labels=True):
    """
    Plot images from the similar set.

    Args:
        similar_set (list): Pyarrow table containing the similar data points
        plot_labels (bool): Whether to plot labels or not
    """
    similar_set = similar_set.to_pydict()
    empty_masks = [[[]]]
    empty_boxes = [[]]
    images = similar_set.get('im_file', [])
    bboxes = similar_set.get('bboxes', []) if similar_set.get('bboxes') is not empty_boxes else []
    masks = similar_set.get('masks') if similar_set.get('masks')[0] != empty_masks else []
    kpts = similar_set.get('keypoints') if similar_set.get('keypoints')[0] != empty_masks else []
    cls = similar_set.get('cls', [])

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
    masks = np.stack(plot_masks, axis=0) if len(plot_masks) > 0 else np.zeros(0, dtype=np.uint8)
    kpts = np.concatenate(plot_kpts, axis=0) if len(plot_kpts) > 0 else np.zeros((0, 51), dtype=np.float32)
    boxes = xyxy2xywh(np.concatenate(plot_boxes, axis=0)) if len(plot_boxes) > 0 else np.zeros(0, dtype=np.float32)
    batch_idx = np.concatenate(batch_idx, axis=0)
    cls = np.concatenate([np.array(c, dtype=np.int32) for c in cls], axis=0)

    return plot_images(imgs,
                       batch_idx,
                       cls,
                       bboxes=boxes,
                       masks=masks,
                       kpts=kpts,
                       max_subplots=len(images),
                       save=False,
                       threaded=False)
