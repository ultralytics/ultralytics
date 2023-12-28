import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import plot_images, Annotator, colors
from ultralytics.utils import LOGGER as logger
from ultralytics.utils.checks import check_requirements
from ultralytics.data.augment import LetterBox
from typing import List
check_requirements('lancedb')
from lancedb.pydantic import LanceModel, Vector


def get_schema(vector_size):
    class Schema(LanceModel):
        im_file: str
        labels: List[str]
        cls: List[int]
        bboxes: List[List[float]]
        masks: List[List[List[int]]]
        keypoints: List[List[List[float]]]
        vector: Vector(vector_size)
    
    return Schema
    

def sanitize_batch(batch, dataset_info):
    batch['cls'] = batch['cls'].flatten().int().tolist()
    box_cls_pair = sorted(zip(batch['bboxes'].tolist(), batch['cls']), key=lambda x: x[1])
    batch['bboxes'] = [box for box, _ in box_cls_pair]
    batch['cls'] = [cls for _, cls in box_cls_pair]
    batch['labels'] = [dataset_info['names'][i] for i in batch['cls']]
    batch["masks"] = batch["masks"].tolist() if "masks" in batch else [[[]]]
    batch["keypoints"] = batch["keypoints"].tolist() if "keypoints" in batch else [[[]]]

    return batch

def plot_similar_images(similar_set):
    """
    Plot images from the similar set.

    Args:
        similar_set (list): Pyarrow table containing the similar data points
    """
    similar_set = similar_set.to_pydict()
    empty_masks = [[[]]]
    empty_boxes = [[]]
    images = similar_set.get('im_file', [])
    bboxes = similar_set.get('bboxes', []) if similar_set.get('bboxes') is not empty_boxes else []
    labels = similar_set.get('labels', [])
    masks = similar_set.get('masks') if similar_set.get('masks')[0] != empty_masks else []
    kpts = similar_set.get('keypoints') if similar_set.get('keypoints')[0] != empty_masks else []
    cls = similar_set.get('cls', [])

    # handle empty

    resized_images = []
    if len(images) == 0:
        logger.info('No similar images found')
        return
    
    for idx, img in enumerate(images):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if labels:
            ann = Annotator(img)
            if len(bboxes) > idx:
                [ann.box_label(bbox, label, color=colors(cls, True)) for bbox, label, cls in zip(bboxes[idx], labels[idx], cls[idx])]
            if len(masks) > idx:
                mask = torch.tensor(np.array(masks[idx]))
                img = LetterBox(mask.shape[1:])(image=ann.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=mask.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
                ann.masks(mask, colors=[colors(x, True) for x in cls[idx]], im_gpu=im_gpu)
            if len(kpts) > idx:
                [ann.kpts(torch.tensor(np.array(kpt))) for kpt in kpts[idx]]

            img = ann.result()
        resized_images.append(img)

    # Create a grid of the images
    cols = 10 if len(resized_images) > 10 else max(2, len(resized_images))
    rows = max(1, math.ceil(len(resized_images) / cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.ravel()):
        if i < len(resized_images):
            ax.imshow(resized_images[i])
        ax.axis("off")
    # Display the grid of images
    plt.show()
