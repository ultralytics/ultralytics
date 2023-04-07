import numpy as np
import torch
from torchvision.ops import nms

# ALADDIN'S
def iou_width_height(gt_box, anchors, strided_anchors=True, stride=[8, 16, 32]):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
        strided_anchors (bool): if the anchors are divided by the stride or not
    Returns:
        tensor: Intersection over union between the gt_box and each of the n-anchors
    """
    # boxes 1 (gt_box): shape (2,)
    # boxes 2 (anchors): shape (9,2)
    # intersection shape: (9,)
    anchors /= 640
    if strided_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)

    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    union = (
        gt_box[..., 0] * gt_box[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection
    )
    # intersection/union shape (9,)
    return intersection / union


# ALADDIN'S MODIFIED
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, eps=1e-7):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        GIoU (bool): if True it computed GIoU loss (https://giou.stanford.edu)
        eps (float): for numerical stability

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:  # if not midpoints box coordinates are considered to be in coco format
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    w1, h1, w2, h2 = box1_x2 - box1_x1, box1_y2 - box1_y1, box2_x2 - box2_x1, box2_y2 - box2_y1
    # Intersection area
    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
            (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c_area = cw * ch + eps  # convex height
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

# found here: https://gist.github.com/cbernecker/1ac2f9d45f28b6a4902ba651e3d4fa91#file-coco_to_yolo-py
def coco_to_yolo(bbox, image_w=640, image_h=640):
    x1, y1, w, h = bbox
    #return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return [((2*x1 + w)/(2*image_w)), ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def coco_to_yolo_tensors(bbox, w0=640, h0=640):
    x1, y1, w, h = np.split(bbox, 4, axis=1)
    #return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return np.concatenate([((2*x1 + w)/(2*w0)), ((2*y1 + h)/(2*h0)), w/w0, h/h0], axis=1)


# rescales bboxes from an image_size to another image_size
"""def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    new_boxes = []
    for bbox in bboxes:
        x = math.floor(bbox[0] * ew/sw * 100)/100
        y = math.floor(bbox[1] * eh/sh * 100)/100
        w = math.floor(bbox[2] * ew/sw * 100)/100
        h = math.floor(bbox[3] * eh/sh * 100)/100
        
        new_boxes.append([x, y, w, h])
    return new_boxes"""


def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    y = np.copy(bboxes)

    y[:, 0:1] = np.floor(bboxes[:, 0:1] * ew / sw * 100)/100
    y[:, 1:2] = np.floor(bboxes[:, 1:2] * eh / sh * 100)/100
    y[:, 2:3] = np.floor(bboxes[:, 2:3] * ew / sw * 100)/100
    y[:, 3:4] = np.floor(bboxes[:, 3:4] * eh / sh * 100)/100

    return y

# ALADDIN'S
def non_max_suppression_aladdin(bboxes, iou_threshold, threshold, box_format="corners", max_detections=300):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    if len(bboxes) > max_detections:
        bboxes = bboxes[:max_detections]

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True):

    """new_bboxes = []
    for box in bboxes:
        if box[1] > threshold:
            box[3] = box[0] + box[3]
            box[2] = box[2] + box[4]
            new_bboxes.append(box)"""

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
        boxes = boxes[indices]

        # sorts boxes by objectness score but it's already done internally by torch metrics's nms
        # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
        # boxes = boxes[si, :]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(
            boxes.tolist() if tolist else boxes
        )

    return bboxes_after_nms if tolist else torch.cat(bboxes_after_nms, dim=0)



