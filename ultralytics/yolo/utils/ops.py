# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ultralytics.yolo.utils import LOGGER

from .metrics import box_iou


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def segment2box(segment, width=640, height=640):
    """
    > Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to
    (xyxy)
    Args:
      segment: the segment label
      width: the width of the image. Defaults to 640
      height: The height of the image. Defaults to 640

    Returns:
      the minimum and maximum x and y values of the segment.
    """
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros(4)  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    > Rescale boxes (xyxy) from img1_shape to img0_shape
    Args:
      img1_shape: The shape of the image that the bounding boxes are for.
      boxes: the bounding boxes of the objects in the image
      img0_shape: the shape of the original image
      ratio_pad: a tuple of (ratio, pad)

    Returns:
      The boxes are being returned.
    """
    #
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """
    > Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nm (int): The number of masks output by the model.

    Returns:
        List[torch.Tensor]: A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """
    > It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes: the bounding boxes to clip
      shape: the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    > It takes a mask, and resizes it to the original image size

    Args:
      im1_shape: model input shape, [h, w]
      masks: [h, w, num]
      im0_shape: the original image shape
      ratio_pad: the ratio of the padding to the original image.

    Returns:
      The masks are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def xyxy2xywh(x):
    """
    > It takes a list of bounding boxes, and converts them from the format [x1, y1, x2, y2] to [x, y, w,
    h]  where xy1=top-left, xy2=bottom-right

    Args:
      x: the input tensor

    Returns:
      the center of the box, the width and the height of the box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    > It converts the bounding box from x,y,w,h to x1,y1,x2,y2 where xy1=top-left, xy2=bottom-right

    Args:
      x: the input tensor

    Returns:
      the top left and bottom right coordinates of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    > It converts the normalized coordinates to the actual coordinates [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
      x: the bounding box coordinates
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      padw: padding width. Defaults to 0
      padh: height of the padding. Defaults to 0

    Returns:
      the xyxy coordinates of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    > It takes in a list of bounding boxes, and returns a list of bounding boxes, but with the x and y
    coordinates normalized to the width and height of the image

    Args:
      x: the bounding box coordinates
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      clip: If True, the boxes will be clipped to the image boundaries. Defaults to False
      eps: the minimum value of the box's width and height.

    Returns:
      the xywhn format of the bounding boxes.
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    > It converts normalized segments into pixel segments of shape (n,2)

    Args:
      x: the normalized coordinates of the bounding box
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      padw: padding width. Defaults to 0
      padh: padding height. Defaults to 0

    Returns:
      the x and y coordinates of the top left corner of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def xywh2ltwh(x):
    """
    > It converts the bounding box from [x, y, w, h] to [x1, y1, w, h] where xy1=top-left

    Args:
      x: the x coordinate of the center of the bounding box

    Returns:
      the top left x and y coordinates of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    > Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right

    Args:
      x: the input tensor

    Returns:
      the xyxy2ltwh function.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def ltwh2xywh(x):
    """
    > Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center

    Args:
      x: the input tensor
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] + x[:, 2] / 2  # center x
    y[:, 1] = x[:, 1] + x[:, 3] / 2  # center y
    return y


def ltwh2xyxy(x):
    """
    > It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left,
    xy2=bottom-right

    Args:
      x: the input image

    Returns:
      the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]  # width
    y[:, 3] = x[:, 3] + x[:, 1]  # height
    return y


def segments2boxes(segments):
    """
    > It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
      segments: list of segments, each segment is a list of points, each point is a list of x, y
    coordinates

    Returns:
      the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    > It takes a list of segments (n,2) and returns a list of segments (n,2) where each segment has been
    up-sampled to n points

    Args:
      segments: a list of (n,2) arrays, where n is the number of points in the segment.
      n: number of points to resample the segment to. Defaults to 1000

    Returns:
      the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    > It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks: [h, w, n] tensor of masks
      boxes: [n, 4] tensor of bbox coords in relative point form

    Returns:
      The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    > It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.

    Args:
      protos: [mask_dim, mask_h, mask_w]
      masks_in: [n, mask_dim], n is number of masks after nms
      bboxes: [n, 4], n is number of masks after nms
      shape: the size of the input image

    Returns:
      mask
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    > It takes the output of the mask head, and applies the mask to the bounding boxes. This is faster but produces
    downsampled quality of mask

    Args:
      protos: [mask_dim, mask_h, mask_w]
      masks_in: [n, mask_dim], n is number of masks after nms
      bboxes: [n, 4], n is number of masks after nms
      shape: the size of the input image

    Returns:
      mask
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    > It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
      protos: [mask_dim, mask_h, mask_w]
      masks_in: [n, mask_dim], n is number of masks after nms
      bboxes: [n, 4], n is number of masks after nms
      shape: input_image_size, (h, w)

    Returns:
      masks: [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    """
    > Rescale segment coords (xyxy) from img1_shape to img0_shape

    Args:
      img1_shape: The shape of the image that the segments are from.
      segments: the segments to be scaled
      img0_shape: the shape of the image that the segmentation is being applied to
      ratio_pad: the ratio of the image size to the padded image size.
      normalize: If True, the coordinates will be normalized to the range [0, 1]. Defaults to False

    Returns:
      the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def masks2segments(masks, strategy='largest'):
    """
    > It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
      masks: the output of the model, which is a tensor of shape (batch_size, 160, 160)
      strategy: 'concat' or 'largest'. Defaults to largest

    Returns:
      segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments


def clip_segments(segments, shape):
    """
    > It takes a list of line segments (x1,y1,x2,y2) and clips them to the image shape (height, width)

    Args:
      segments: a list of segments, each segment is a list of points, each point is a list of x,y
    coordinates
      shape: the shape of the image
    """
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
