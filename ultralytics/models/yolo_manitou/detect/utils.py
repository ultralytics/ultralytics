
def invert_manitou_resize_crop_xyxy(bboxes, pre_crop_cfg):
    """
    Post-process bounding box predictions for Manitou detection.
    Details in `ManitouResizeCrop` data augmentation.
    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is the number of detections, xyxy format.
        pre_crop_cfg (dict): Configuration dictionary containing pre-crop settings.
            - is_crop (bool): Whether cropping was applied.
            - scale (float): Scaling factor used during cropping.
            - target_size (tuple): Target size after resizing and cropping.
            - original_size (tuple): Original size before resizing and cropping.
    """
    if pre_crop_cfg["is_crop"]:
        scale = pre_crop_cfg["scale"]
        target_size = pre_crop_cfg["target_size"]
        y_offset = pre_crop_cfg["original_size"][0] * scale - target_size[0]
        x_offset = 0

        bboxes[:, [1, 3]] += y_offset
        bboxes[:, :4] = bboxes[:, :4] / scale

    return bboxes