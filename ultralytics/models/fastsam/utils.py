# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """
    Adjust bounding boxes to stick to image border if they are within a certain threshold.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape (n, 4) in xyxy format.
        image_shape (Tuple[int, int]): Image dimensions as (height, width).
        threshold (int): Pixel threshold for considering a box close to the border.

    Returns:
       boxes (torch.Tensor): Adjusted bounding boxes with shape (n, 4).
    """
    # Image dimensions
    h, w = image_shape

    # Adjust boxes that are close to image borders
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes
