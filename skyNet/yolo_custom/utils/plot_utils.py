import os
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.patches as patches
import torch
from .bboxes_utils import non_max_suppression as nms
import skyNet.config as cfg

def cells_to_bboxes(predictions, anchors, strides, device, is_pred=False, to_list=True):
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize

    # device = "cpu"

    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)

        grid[i] = grid[i].to(device, non_blocking=True)
        anchor_grid[i] = anchor_grid[i].to(device, non_blocking=True)

        if is_pred:
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            #xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
            predictions[i] = predictions[i].to(device, non_blocking=True)
            layer_prediction = predictions[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
            wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to(device, non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i]) * stride
            wh = predictions[i][..., 2:4] * stride
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)

def make_grids(anchors, naxs, stride, nx=20, ny=20, i=0):

    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)

    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
    anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid


def save_predictions(model, loader, folder, epoch, device, filename, num_images=10, labels=cfg.COCO):

    print("=> Saving images predictions...")

    if not os.path.exists(path=os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}')):
        os.makedirs(os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}'))

    path = os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}')
    anchors = model.head.anchors

    model.eval()

    for idx, (images, targets) in enumerate(loader):

        images = images.to(device).float()/255

        if idx < num_images:
            with torch.no_grad():
                out = model(images)

            boxes = cells_to_bboxes(out, anchors, model.head.stride, device, is_pred=True, to_list=False)
            gt_boxes = cells_to_bboxes(targets, anchors, model.head.stride, device, is_pred=False, to_list=False)

            # here using different nms_iou_thresh and config_thresh because of
            # https://github.com/ultralytics/yolov5/issues/4464
            boxes = nms(boxes, iou_threshold=0.45, threshold=0.25)[0]
            gt_boxes = nms(gt_boxes, iou_threshold=0.45, threshold=0.7)[0]

            cmap = plt.get_cmap("tab20b")
            class_labels = labels
            colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
            im = np.array(images[0].permute(1, 2, 0).cpu())

            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # Display the image
            ax1.imshow(im)
            ax2.imshow(im)

            # box[0] is x midpoint, box[2] is width
            # box[1] is y midpoint, box[3] is height
            axes = [ax1, ax2]
            # Create a Rectangle patch
            boxes = [gt_boxes, boxes]
            for i in range(2):
                for box in boxes[i]:
                    assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"

                    class_pred = int(box[0])
                    box = box[2:]
                    upper_left_x = max(box[0], 0)
                    upper_left_x = min(upper_left_x, im.shape[1])
                    lower_left_y = max(box[1], 0)
                    lower_left_y = min(lower_left_y, im.shape[0])

                    # print(upper_left_x)
                    # print(lower_left_y)
                    rect = patches.Rectangle(
                        (upper_left_x, lower_left_y),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=1,
                        edgecolor=colors[class_pred],
                        facecolor="none",
                    )
                    # Add the patch to the Axes
                    if i == 0:
                        axes[i].set_title("Ground Truth bboxes")
                    else:
                        axes[i].set_title("Predicted bboxes")
                    axes[i].add_patch(rect)
                    axes[i].text(
                        upper_left_x,
                        lower_left_y,
                        s=class_labels[class_pred],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": colors[class_pred], "pad": 0},
                        fontsize="xx-small"
                    )

            fig.savefig(f'{path}/image_{idx}.png', dpi=300)
            plt.close(fig)
        # if idx > num images
        else:
            break

    model.train()


def plot_image(image, boxes, labels=cfg.COCO):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = labels
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        bbox = box[2:]

        # FOR MY_NMS attempts, also rect = patches.Rectangle box[2] becomes box[2] - box[0] and box[3] - box[1]
        upper_left_x = max(bbox[0], 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(bbox[1], 0)
        lower_left_y = min(lower_left_y, im.shape[0])

        """upper_left_x = max(box[0] - box[2] / 2, 0)
        upper_left_x = min(upper_left_x, im.shape[1])
        lower_left_y = max(box[1] - box[3] / 2, 0)
        lower_left_y = min(lower_left_y, im.shape[0])"""

        rect = patches.Rectangle(
            (upper_left_x, lower_left_y),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            lower_left_y,
            s=f"{class_labels[int(class_pred)]}: {box[1]:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.show()
