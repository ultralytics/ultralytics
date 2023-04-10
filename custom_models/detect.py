import argparse
import os
import torch
import numpy as np
from custom_models.yolo_custom.models.yolov5m import YOLOV5m
from utils.utils import load_model_checkpoint
from utils.plot_utils import cells_to_bboxes, plot_image
from utils.bboxes_utils import non_max_suppression
from PIL import Image
import random
import config


if __name__ == "__main__":
    # do not modify
    first_out = config.FIRST_OUT
    nc = len(config.FLIR)
    img_path = "TFront-South-09-31-48-31-04610_jpg.rf.89effbdf6e51b340ad5d12b37e0da7b1.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,default="model_1" ,help="Indicate the folder inside SAVED_CHECKPOINT")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_8.pth.tar", help="Indicate the ckpt name inside SAVED_CHECKPOINT/model_name")
    parser.add_argument("--img", type=str, default=img_path, help="Indicate path to the img to predict")
    parser.add_argument("--save_pred", action="store_true", help="If save_pred is set, prediction is saved in detections_exp")
    args = parser.parse_args()

    random_img = not args.img

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=config.ANCHORS,
                    ch=(first_out * 4, first_out * 8, first_out * 16)).to(config.DEVICE)

    path2model = os.path.join("../../../Downloads/YOLOV5m-main/YOLOV5m-main/SAVED_CHECKPOINT", args.model_name, args.checkpoint)
    load_model_checkpoint(model=model, model_name=path2model, training=False)

    config.ROOT_DIR = "/".join((config.ROOT_DIR.split("/")[:-1] + ["flir"]))
    imgs = os.listdir(os.path.join(config.ROOT_DIR, "images", "test"))
    if random_img:
        img = np.array(Image.open(os.path.join(config.ROOT_DIR, "images", "test", random.choice(imgs))))
    else:

        img = np.array(Image.open(os.path.join(config.ROOT_DIR, "images", "test", args.img)))

    img = img.transpose((2, 0, 1))
    img = img[None, :]
    img = torch.from_numpy(img)
    img = img.float() / 255

    with torch.no_grad():
        out = model(img)

    bboxes = cells_to_bboxes(out, model.head.anchors, model.head.stride, is_pred=True, to_list=False)
    bboxes = non_max_suppression(bboxes, iou_threshold=0.45, threshold=0.25, to_list=False)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), bboxes, config.FLIR)


