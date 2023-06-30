import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib.colors import TABLEAU_COLORS

from ultralytics import YOLO


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]


colors = color_list()


def xyxy2xywh(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5 * (x1 + x2) / W, 0.5 * (y1 + y2) / H, (x2 - x1) / W, (y2 - y1) / H]


def load_img(img_file, img_mean=0, img_scale=1 / 255):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return img


def model_inference(model_path, image_np):
    model = YOLO(model_path)

    output = model.predict(source=image_np, save=False)

    return output


def post_process(output):
    boxes = []
    texts = []

    for result in output:
        # Detection
        boxes.append(result.boxes.xyxy)  # box with xyxy format, (N, 4)
        texts.append(result.boxes.cls)  # cls, (N, 1)

    return boxes, texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./my_yolov8.pt', help='PyTorch model path')
    parser.add_argument('--img-path', type=str, help='input image path')
    parser.add_argument('--dst-path', type=str, default='./predictions', help='folder path destination')
    parser.add_argument('--device', type=str, default='cpu', help='device for model inference')
    parser.add_argument('--score-tresh', type=float, default=0.3, help='score threshold')
    parser.add_argument('--bbox-format',
                        type=str,
                        default='xywh',
                        help='bounding box format to save annotation (or xyxy)')
    args = parser.parse_args()

    assert args.device == 'cpu' or args.device == 'cuda'

    # Load image
    img = load_img(args.img_path)

    # Inference
    out = model_inference(args.model_path, img, args.device)

    # Post-processing
    start_time = time.time()
    out_img, out_txt = post_process(args.img_path, out, args.score_tresh, args.bbox_format)
    elapsed_time = time.time() - start_time
    print(f'Inference completed in {elapsed_time:.3f} secs.')

    # Save prediction
    os.makedirs(args.dst_path, exist_ok=True)
    bn = os.path.basename(args.img_path).split('.')[0]
    cv2.imwrite(os.path.join(args.dst_path, bn + '.png'), out_img[..., ::-1])
    with open(os.path.join(args.dst_path, bn + '.txt'), 'w') as f:
        f.write(out_txt)

    print(f'Predicted image and annotations are now saved in {args.dst_path}.')
