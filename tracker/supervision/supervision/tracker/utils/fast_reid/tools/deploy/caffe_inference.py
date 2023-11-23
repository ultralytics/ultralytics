# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import caffe
import tqdm
import glob
import os
import cv2
import numpy as np

caffe.set_mode_gpu()

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Caffe model inference")

    parser.add_argument(
        "--model-def",
        default="logs/test_caffe/baseline_R50.prototxt",
        help="caffe model prototxt"
    )
    parser.add_argument(
        "--model-weights",
        default="logs/test_caffe/baseline_R50.caffemodel",
        help="caffe model weights"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='caffe_output',
        help='path to save converted caffe model'
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    image = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    image = image.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    image = (image - np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape((1, -1, 1, 1))) / np.array(
        [0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape((1, -1, 1, 1))
    return image


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


if __name__ == "__main__":
    args = get_parser().parse_args()

    net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)
    net.blobs['blob1'].reshape(1, 3, args.height, args.width)

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            image = preprocess(path, args.height, args.width)
            net.blobs["blob1"].data[...] = image
            feat = net.forward()["output"]
            feat = normalize(feat[..., 0, 0], axis=1)
            np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)

