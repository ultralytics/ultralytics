# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.utils.logger import setup_logger
from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


if __name__ == '__main__':
    args = get_parser().parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    train_data = 'DukeMTMC'
    method = 'sbs_S50'  # bagtricks_S50 | sbs_S50
    seq = 'MOT20-02'

    args.config_file = r'../configs/' + train_data + '/' + method + '.yml'
    args.input = [r'/home/nir/Datasets/MOT20/train/' + seq + '/img1', '*.jpg']
    args.output = seq + '_' + method + '_' + train_data
    args.opts = ['MODEL.WEIGHTS', '../pretrained/duke_bot_S50.pth']
    # ------------------------------------------------------------------------------------------------------------------

    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    PathManager.mkdirs(args.output)
    if args.input:
        if PathManager.isdir(args.input[0]):
            # args.input = glob.glob(os.path.expanduser(args.input[0]))
            args.input = glob.glob(os.path.expanduser(os.path.join(args.input[0], args.input[1])))
            args.input = sorted(args.input)
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            feat = demo.run_on_image(img)
            feat = postprocess(feat)
            np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)
