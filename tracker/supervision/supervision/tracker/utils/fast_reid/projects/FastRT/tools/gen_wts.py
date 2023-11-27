# encoding: utf-8

import sys
import time
import struct
import argparse
sys.path.append('.')

import torch
import torchvision
#from torchsummary import summary

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch import build_model
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer

sys.path.append('./projects/FastDistill')
from fastdistill import *

def setup_cfg(args):
    # load confiimport argparseg from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Encode pytorch weights for tensorrt.")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--wts_path",
        default='./trt_demo',
        help='path to save tensorrt weights file(.wts)'
    )
    parser.add_argument(
        "--show_model",
        action='store_true',
        help='print model architecture'
    )
    parser.add_argument(
        "--verify",
        action='store_true',
        help='print model output for verify'
    )
    parser.add_argument(
        "--benchmark",
        action='store_true',
        help='preprocessing + inference time'
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def gen_wts(args):
    """
        Thanks to https://github.com/wang-xinyu/tensorrtx
    """
    print("Wait for it: {} ...".format(args.wts_path))
    f = open(args.wts_path, 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        #print('key: ', k)
        #print('value: ', v.shape)     
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
        
if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    print("[Config]: \n", cfg)
    
    model = build_model(cfg)
    
    if args.show_model:
        print('[Model]: \n', model)
        #summary(model, (3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]))
    
    print("Load model from: ", cfg.MODEL.WEIGHTS)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    
    model = model.to(cfg.MODEL.DEVICE)
    model.eval()
    
    if args.verify:
        input = torch.ones(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).to(cfg.MODEL.DEVICE) * 255.
        out = model(input).view(-1).cpu().detach().numpy()
        print('[Model output]: \n', out) 
        
    if args.benchmark:
        start_time = time.time()
        input = torch.ones(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).to(cfg.MODEL.DEVICE) * 255.
        for i in range(100):
            out = model(input).view(-1).cpu().detach()
        print("--- %s seconds ---" % ((time.time() - start_time)/100.) )
    
    gen_wts(args)
    