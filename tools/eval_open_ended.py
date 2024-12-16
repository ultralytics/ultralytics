import argparse
import torch
import os
import random
import json
from tqdm import tqdm
from ultralytics.nn.text_model import MobileCLIP
import numpy as np
import argparse
from transformers import logging
import torch.multiprocessing as mp
from ultralytics.utils import yaml_load

import clip
import copy
from lvis import LVIS, LVISEval
from pathlib import Path

logging.set_verbosity_error()

class LVISOpenEndedEval():
    def __init__(self, anno_path, pred_path, name_path):
        self.anno_path = anno_path
        self.pred_path = pred_path
        
        categories = yaml_load("ultralytics/cfg/datasets/lvis.yaml")["names"].values()
        self.categories = [c.split("/")[0] for c in categories]

        with open(name_path, 'r') as f:
            self.names = [x.strip() for x in f.readlines()]

        self.topk_for_mapping = 1

    def setup_clip_matching(self):
        print("Loading CLIP model for matching")
        self.clip_model = MobileCLIP("blt", device="cuda")
        
        tokens = clip.tokenize(self.categories).cuda()
        text_features = self.clip_model.encode_text(tokens)
        
        self.lvis_embed = text_features.transpose(-1, -2)

    @torch.inference_mode()
    def match(self, total, rank, verbose=False):
        self.setup_clip_matching()
        
        predictions = []
        data_all = json.load(open(self.pred_path))
        data_all = np.array_split(data_all, total)[rank].tolist()

        batch = 512
        chunk_num = (len(data_all) + batch - 1) // batch
        data_all = np.array_split(data_all, chunk_num)
        
        for batch_data in tqdm(data_all):
            text = []
            for data in batch_data:
                description = self.names[data['category_id'] - 1]
                text.append(description)

            tokens = clip.tokenize(text).cuda()
            text_features = self.clip_model.encode_text(tokens)
            similarity = (100.0 * text_features @ self.lvis_embed).softmax(dim=-1)
            sim_values, sim_indices = similarity.topk(self.topk_for_mapping)
            category_ids = sim_indices.cpu().numpy() + 1
            scores = sim_values.cpu().numpy()
            for i, data in enumerate(batch_data):
                for cat_id, score in zip(category_ids[i], scores[i]):
                    d = copy.deepcopy(data)
                    d['category_id'] = int(cat_id)
                    predictions.append(d)
                    if verbose:
                        print(f"[pred | gt] ({score:.2f}): {text[i]} | {self.categories[cat_id-1]}")
            
        return predictions

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker(args):
    json, pred, names, total, rank, device, seed, verbose = args
    setup_seed(seed)
    torch.cuda.set_device(int(device))
    
    lvis_eval = LVISOpenEndedEval(json, pred, names)
    match_result = lvis_eval.match(total, rank, verbose=(rank == 0) and verbose)
    return match_result

def main(args):
    pred_file = Path(args.pred).with_suffix('.mt.json')
    devices = args.devices.split(",")
    ranks = list(range(len(devices)))
    total = len(devices)
    mp.set_start_method('spawn')
    with mp.Pool(total) as pool:
        results = pool.map(worker,
                        [(args.json, args.pred, args.names,
                            total, rank, device, args.seed, args.verbose)
                            for rank, device in zip(ranks, devices)])

    predictions = []
    for result in results:
        predictions.extend(result)
    print(f"Total predictions: {len(predictions)}")
    predictions = sorted(predictions, key=lambda x: (x['image_id'], -x['score']))
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {pred_file}")

    if args.fixed:
        print("Evaluating LVIS fixed mAP...")
        os.system(f"python ./tools/eval_fixed_ap.py {args.json} {pred_file}")
    else:
        print("Evaluating LVIS standard mAP...")
        anno = LVIS(args.json)
        pred = anno._load_json(pred_file)
        lvis_eval = LVISEval(anno, pred, "bbox")
        lvis_eval.evaluate()
        lvis_eval.accumulate()
        lvis_eval.summarize()
        lvis_eval.print_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--json', help='path to lvis minival json file', default='./data/lvis_v1_minival.json')
    parser.add_argument('--pred', help='path to pred json file', default='./data/predictions.json')
    parser.add_argument('--names', help='path to vocab names', default='./tools/ram_tag_list.txt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--verbose', help='verbose', action='store_true')
    parser.add_argument('--fixed', help='evaluate by fixed ap', action='store_true')
    args = parser.parse_args()

    main(args)