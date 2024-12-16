import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import json
from generate_label_embedding import generate_label_embedding
import torch
from ultralytics.utils import yaml_load

def obtain_cat_freq(cache_path, cat_name_freq):
    labels = np.load(cache_path, allow_pickle=True)
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_name_freq[t] += 1

if __name__ == '__main__':
    os.environ["PYTHONHASHSEED"] = "0"
    cat_name_freq = defaultdict(int)
    
    flickr_cache_path = Path('../datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache')
    obtain_cat_freq(flickr_cache_path, cat_name_freq)

    mixed_grounding_cache_path = Path('../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache')
    obtain_cat_freq(mixed_grounding_cache_path, cat_name_freq)

    global_neg_cat = []
    for k, v in cat_name_freq.items():
        if v >= 100:
            global_neg_cat.append(k)

    print(len(global_neg_cat))

    with open('tools/global_grounding_neg_cat.json', 'w') as f:
        json.dump(global_neg_cat, f, indent=2)
    
    model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    global_neg_embeddings = generate_label_embedding(model, global_neg_cat)
    os.makedirs(f'tools/{model}', exist_ok=True)
    torch.save(global_neg_embeddings, f'tools/{model}/global_grounding_neg_embeddings.pt')
        
