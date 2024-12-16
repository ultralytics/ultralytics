import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from tqdm import tqdm
import os
from ultralytics.nn.text_model import build_text_model

@smart_inference_mode()
def generate_label_embedding(model, texts, batch=512):
    model = build_text_model(model, device='cuda')
    assert(not model.training)
    
    text_tokens = model.tokenize(texts)
    txt_feats = []
    for text_token in tqdm(text_tokens.split(batch)):
        txt_feats.append(model.encode_text(text_token))
    txt_feats = torch.cat(txt_feats, dim=0)
    return txt_feats.cpu()


def collect_grounding_labels(cache_path):
    labels = np.load(cache_path, allow_pickle=True)
    cat_names = set()
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_names.add(t)
    
    return cat_names

def collect_detection_labels(yaml_path):
    cat_names = set()
    
    data = yaml_load(yaml_path, append_filename=True)
    names = [name.split("/") for name in data["names"].values()]
    for name in names:
        for n in name:
            n = n.strip()
            assert(n)
            cat_names.add(n)
    
    return cat_names

if __name__ == '__main__':
    os.environ["PYTHONHASHSEED"] = "0"
    
    flickr_cache = '../datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache'
    mixed_grounding_cache = '../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache'
    objects365v1_yaml = 'ultralytics/cfg/datasets/Objects365v1.yaml'
    
    all_cat_names = set()
    all_cat_names |= collect_detection_labels(objects365v1_yaml)
    all_cat_names |= collect_grounding_labels(flickr_cache)
    all_cat_names |= collect_grounding_labels(mixed_grounding_cache)
    
    all_cat_names = list(all_cat_names)
    
    model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    all_cat_feats = generate_label_embedding(model, all_cat_names)
    
    cat_name_feat_map = {}
    for name, feat in zip(all_cat_names, all_cat_feats):
        cat_name_feat_map[name] = feat
    
    os.makedirs(f'tools/{model}', exist_ok=True)
    torch.save(cat_name_feat_map, f'tools/{model}/train_label_embeddings.pt')
