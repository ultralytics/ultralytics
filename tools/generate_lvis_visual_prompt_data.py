from pathlib import Path
from collections import defaultdict
import random
import numpy as np
import shutil
import os
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(0)

    lvis_path = "../datasets/lvis"
    lvis_train_path = f"{lvis_path}/train.txt"
    
    visual_prompt_cache_path = Path("../datasets/lvis_train_vps")

    shutil.rmtree(visual_prompt_cache_path / "images", ignore_errors=True)
    shutil.rmtree(visual_prompt_cache_path / "labels", ignore_errors=True)
    os.makedirs(visual_prompt_cache_path / "images")
    os.makedirs(visual_prompt_cache_path / "labels")
    
    with open(lvis_train_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    im_files = [str(Path(lvis_path) / f) for f in lines]
    label_files = [f.replace("images", "labels").replace( \
        "jpg", "txt") for f in im_files]
    
    cls_file_map = defaultdict(list)
    file_label_map = {}
    
    for im_file, label_file in tqdm(zip(im_files, label_files), total=len(im_files)):
        if not os.path.exists(label_file):
            continue
        with open(label_file, 'r') as f:
            labels = f.readlines()
        
        cls = [int(label.split()[0]) for label in labels]
        for c in sorted(set(cls)):
            cls_file_map[c].append(im_file)
        assert(im_file not in file_label_map)
        file_label_map[im_file] = labels
            
    N = 16
    cls_sample_file_map = {}

    for c, files in tqdm(cls_file_map.items(), total=len(cls_file_map)):
        cls_sample_file_map[c] = random.sample(list(files), k=min(N, len(files)))
        
    instance_count = 0
    sample_labels = []
    sample_id = 0
    for c, files in tqdm(cls_sample_file_map.items(), total=len(cls_sample_file_map)):
        for file in files:
            labels = file_label_map[file]
            cls = np.array([int(label.split()[0]) for label in labels])
            index = (cls == c)
            valid_labels = np.array(labels)[index].tolist()
            shutil.copy(file, visual_prompt_cache_path / "images" / f"{sample_id}.jpg")
            with open(visual_prompt_cache_path / "labels" / f"{sample_id}.txt", "w") as f:
                f.write("\n".join(valid_labels))
            sample_id += 1
    
    assert(sample_id == 15098)
