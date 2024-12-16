from tqdm import tqdm

import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
import shutil
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.utils import yaml_save
from ultralytics.data.converter import merge_multi_segment

base = '../datasets/Objects365v1'
duplicates = 0

for split in ['train']:
    labels = Path(f'{base}/labels') / split
    shutil.rmtree(labels, ignore_errors=True)
    labels.mkdir(parents=True, exist_ok=False)
    coco = COCO(f'{base}/annotations/objects365_{split}_segm.json')
    names = [x["name"] for x in coco.loadCats(sorted(coco.getCatIds()))]
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])  # image filename
            segments = set()
            with open(labels / path.with_suffix('.txt').name, 'a') as file:
                annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                for a in coco.loadAnns(annIds):
                    if a['iscrowd']:
                        continue

                    if a.get("segmentation") is not None and len(a["segmentation"]) > 0:
                        if len(a["segmentation"]) > 1:
                            s = merge_multi_segment(a["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([width, height])).reshape(-1).tolist()
                        else:
                            s = [j for i in a["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([width, height])).reshape(-1).tolist()
                        s = tuple([cid] + s)
                        # de-duplicate as some images have the same segment labels (due to highly similar boxes)
                        if s not in segments:
                            segments.add(s)
                            file.write(("%g " * len(s)).rstrip() % s + "\n")
                            file.flush()
                        else:
                            duplicates += 1
                    else:
                        assert(False)
                        box = np.array(a["bbox"], dtype=np.float64)
                        
                        box[2] += box[0]
                        box[3] += box[1]
                        
                        box = xyxy2xywhn(box, w=width, h=height, clip=True)
                        
                        if box[2] <= 0 or box[3] <= 0:
                            continue
                        
                        file.write(f"{cid} {box[0]:g} {box[1]:g} {box[2]:g} {box[3]:g}\n")
                        file.flush()

print("Total segmentation duplicates:", duplicates)

# Create Objects365v1.yaml
coco = COCO(f'{base}/annotations/objects365_train.json')
data = dict()

data['path'] = f'{base}'
data['train'] = 'images/train'
data['val'] = None
data['test'] = None

cat_id = sorted(coco.getCatIds())
data['names'] = {x['id'] - 1: x['name'].strip().lower() for x in coco.loadCats(cat_id)}

yaml_save('ultralytics/cfg/datasets/Objects365v1.yaml', data)