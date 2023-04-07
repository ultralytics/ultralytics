import json
import os
from utils.utils import coco91_2_coco80
from tqdm import tqdm
import glob
import csv
from time import sleep
"""
with open("/Users/Alessandro/desktop/ML/DL_DATASETS/COCO/annotations/instances_train2017.json") as f:
    coco_ann = json.load(f)

coco_parsed = []

loop = tqdm(coco_ann["images"])

for idx, image in enumerate(loop):
    if idx < 16:
        name = image["file_name"]
        h, w = image["height"], image["width"]
        id = image["id"]
        bboxes = []

        for ann in coco_ann["annotations"]:
            if ann["image_id"] == id:
                box = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
                bboxes.append({"box": box, "class": ann["category_id"]})

        coco_parsed.append(
            {"img_name": name,
             "height": h,
             "width": w,
             "bboxes": bboxes}
        )
    else:
        break


path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_16.json"

with open(path, "w") as f:
    json.dump(coco_parsed, f)


"""
"""


path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_2017_train_AM.json"
with open(path, "r") as f:
    annotations = json.load(f)

loop = tqdm(annotations)

for idx, annot in enumerate(loop):
    if annot["img_name"] in os.listdir("../datasets/coco128/images/train2017"):
        img_name = annot["img_name"]
        height = annot["height"]
        width = annot["width"]

        with open("/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/"
                  "annotations/coco_2017_coco128_csv.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([img_name, height, width])
            f.close()

        folder_path = "/Users/Alessandro/Desktop/ML/DL_DATASETS/COCO/annotations/coco_2017_coco128_txt"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, "{}.txt".format(img_name[:-4])), "w") as fp:
            for bbox in annot["bboxes"]:
                box = bbox["box"]
                x, y, w, h = box
                # cleans empty bboxes
                if w > 0.1 and h > 0.1:
                    w = width if w >= width else w
                    h = height if h >= height else h
                    box = [x, y, w, h, coco91_2_coco80(bbox["class"])]
                    fp.write(str(box).strip("[]").replace(",", "") + "\n")
            fp.close()
"""