import os
from tqdm import tqdm
from pycocotools.coco import COCO
import shutil

# =========================
# CONFIG
# =========================
DATA_DIR = "tmp/coco"
OUTPUT_DIR = "tmp/coco_pva"  # people-vehicle-animal

SPLITS = ["train2017", "val2017"]

# COCO category IDs
PERSON_ID = 1
VEHICLE_IDS = [2, 3, 4, 6, 8]  # bicycle, car, motorcycle, bus, truck

# Option A: All animals
ANIMAL_IDS = [16,17,18,19,20,21,22,23,24,25]

# Option B (recommended for CCTV): only dog + cat
# ANIMAL_IDS = [17, 18]

# Filtering
MIN_BBOX_SIZE = 0  # pixels (set 0 to disable)

# =========================
# UTILS
# =========================
def convert_bbox(img_w, img_h, bbox):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

# =========================
# MAIN
# =========================
for split in SPLITS:
    print(f"\nProcessing {split}...")

    ann_file = os.path.join(DATA_DIR, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(DATA_DIR, split)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    out_img_dir = os.path.join(OUTPUT_DIR, "images", split)
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    kept_images = 0

    for img_id in tqdm(img_ids):
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        label_lines = []

        for ann in anns:
            cat_id = ann["category_id"]

            # Class mapping
            if cat_id == PERSON_ID:
                cls = 0
            elif cat_id in VEHICLE_IDS:
                cls = 1
            elif cat_id in ANIMAL_IDS:
                cls = 2
            else:
                continue

            x, y, w, h = ann["bbox"]

            # Optional bbox size filtering
            if MIN_BBOX_SIZE > 0:
                if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
                    continue

            bbox = convert_bbox(img["width"], img["height"], (x, y, w, h))

            label_lines.append(f"{cls} {' '.join(f'{b:.6f}' for b in bbox)}")

        # Skip images with no valid labels
        if not label_lines:
            continue

        # Copy image
        src = os.path.join(img_dir, img["file_name"])
        dst = os.path.join(out_img_dir, img["file_name"])

        if not os.path.exists(dst):
            shutil.copy(src, dst)

        # Write label file
        label_path = os.path.join(
            out_lbl_dir,
            os.path.splitext(img["file_name"])[0] + ".txt"
        )

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        kept_images += 1

    print(f"{split}: kept {kept_images} images")

print("\n✅ Conversion complete!")
