from pathlib import Path

import cv2

IMAGES = Path("data/raw/images")
LABELS = Path("data/raw/labels")
OUT_SAFE = Path("data/processed/crops_safe")
OUT_UNSAFE = Path("data/processed/crops_unsafe")

OUT_SAFE.mkdir(parents=True, exist_ok=True)
OUT_UNSAFE.mkdir(parents=True, exist_ok=True)


# Helper: IOU
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def read_yolo_labels(label_file, img_w, img_h):
    boxes = []
    with open(label_file) as f:
        for line in f:
            _cls, x, y, w, h = map(float, line.split())
            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def generate_crops():
    for img_path in IMAGES.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = LABELS / (img_path.stem + ".txt")

        if label_path.exists():
            boxes = read_yolo_labels(label_path, w, h)
        else:
            boxes = []

        # create grid patches
        patch_size = int(min(w, h) * 0.25)
        k = 0
        for cx in range(patch_size // 2, w, patch_size):
            for cy in range(patch_size // 2, h, patch_size):
                x1 = max(0, cx - patch_size // 2)
                y1 = max(0, cy - patch_size // 2)
                x2 = min(w, cx + patch_size // 2)
                y2 = min(h, cy + patch_size // 2)

                patch = img[y1:y2, x1:x2]

                overlap = any(iou((x1, y1, x2, y2), box) > 0.1 for box in boxes)

                if overlap:
                    # unsafe crop
                    cv2.imwrite(str(OUT_UNSAFE / f"{img_path.stem}_u_{k}.jpg"), patch)
                else:
                    # safe crop
                    cv2.imwrite(str(OUT_SAFE / f"{img_path.stem}_s_{k}.jpg"), patch)

                k += 1

    print("Crops generated successfully!")


if __name__ == "__main__":
    generate_crops()
