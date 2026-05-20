from pathlib import Path

import cv2
import numpy as np


DATASETS = ("NUAA-SIRST", "NUDT-SIRST")
CLASS_ID = 0
MIN_AREA = 1


def read_grayscale(path: Path):
    """Read images from Unicode paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def label_stem(mask_stem: str) -> str:
    """NUAA mask names look like Misc_1_pixels0.png, but images are Misc_1.png."""
    return mask_stem.removesuffix("_pixels0")


def mask_to_yolo_boxes(mask: np.ndarray) -> list[str]:
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = mask.shape[:2]
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        boxes.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return boxes


def convert_dataset(dataset_dir: Path) -> tuple[int, int, int]:
    mask_dir = dataset_dir / "masks"
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    empty = 0
    skipped = 0

    for mask_path in sorted(mask_dir.glob("*.png")):
        stem = label_stem(mask_path.stem)
        image_path = image_dir / f"{stem}.png"
        if not image_path.exists():
            print(f"[skip] image not found for mask: {mask_path}")
            skipped += 1
            continue

        mask = read_grayscale(mask_path)
        if mask is None:
            print(f"[skip] failed to read mask: {mask_path}")
            skipped += 1
            continue

        boxes = mask_to_yolo_boxes(mask)
        label_path = label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8")

        converted += 1
        if not boxes:
            empty += 1

    return converted, empty, skipped


def main() -> None:
    root = Path(__file__).resolve().parent
    total_converted = 0
    total_empty = 0
    total_skipped = 0

    for dataset in DATASETS:
        dataset_dir = root / dataset
        if not dataset_dir.exists():
            print(f"[skip] dataset not found: {dataset_dir}")
            continue

        converted, empty, skipped = convert_dataset(dataset_dir)
        total_converted += converted
        total_empty += empty
        total_skipped += skipped
        print(f"{dataset}: converted={converted}, empty={empty}, skipped={skipped}")

    print(
        f"done: converted={total_converted}, empty={total_empty}, skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()
