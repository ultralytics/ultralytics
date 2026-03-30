# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO11 Instance Segmentation with Axelera Voyager SDK.

Standalone example using the axelera-rt pipeline API.
No ultralytics dependency at runtime.

Usage:
    python yolo11-seg.py --model yolo11n-seg.axm --source 0
    python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from axelera.runtime import op

# fmt: off
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
# fmt: on

# Deterministic per-class color palette (BGR)
_rng = np.random.default_rng(42)
CLASS_COLORS = _rng.integers(50, 230, size=(80, 3)).tolist()


def build_pipeline(model_path: str, conf: float = 0.25, iou: float = 0.45):
    """Build YOLO11 instance segmentation pipeline.

    Data flow:
        decode_segmentation -> (detections, protos) as a tuple
        par(itemgetter+nms, itemgetter) -> (filtered_dets, protos) unpacked
        par(pack+itemgetter+to_image_space, proto_to_mask) -> (img_dets, masks)

    Calls .optimized() so the runtime can fuse operators for maximum throughput.
    """
    model_op = op.load(model_path)
    return op.seq(
        op.colorconvert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
        op.letterbox(640, 640),
        op.totensor(),
        model_op,
        op.decode_segmentation(algo="yolo11", num_classes=80, num_mask_coeffs=32, confidence_threshold=conf),
        # decode_segmentation returns (detections, protos) as a tuple
        op.par(
            op.seq(op.itemgetter(0), op.nms(iou_threshold=iou, max_boxes=300)),  # NMS on detections
            op.itemgetter(1),  # pass protos through
        ),
        # par() unpacks its result, so the next par() receives (filtered_dets, protos) as two args
        op.par(
            op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),  # re-pack, extract dets, rescale
            op.proto_to_mask(),  # (dets, protos) -> masks
        ),
    ).optimized()


def draw_segmentation(image: np.ndarray, detections: np.ndarray, masks: list, conf: float = 0.25) -> np.ndarray:
    """Draw instance segmentation results on the image.

    proto_to_mask() returns bbox-cropped masks at prototype resolution. Each mask must be resized to its detection's
    bounding box and placed at the correct image coordinates.
    """
    overlay = image.copy()  # one copy for mask blending
    h, w = image.shape[:2]

    for i, det in enumerate(detections):
        score = det[4]
        if score < conf:
            continue

        class_id = int(det[5])
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id)

        # Bounding box (clipped to image bounds)
        x0, y0, x1, y1 = map(int, det[:4])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        # Semi-transparent mask overlay: resize bbox-cropped mask and place it
        if i < len(masks):
            mask = masks[i]
            if mask.ndim == 2 and (x1 - x0) > 0 and (y1 - y0) > 0:
                mask_resized = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                overlay[y0:y1, x0:x1][mask_resized > 127] = color

        # Bounding box
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        # Label
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x0, y0 - th - 4), (x0 + tw, y0), color, -1)
        cv2.putText(image, label, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend mask overlay
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    return image


def main():
    """YOLO11 Instance Segmentation example."""
    parser = argparse.ArgumentParser(description="YOLO11 Instance Segmentation -- Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, default="0", help="Image, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    pipeline = build_pipeline(args.model, args.conf, args.iou)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_image = frames == 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, masks = pipeline(frame)
        annotated = draw_segmentation(frame, detections, masks, args.conf)

        cv2.imshow("YOLO11 Segmentation", annotated)
        if cv2.waitKey(0 if is_image else 1) & 0xFF in [ord("q"), ord("Q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
