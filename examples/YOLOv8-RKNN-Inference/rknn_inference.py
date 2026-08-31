# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLOv8 RKNN inference with rknn-toolkit2.

Supports the *separate-head* model layout produced by `export_onnx.py`:
  - 6-output: [box, Sigmoid(cls)] x scales
  - 9-output: [box, cls_logit, score_sum] x scales
The layout and the number of scales (3 or 4) are detected automatically from
the model's output count.

This script merges the author's `detect_rknn.py` (image) and
`detect_video_rknn.py` (video) examples into one entry point, with COCO labels
and gray-114 letterbox (matching the rknn_model_zoo calibration).

Requires a `.rknn` model, which is converted from the separate-head ONNX with
Rockchip's rknn-toolkit2 (not pip-installable, see the README for the download
and conversion steps).

Usage:
    python rknn_inference.py --model yolov8n_6out.rknn --image bus.jpg [img2.jpg ...]
    python rknn_inference.py --model yolov8n_6out.rknn --video demo.mp4 --out result.mp4
"""

import argparse
import os

import cv2
import numpy as np

IMG_SIZE = (640, 640)  # (w, h)
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
REG_MAX = 16
PAD_COLOR = 114  # gray-114 letterbox, matching the rknn_model_zoo calibration

# Default to the COCO label set so the example works out of the box with yolov8n.
COCO_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def sigmoid(x):
    """Return the element-wise logistic sigmoid of ``x``."""
    return 1.0 / (1.0 + np.exp(-x))


def letter_box(im, new_shape, pad_color=PAD_COLOR):
    """Resize with aspect ratio preserved and pad to new_shape."""
    h, w = im.shape[:2]
    r = min(new_shape[1] / w, new_shape[0] / h)
    new_unpad = (round(w * r), round(h * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_color, pad_color, pad_color))
    return im, r, (dw, dh)


def dfl_decode(box):
    """Box [1, 4*reg_max, h, w] -> [1, 4, h, w] (distribution focal loss)."""
    n, _c, h, w = box.shape
    prob = box.reshape(n, 4, REG_MAX, h, w)
    prob = np.exp(prob - prob.max(axis=2, keepdims=True))
    prob = prob / prob.sum(axis=2, keepdims=True)
    acc = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1, 1)
    return (prob * acc).sum(axis=2)


def post_process(outputs, imgsz, conf=OBJ_THRESH, nms=NMS_THRESH):
    """Decode the separate-head outputs into (xyxy, scores, classes).

    outputs: fp32 tensors, either [box, cls, score_sum] x scales (9-output, cls
             is logit) or [box, cls] x scales (6-output, cls is already Sigmoid).
    imgsz: square input size (w, h); used for stride decoding.
    Returns boxes in the letterboxed input space, or (None, None, None).
    """
    # 6-output is [box, Sigmoid(cls)] x scales; 9-output is [box, cls_logit, score_sum]
    # x scales. Infer the outputs-per-scale from the total count (3 or 4 scales).
    if len(outputs) % 3 == 0 and len(outputs) // 3 in (3, 4):
        per, n_scales = 3, len(outputs) // 3  # 9-output: cls is logit
    else:
        per, n_scales = 2, len(outputs) // 2  # 6-output: cls is already Sigmoid
    all_boxes, all_scores, all_classes = [], [], []
    for i in range(n_scales):
        box = dfl_decode(outputs[i * per])  # [1, 4, h, w]
        cls = outputs[i * per + 1]
        if per == 3:
            cls = sigmoid(cls)  # 9-output: cls is logit
        _, _, gh, gw = box.shape
        stride = imgsz[0] // gw
        jx, jy = np.meshgrid(np.arange(gw), np.arange(gh))
        x1 = (-box[0, 0] + jx + 0.5) * stride
        y1 = (-box[0, 1] + jy + 0.5) * stride
        x2 = (box[0, 2] + jx + 0.5) * stride
        y2 = (box[0, 3] + jy + 0.5) * stride

        cls_max = cls[0].max(axis=0)  # [h, w]
        cls_id = cls[0].argmax(axis=0)
        mask = cls_max > conf
        if mask.any():
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1)[mask])
            all_scores.append(cls_max[mask])
            all_classes.append(cls_id[mask])

    if not all_boxes:
        return None, None, None
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    # Class-wise NMS
    nboxes, nscores, nclasses = [], [], []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        b, s = boxes[idx], scores[idx]
        x, y = b[:, 0], b[:, 1]
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        areas = w * h
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            k = order[0]
            keep.append(k)
            xx1 = np.maximum(x[k], x[order[1:]])
            yy1 = np.maximum(y[k], y[order[1:]])
            xx2 = np.minimum(x[k] + w[k], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[k] + h[k], y[order[1:]] + h[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1e-5) * np.maximum(0.0, yy2 - yy1 + 1e-5)
            ovr = inter / (areas[k] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= nms)[0] + 1]
        nboxes.append(b[keep])
        nscores.append(s[keep])
        nclasses.append(classes[idx][keep])
    return np.concatenate(nboxes), np.concatenate(nscores), np.concatenate(nclasses)


def unmap_box(boxes, ratio, pad):
    """Map letterboxed coordinates back to the original image."""
    dw, dh = pad
    return (boxes - np.array([dw, dh, dw, dh])) / ratio


def draw(image, boxes, scores, classes, names):
    """Draw detection boxes and labels on ``image`` in place."""
    for b, s, c in zip(boxes, scores, classes):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in np.clip(b, 0, [w, h, w, h])]
        label = f"{names[int(c)] if int(c) < len(names) else str(int(c))} {s:.2f}"
        print(f"  {label} @ ({x1} {y1} {x2} {y2})")
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def inference(rknn, img_src, names, conf, nms, imgsz):
    """Run one frame; returns the annotated BGR image."""
    img_lb, ratio, pad = letter_box(img_src.copy(), new_shape=imgsz)
    input_data = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    input_data = input_data[np.newaxis, ...]  # [1, h, w, 3] static-batch input
    outputs = rknn.inference(inputs=[input_data])
    outputs = [o.astype(np.float32) for o in outputs]
    boxes, scores, classes = post_process(outputs, imgsz, conf, nms)
    img_draw = img_src.copy()
    if boxes is not None:
        draw(img_draw, unmap_box(boxes, ratio, pad), scores, classes, names)
    else:
        print("  no detections")
    return img_draw


def load_rknn(model_path, target):
    """Load a .rknn model and init the runtime; returns the RKNN object."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(model_path) == 0, f"failed to load model: {model_path}"
    if target:
        assert rknn.init_runtime(target=target) == 0, f"NPU init failed for {target}"
    else:
        assert rknn.init_runtime() == 0, "runtime init failed (simulator)"
    return rknn


def run_image(args, rknn, names, imgsz):
    """Run inference on each image in ``args.image`` and save the annotated results."""
    for img_path in args.image:
        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f"cannot read {img_path}")
            continue
        print(f"\n[{os.path.basename(img_path)}] {os.path.basename(args.model)}:")
        img_draw = inference(rknn, img_src, names, args.conf, args.nms, imgsz)
        out = args.out_dir or os.path.dirname(img_path)
        out_path = os.path.join(out, f"{os.path.splitext(os.path.basename(img_path))[0]}_rknn.jpg")
        cv2.imwrite(out_path, img_draw)
        print(f"  result saved: {out_path}")


def run_video(args, rknn, names, imgsz):
    """Run inference on a video and write the annotated result to ``args.out``."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = args.out or os.path.join(
        os.path.dirname(args.video), f"{os.path.splitext(os.path.basename(args.video))[0]}_rknn.avi"
    )
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img_draw = inference(rknn, frame, names, args.conf, args.nms, imgsz)
        writer.write(img_draw)
        frame_id += 1
        if frame_id % 30 == 1:
            print(f"  frame {frame_id}")
    cap.release()
    writer.release()
    print(f"video result saved: {out}")


def main():
    """Parse command-line arguments and run image/video inference."""
    parser = argparse.ArgumentParser(description="YOLOv8 RKNN inference")
    parser.add_argument("--model", required=True, help=".rknn model path")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="square input size, must match the size used at export/conversion",
    )
    parser.add_argument("--image", nargs="+", help="input image path(s)")
    parser.add_argument("--video", help="input video path")
    parser.add_argument("--target", default=None, help="NPU target (e.g. rk3588); omit to run the PC simulator")
    parser.add_argument("--conf", type=float, default=OBJ_THRESH)
    parser.add_argument("--nms", type=float, default=NMS_THRESH)
    parser.add_argument("--out_dir", default=None, help="image result directory")
    parser.add_argument("--out", default=None, help="video result path")
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("provide --image or --video")

    imgsz = (args.imgsz, args.imgsz)
    names = list(COCO_NAMES)
    rknn = load_rknn(args.model, args.target)
    try:
        if args.image:
            run_image(args, rknn, names, imgsz)
        if args.video:
            run_video(args, rknn, names, imgsz)
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
