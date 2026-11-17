# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Backend-agnostic pre/post-processing for the Ultralytics YOLO Python examples.

Only NumPy and OpenCV are needed. A backend just runs the model: ``outputs = run(preprocess(...))``. Everything else
(NMS, mask, pose, OBB, classification and semantic decoding, drawing, printing) lives here.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

SKELETON = [  # COCO 17-keypoint limbs
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]  # fmt: skip


def parse_args(model: str, imgsz: bool = False) -> argparse.Namespace:
    """Parse the command-line arguments shared by every example."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=model, help="path to the exported model")
    p.add_argument("--source", default="bus.jpg", help="input image")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (grid models only)")
    p.add_argument("--out", default="result.jpg", help="output image path")
    p.add_argument("--task", help="override the task: detect, segment, semantic, depth, classify, pose, obb or rtdetr")
    p.add_argument("--show", action="store_true", help="also open a display window")
    if imgsz:
        p.add_argument(
            "--imgsz",
            type=int,
            default=640,
            help="model input size (768 depth, 224 classify, 1024 DOTA OBB and Cityscapes semantic)",
        )
    return p.parse_args()


def infer_task(shapes: list[tuple], nc: int) -> str:
    """Infer the task from the output shapes and class count for backends without task metadata."""
    if any(len(s) == 4 for s in shapes):
        return "semantic" if len(shapes) == 1 else "segment"  # [1, nc, H, W] or (detections, protos)
    s = shapes[0]
    if len(s) == 2:
        return "classify"
    if s[1] > s[2]:  # end-to-end [1, 300, attrs]
        return {6: "detect", 7: "obb"}.get(s[2], "pose")
    if s[1] == 4 + nc:  # grid [1, channels, anchors]
        return "detect"
    return "obb" if s[1] == 5 + nc else "pose"


def preprocess(img: np.ndarray, imgsz: tuple[int, int], task: str, fp16: bool = False):
    """Return a normalized RGB [1, 3, H, W] blob and the (gain, left, top) needed to map outputs back.

    Detection tasks are letterboxed, RT-DETR and depth are stretched, and classification is resized and center-cropped.
    """
    h, w = imgsz
    gain, left, top = 1.0, 0, 0
    if task == "classify":
        r = max(h, w) / min(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * r), round(img.shape[0] * r)), interpolation=cv2.INTER_AREA)
        y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
        img = img[y : y + h, x : x + w]
    elif task in {"rtdetr", "depth"}:
        img = cv2.resize(img, (w, h))
    else:
        gain = min(h / img.shape[0], w / img.shape[1])
        nw, nh = round(img.shape[1] * gain), round(img.shape[0] * gain)
        left, top = round((w - nw) / 2 - 0.1), round((h - nh) / 2 - 0.1)
        img = cv2.resize(img, (nw, nh))
        img = cv2.copyMakeBorder(img, top, h - nh - top, left, w - nw - left, cv2.BORDER_CONSTANT, value=(114,) * 3)
    blob = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None]) / 255
    return blob.astype(np.float16 if fp16 else np.float32), (gain, left, top)


def _nms(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou: float, rotated: bool) -> list[int]:
    """Class-aware NMS on xyxy boxes, or on xywh + radians for rotated boxes."""
    if not rotated:
        xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
        return list(np.ravel(cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), 0.0, iou)))
    keep = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        rects = [((b[0], b[1]), (b[2], b[3]), float(np.degrees(b[4]))) for b in boxes[idx]]
        keep += list(idx[np.ravel(cv2.dnn.NMSBoxesRotated(rects, conf[idx].tolist(), 0.0, iou))])
    return keep


def _masks(
    protos: np.ndarray, coefs: np.ndarray, boxes: np.ndarray, shape: tuple, imgsz: tuple, pad: tuple
) -> np.ndarray:
    """Decode instance masks at original-image size, cropped to their boxes."""
    h, w = shape
    left, top = pad
    m = 1 / (1 + np.exp(-coefs @ protos.reshape(len(protos), -1)))
    out = np.zeros((len(coefs), h, w), bool)
    for i, (mask, (x1, y1, x2, y2)) in enumerate(zip(m.reshape(-1, *protos.shape[1:]), boxes.astype(int))):
        mask = cv2.resize(mask, imgsz[::-1])
        mask = mask[top : imgsz[0] - top, left : imgsz[1] - left] if top or left else mask
        out[i, max(y1, 0) : y2, max(x1, 0) : x2] = (cv2.resize(mask, (w, h)) > 0.5)[max(y1, 0) : y2, max(x1, 0) : x2]
    return out


def postprocess(outputs: list[np.ndarray], img: np.ndarray, imgsz: tuple[int, int], task: str, names: dict, conf: float, iou: float, pad: tuple) -> dict:  # fmt: skip
    """Decode raw model outputs of any task into a result dict in original-image coordinates."""
    gain, left, top = pad
    r = {"task": task, "names": names, "shape": img.shape[:2]}
    out = outputs[0].astype(np.float32)
    if task == "classify":
        return {**r, "probs": out[0]}
    if task == "depth":  # [1, 1, H, W] or [1, H, W] metric depth, resized straight to the image size
        return {**r, "depth": cv2.resize(out.reshape(out.shape[-2:]), img.shape[1::-1])}
    if task == "semantic":  # [1, nc, H, W] logits or [1, H, W] class map
        x = out[0].transpose(1, 2, 0) if out.ndim == 4 else out[0][..., None]
        x = cv2.resize(x, imgsz[::-1], interpolation=cv2.INTER_LINEAR if out.ndim == 4 else cv2.INTER_NEAREST)
        x = x[..., None] if x.ndim == 2 else x
        cls = x.argmax(-1) if x.shape[-1] > 1 else (x[..., 0] if out.ndim == 3 else x[..., 0] > 0)
        cls = cls[top : imgsz[0] - top, left : imgsz[1] - left].astype(np.uint8)
        return {**r, "semantic": cv2.resize(cls, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)}

    out = out[0]
    nc = len(names)
    rotated = task == "obb"
    if task == "rtdetr":  # [300, 6] rows: normalized xywh, conf, cls (no NMS needed)
        x = out[out[:, 4] > conf]
        xywh = x[:, :4] * np.tile(img.shape[1::-1], 2)
        boxes = np.c_[xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2]
        return {**r, "boxes": boxes, "conf": x[:, 4], "cls": x[:, 5].astype(int)}
    if out.shape[0] > out.shape[1]:  # end-to-end rows: xyxy (xywh for obb), conf, cls, extra
        x = out[out[:, 4] > conf]
        boxes, scores, cls, extra = x[:, :4], x[:, 4], x[:, 5].astype(int), x[:, 6:]
    else:  # grid rows: xywh, class scores, extra
        x = out.T
        s = x[:, 4 : 4 + nc]
        x = x[s.max(1) > conf]
        s = s[s.max(1) > conf]
        boxes, scores, cls, extra = x[:, :4], s.max(1), s.argmax(1), x[:, 4 + nc :]
        if not rotated:
            boxes = np.c_[boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2]
    if rotated:
        boxes = np.c_[boxes, extra[:, 0]]  # xywh + angle in radians
    if out.shape[0] < out.shape[1] and len(boxes):  # grid models need NMS
        keep = _nms(boxes, scores, cls, iou, rotated)
        boxes, scores, cls, extra = boxes[keep], scores[keep], cls[keep], extra[keep]
    if rotated:
        boxes[:, :4] = (boxes[:, :4] - [left, top, 0, 0]) / gain
    else:
        boxes = (boxes - [left, top] * 2) / gain
    r |= {"boxes": boxes, "conf": scores, "cls": cls}
    if task == "segment":
        r["masks"] = _masks(outputs[1][0], extra, boxes, img.shape[:2], imgsz, (left, top))
    elif task == "pose":
        kpts = extra.reshape(len(extra), -1, 3).copy()
        kpts[..., :2] = (kpts[..., :2] - [left, top]) / gain
        r["kpts"] = kpts
    return r


def _color(i: int) -> tuple:
    """Deterministic BGR color for class ``i``."""
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 47) % 180, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw(img: np.ndarray, r: dict) -> np.ndarray:
    """Return a copy of ``img`` annotated with the decoded result."""
    img, names = img.copy(), r["names"]
    if "probs" in r:
        for k, i in enumerate(np.argsort(-r["probs"])[:5]):
            cv2.putText(img, f"{names[i]} {r['probs'][i]:.2f}", (10, 30 + 28 * k), 0, 0.9, _color(int(i)), 2)
        return img
    if "depth" in r:
        d = r["depth"]
        d8 = ((d - d.min()) / max(float(d.max() - d.min()), 1e-6) * 255).astype(np.uint8)
        return cv2.addWeighted(img, 0.4, cv2.applyColorMap(255 - d8, cv2.COLORMAP_TURBO), 0.6, 0)
    if "semantic" in r:
        palette = np.array([_color(i) for i in range(256)], np.uint8)
        return cv2.addWeighted(img, 0.5, palette[r["semantic"]], 0.5, 0)
    for i, (box, c, k) in enumerate(zip(r["boxes"], r["conf"], r["cls"])):
        color, label = _color(int(k)), f"{names[int(k)]} {c:.2f}"
        if "masks" in r:
            img[r["masks"][i]] = (0.5 * img[r["masks"][i]] + 0.5 * np.array(color)).astype(np.uint8)
        if r["task"] == "obb":
            pts = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), np.degrees(box[4]))).astype(int)
            cv2.polylines(img, [pts], True, color, 2)
            x, y = pts[pts[:, 1].argmin()]
        else:
            x, y = int(box[0]), int(box[1])
            cv2.rectangle(img, (x, y), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, label, (x, max(y - 5, 12)), 0, 0.6, color, 2)
        if "kpts" in r:
            kp = r["kpts"][i]
            if len(kp) == 17:
                for a, b in SKELETON:
                    cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 2)
            for px, py, _ in kp:
                cv2.circle(img, (int(px), int(py)), 3, (0, 255, 255), -1)
    return img


def finish(img: np.ndarray, r: dict, args: argparse.Namespace) -> None:
    """Print the detections, save the annotated image to ``--out`` and optionally show it."""
    names = r["names"]
    if "probs" in r:
        print(", ".join(f"{names[i]} {r['probs'][i]:.2f}" for i in np.argsort(-r["probs"])[:5]))
    elif "depth" in r:
        print(f"depth {r['depth'].min():.2f}-{r['depth'].max():.2f}m")
    elif "semantic" in r:
        print("classes:", ", ".join(names[int(i)] for i in np.unique(r["semantic"]) if int(i) in names))
    else:
        for box, c, k in zip(r["boxes"], r["conf"], r["cls"]):
            print(f"{names[int(k)]} {c:.2f} {np.round(box[:4]).astype(int).tolist()}")
        print(f"{len(r['boxes'])} detections")
    out = draw(img, r)
    cv2.imwrite(args.out, out)
    print(f"Result image written to {args.out}")
    if args.show:
        cv2.imshow("YOLO", out)
        cv2.waitKey(0)
