# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import yolo_common as yolo
from coco_names import COCO_NAMES

args = yolo.parse_args("yolo26n.onnx", imgsz=True)
net = cv2.dnn.readNetFromONNX(args.model)  # OpenCV DNN reads no model metadata, so imgsz/task come from the CLI
imgsz = (args.imgsz, args.imgsz)

img = cv2.imread(args.source)
blob, pad = yolo.preprocess(img, imgsz, args.task or "detect")
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())
task = args.task or yolo.infer_task([o.shape for o in outputs], len(COCO_NAMES))
out = outputs[0]
if task in {"pose", "depth"}:  # single-class models
    names = {0: "person" if task == "pose" else "depth"}
elif task in {"detect", "segment"} and out.shape[1] < out.shape[2]:  # grid output: class count from the channels
    nc = out.shape[1] - (36 if task == "segment" else 4)
    names = COCO_NAMES if nc == len(COCO_NAMES) else {i: str(i) for i in range(nc)}
elif task in {"detect", "segment", "rtdetr"}:  # end-to-end outputs carry no class count, so COCO is assumed
    names = COCO_NAMES
else:  # classify, semantic and OBB have no COCO labels, so classes are shown as ids
    nc = out.shape[1] - 5 if task == "obb" else out.shape[1] if out.ndim == 4 or task == "classify" else 256
    names = {i: str(i) for i in range(nc)}
print(f"Model: {args.model} | task: {task} | classes: {len(names)}")
yolo.finish(img, yolo.postprocess(outputs, img, imgsz, task, names, args.conf, args.iou, pad), args)
