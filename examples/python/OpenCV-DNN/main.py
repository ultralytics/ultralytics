# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import yolo_common as yolo

args = yolo.parse_args("yolo26n.onnx", imgsz=True)
net = cv2.dnn.readNetFromONNX(args.model)  # OpenCV DNN reads no model metadata, so imgsz/task come from the CLI
imgsz = (args.imgsz, args.imgsz)

img = cv2.imread(args.source)
blob, pad = yolo.preprocess(img, imgsz, args.task or "detect")
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())
out = outputs[0]
task = args.task
grid = out.ndim == 3 and out.shape[1] < out.shape[2]  # [1, channels, anchors], else end-to-end [1, 300, attrs]
if not task:  # infer detect, segment, pose or end-to-end OBB from the output shapes; other tasks need --task
    if len(outputs) == 2:
        task = "segment"  # (detections, protos)
    elif grid:
        task = "pose" if out.shape[1] == 56 else "detect"  # 56 = COCO pose; other layouts are custom detectors
    else:
        task = {6: "detect", 7: "obb"}.get(out.shape[2], "pose")
if task == "pose":
    names = {0: "person"}
elif grid and task in {"detect", "segment", "obb"}:  # grid outputs carry the class count in their channels
    nc = out.shape[1] - {"detect": 4, "segment": 36, "obb": 5}[task]
    names = yolo.COCO_NAMES if nc == 80 and task != "obb" else {i: str(i) for i in range(nc)}
else:  # end-to-end and RT-DETR outputs carry no class count, so COCO is assumed; other tasks show class ids
    names = yolo.COCO_NAMES if task in {"detect", "segment", "rtdetr"} else {}
print(f"Model: {args.model} | task: {task}")
yolo.finish(img, yolo.postprocess(outputs, img, imgsz, task, names, args.conf, args.iou, pad), args)
