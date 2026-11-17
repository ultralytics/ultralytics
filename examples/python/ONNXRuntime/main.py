# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast
import sys
from pathlib import Path

import cv2
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import yolo_common as yolo

args = yolo.parse_args("yolo26n.onnx")
available = ort.get_available_providers()
providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
session = ort.InferenceSession(args.model, providers=providers)

# Task, class names and input size are baked into the model metadata by Ultralytics export
meta = session.get_modelmeta().custom_metadata_map
names = ast.literal_eval(meta["names"])
imgsz = tuple(ast.literal_eval(meta["imgsz"]))
inp = session.get_inputs()[0]
task = args.task or (
    "rtdetr" if meta.get("head") == "RTDETRDecoder" else meta["task"]
)  # RT-DETR is stretched, not letterboxed
print(f"Model: {args.model} | task: {task} | classes: {len(names)}")

img = cv2.imread(args.source)
blob, pad = yolo.preprocess(img, imgsz, task, fp16="float16" in inp.type)
outputs = session.run(None, {inp.name: blob})
yolo.finish(img, yolo.postprocess(outputs, img, imgsz, task, names, args.conf, args.iou, pad), args)
