#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLOv8 model to a separate-head ONNX graph for RKNN.

Adapted from the author's `export_6out_onnx.py`, extended with a `--heads 9`
option for the 9-output layout.

Ultralytics' default ONNX export decodes the boxes and concatenates everything
into a single `[1, 4+nc, 8400]` tensor. For RKNN INT8 quantization it is often
better to keep the *separate* per-scale outputs:

  6-output (recommended):  [box, Sigmoid(cls)] x 3 scales
  9-output:                [box, cls_logit, score_sum] x 3 scales

where `box` are raw DFL logits [1, 4*reg_max, h, w]. Keeping `Sigmoid(cls)`
fused in the graph (no `score_sum`) lets rknn-toolkit2 fuse the sigmoid into the
convolution during quantization, preserving the INT8 confidence scale
(see examples/yolov8 in https://github.com/airockchip/rknn_model_zoo).

The matching post-processing is in `rknn_inference.py` (layout auto-detected
from the output count). Convert the exported ONNX to RKNN with rknn-toolkit2:

    python convert.py yolov8n_6out.onnx rk3588 i8 yolov8n_6out.rknn

Usage:
    python export_onnx.py yolov8n.pt yolov8n_6out.onnx            # 6-output
    python export_onnx.py yolov8n.pt yolov8n_9out.onnx --heads 9  # 9-output
"""

import argparse

import torch
from torch import nn


def export_onnx(pt_path, onnx_path, imgsz=640, opset=17, heads=6):
    from ultralytics import YOLO

    model = YOLO(pt_path)
    detect_model = model.model
    detect_model.eval()

    detect = detect_model.model[-1]  # Detect head
    backbone = detect_model.model[:-1]

    class YOLOv8SeparateHeads(nn.Module):
        """Rerun the backbone and output [box, (score_sum,) Sigmoid(cls)] per branch."""

        def __init__(self, backbone, detect, heads):
            super().__init__()
            self.backbone = backbone
            self.detect = detect
            self.heads = heads

        def forward(self, x):
            y = []
            detect_inputs = []
            for layer in self.backbone:
                if layer.f != -1:
                    if isinstance(layer.f, int):
                        x = y[layer.f] if layer.f >= 0 else x
                    elif isinstance(layer.f, (list, tuple)):
                        x = [y[j] if j >= 0 else x for j in layer.f]
                x = layer(x)
                if hasattr(layer, "i") and layer.i in self.detect.f:
                    detect_inputs.append(x)
                y.append(x)

            outputs = []
            for i in range(self.detect.nl):
                feat = detect_inputs[i]
                box_out = self.detect.cv2[i](feat)  # [1, 4*reg_max, h, w]
                cls_out = self.detect.cv3[i](feat)  # [1, nc, h, w]
                cls_prob = torch.sigmoid(cls_out)
                if self.heads == 9:
                    score_sum = torch.clip(cls_prob.sum(dim=1, keepdim=True), 0, 1)
                    outputs.extend([box_out, cls_out, score_sum])
                else:
                    outputs.extend([box_out, cls_prob])
            return tuple(outputs)

    wrapper = YOLOv8SeparateHeads(backbone, detect, heads)
    wrapper.eval()

    n_out = heads  # total outputs (6 -> 2 per branch, 9 -> 3 per branch)
    dummy = torch.randn(1, 3, imgsz, imgsz)
    output_names = [f"output{i}" for i in range(n_out)]
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export succeeded: {onnx_path} ({heads}-output)")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to a separate-head ONNX")
    parser.add_argument("model", help="input .pt path")
    parser.add_argument("output", help="output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--heads", type=int, choices=[6, 9], default=6, help="outputs per branch x 3 scales (6 or 9)")
    args = parser.parse_args()
    export_onnx(args.model, args.output, args.imgsz, args.opset, args.heads)
