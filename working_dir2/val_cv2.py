#!/usr/bin/env python3
"""Validate an RF-DETR engine with cv2 INTER_LINEAR preprocessing (no antialias).

Monkey-patches applied at runtime — val.py is NOT modified:
  1. coco91 → coco80 class remapping  (RTDETRValidator.postprocess)
  2. stride=1 imgsz fix               (RTDETRValidator.__init__)

Usage:
  python working_dir2/val_rfdetr_cv2.py \\
      --model onnx_exports/rtdetr_rf-detr-nano.engine \\
      --data coco91.yaml --imgsz 384 --half

  python working_dir2/val_rfdetr_cv2.py \\
      --model onnx_exports/rf-detr-xlarge.engine \\
      --data coco91.yaml --imgsz 700 --half
"""

from __future__ import annotations

import argparse

from ultralytics.data.converter import coco91_to_coco80_class
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops
from ultralytics.utils.checks import check_imgsz as _check_imgsz

# ── Patch 1: coco91 → coco80 class remapping ─────────────────────────────────
def _patched_postprocess(self, preds):
    if not isinstance(preds, (list, tuple)):
        preds = [preds, None]
    bboxes, scores, labels = preds[0].split((4, 1, 1), dim=-1)
    bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
    labels = labels.squeeze(-1)
    if self.data.get("coco91", False):
        mapping = labels.new_tensor([x if x is not None else -1 for x in coco91_to_coco80_class()])
        labels = mapping[(labels.long() - 1).clamp(min=0)]
    return [
        {"bboxes": b, "conf": s.squeeze(-1), "cls": l}
        for b, s, l in zip(bboxes, scores, labels)
    ]

RTDETRValidator.postprocess = _patched_postprocess
print("[val_rfdetr_cv2] Patch 1: coco91 remapping applied")

# ── Patch 2: stride=1 imgsz fix ──────────────────────────────────────────────
# RF-DETR engines use non-stride-32 input sizes (e.g. xlarge=700). Ultralytics
# rounds imgsz up to a multiple of stride (32) in two places:
#   • RTDETRValidator.__init__          → self.args.imgsz (dataloader resize)
#   • BaseValidator.__call__ (~L187)    → local imgsz used for model.warmup()
# Both must use stride=1, else warmup builds a 704×704 tensor that a static
# 700×700 engine rejects. Patch __init__ for the dataloader and the validator
# module's check_imgsz for the warmup path.
import ultralytics.engine.validator as _validator_mod

_orig_validator_init = RTDETRValidator.__init__

def _patched_validator_init(self, *args, **kwargs):
    _orig_validator_init(self, *args, **kwargs)
    self.args.imgsz = _check_imgsz(self.args.imgsz, stride=1, max_dim=1)

RTDETRValidator.__init__ = _patched_validator_init

_orig_check_imgsz = _validator_mod.check_imgsz
_validator_mod.check_imgsz = lambda imgsz, stride=32, **kw: _orig_check_imgsz(imgsz, stride=1, **kw)

print("[val_rfdetr_cv2] Patch 2: stride=1 imgsz fix applied (init + warmup)")
print("[val_rfdetr_cv2] Preprocessing: cv2.INTER_LINEAR (no antialias)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_engine_imgsz(engine_path: str) -> int:
    """Read the input spatial resolution from a TensorRT engine file."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    input_name = engine.get_tensor_name(0)
    shape = tuple(engine.get_tensor_shape(input_name))
    return shape[-1]  # (B, C, H, W) → W, assumes square


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",   required=True, help="Path to .engine or .onnx file.")
    p.add_argument("--data",    default="coco91.yaml")
    p.add_argument("--imgsz",   type=int, default=None,
                   help="Input resolution. Auto-read from engine if omitted.")
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device",  default="0")
    p.add_argument("--half",    action="store_true", help="FP16 inference.")
    p.add_argument("--project", default=None)
    p.add_argument("--name",    default=None)
    return p.parse_args()


def main():
    args = parse_args()

    imgsz = args.imgsz
    if imgsz is None and args.model.endswith(".engine"):
        imgsz = get_engine_imgsz(args.model)
        print(f"[val_rfdetr_cv2] Auto-detected engine resolution: {imgsz}x{imgsz}")

    from ultralytics import RTDETR, settings
    settings.update({"runs_dir": "/home/esat/git/rfdetr/runs"})

    val_kwargs = dict(
        data=args.data,
        imgsz=imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        half=args.half,
        rtdetr_input_normalize=True,
    )
    if args.project:
        val_kwargs["project"] = args.project
    if args.name:
        val_kwargs["name"] = args.name

    model = RTDETR(args.model)
    model.val(**val_kwargs)


if __name__ == "__main__":
    main()
