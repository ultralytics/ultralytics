"""Rebuild the four trained engines from their existing ONNX under a different TensorRT version.

The ONNX files are reused byte for byte and Esat's own build_engine_fp16 does the work, so the only
variable between this engine set and the TRT 10.16 set is the TensorRT builder version itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/code/ultravit-lane-b")

import tensorrt as trt

from working_dir.export_deimv2 import build_engine_fp16

SRC = Path("/root/autodl-tmp/esat-models")
DST = Path("/root/autodl-tmp/data/trt1011-engines")
DST.mkdir(parents=True, exist_ok=True)
S = "_op17_nosim_norope_imgsz640_fp32attn_debug_fp16"

STEMS = {
    "dinov3splus": f"rtdetr_dinov3sp_deim_deimv2Neck_coco{S}",
    "ffnattn2": f"rtdetr_ultravitX_fastVIT_attnv2_deim_deimv2Neck_fatih_coco{S}",
    "p4pooled": f"rtdetr_ultravitX_fastVIT_attnv2p4pool_deim_deimv2Neck_fatih_coco{S}",
    "ultravit-attn2": f"rtdetr_ultravitX_attnv2_deim_deimv2Neck_fatih_coco{S}",
}

print(f"=== building with TensorRT {trt.__version__}", flush=True)
for name, stem in STEMS.items():
    onnx, engine = SRC / f"{stem}.onnx", DST / f"{stem}.engine"
    if engine.exists():
        print(f"  skip {name}, exists", flush=True)
        continue
    print(f"\n--- {name}  {onnx.name}", flush=True)
    build_engine_fp16(onnx, engine, half=True, fp32_attn=True, debug=True)
print("\n=== build done", flush=True)
