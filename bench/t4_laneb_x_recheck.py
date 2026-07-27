# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 Lane B at X scale: DINOv3-S+ baseline vs five UltraViT backbones on the same DEIMv2 neck.

These are RTDETR-family yamls, so the RTDETR facade resolves the task. Timing, rounds and CSV layout all come from
t4_bench_common, which reads the predictor's speed["inference"] exactly as profile_depth.py and ProfileModels do.
"""

# ruff: noqa: E402  every import below has to follow the sys.path setup

import os
import sys
import warnings

warnings.filterwarnings("ignore")
# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from the same commit.
sys.path.insert(0, os.environ.get("ULTRA_CHECKOUT", "/root/autodl-tmp/code/ultravit-lane-b"))

from t4_bench_common import build_variant, run_benchmark
from ultralytics import RTDETR
from working_dir.export_deimv2 import build_engine_fp16

BASELINE = "dinov3splus"
YAMLS = {
    "dinov3splus": "deim_dinov3splus_sta_l6_xl.yaml",
    "ffnattn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml",
    "fastvitffn-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml",
    "repmixer-dinop5": "yolo26x-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml",
    "fastvitffn-dinop5-depthmatched": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-depthmatched-deim_mal_deimv2Neck.yaml",
    "ffnattn2-p4win": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4win-deim_mal_deimv2Neck.yaml",
}
ENGINE_DIR = "/root/autodl-tmp/data/t4-laneb-x-recheck-engines"
CSV_PATH = "/root/autodl-tmp/data/t4_laneb_x_recheck_results.csv"


# Esat's builder with his --debug flag, so these engines match the ones his published numbers came from.
def pinned(onnx, engine):
    """Build the engine with attention softmax and norm internals pinned to fp32."""
    build_engine_fp16(onnx, engine, half=True, fp32_attn=True, debug=True)


variants = [build_variant(tag, y, ENGINE_DIR, model_cls=RTDETR, engine_builder=pinned) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
