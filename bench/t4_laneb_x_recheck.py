# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 Lane B at X scale: DINOv3-S+ baseline vs five UltraViT backbones on the same DEIMv2 neck.

These are RTDETR-family yamls, so the RTDETR facade resolves the task. Timing, rounds and CSV layout all come from
t4_bench_common, which reads the predictor's speed["inference"] exactly as profile_depth.py and ProfileModels do.
"""

import os
import sys

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from the same commit. Ruff permits a
# sys.path mutation ahead of imports, so this needs no E402 suppression, which the formatter strips anyway.
sys.path.insert(0, os.environ.get("ULTRA_CHECKOUT", "/root/autodl-tmp/code/ultravit-lane-b"))

from t4_bench_common import build_variant, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR

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
CSV_PATH = "/root/autodl-tmp/data/t4_laneb_x_protocol0727.csv"  # not the _recheck_results.csv, that holds an older run

variants = [build_variant(tag, y, ENGINE_DIR, model_cls=RTDETR, engine_builder=pinned_fp32_attn) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
