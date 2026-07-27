# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 Lane B at X scale: DINOv3-S+ baseline vs three UltraViT backbones on the same DEIMv2 neck.

These are RTDETR-family yamls, so the RTDETR facade resolves the task. Timing, rounds and CSV layout all come from
t4_bench_common, which reads the predictor's speed["inference"] exactly as profile_depth.py and ProfileModels do.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("/root/autodl-tmp/code")  # the one shared harness copy, resolved after the checkout
sys.path.insert(0, os.environ.get("ULTRA_CHECKOUT", "/root/autodl-tmp/code/ultravit-lane-b"))

from ultralytics import RTDETR  # noqa: E402  sys.path is set above

from working_dir.export_deimv2 import build_engine_fp16  # noqa: E402

from t4_bench_common import build_variant, run_benchmark  # noqa: E402

BASELINE = "dinov3splus"
YAMLS = {
    "dinov3splus": "deim_dinov3splus_sta_l6_xl.yaml",
    "ffnattn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml",
    "fastvitffn-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml",
    "repmixer-dinop5": "yolo26x-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml",
}
ENGINE_DIR = "/root/autodl-tmp/data/t4-laneb-x-recheck-engines"
CSV_PATH = "/root/autodl-tmp/data/t4_laneb_x_recheck_results.csv"


# Esat's builder with his --debug flag, so these engines match the ones his published numbers came from.
def pinned(onnx, engine):
    """Build the engine with attention softmax and norm internals pinned to fp32."""
    build_engine_fp16(onnx, engine, half=True, fp32_attn=True, debug=True)


variants = [build_variant(tag, y, ENGINE_DIR, model_cls=RTDETR, engine_builder=pinned) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
