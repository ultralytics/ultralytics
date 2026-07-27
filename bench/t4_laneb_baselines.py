# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 Lane B baselines: the six yolo27-detr scales every Lane B detector yaml is now measured against.

These come from detr_decoder_clean2 and are the reference arms for n through xxl. They are RTDETR-family yamls, so
the RTDETR facade resolves the task, and they take the same pinned fp32-attention engine build as the rest of Lane B.

The baseline is x, so `delta_vs_base_pct` is an architecture comparison for nothing here. Every row is a different
scale, and the family changes with it: n and s carry RTDETRDecoderEfficient on a CSP trunk, m and l swap in
DeimDecoder, x and xxl swap the trunk for a plain ViT. Read this table as the scale ladder each variant row is
compared against, one baseline per scale, not as a backbone trend.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry
carries an explicit size letter. The two-letter xxl needs the widened scale regex in tasks.py to resolve at all.
"""

import os
import sys

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from the same commit. Ruff permits a
# sys.path mutation ahead of imports, so this needs no E402 suppression, which the formatter strips anyway.
sys.path.insert(0, os.environ.get("ULTRA_CHECKOUT", "/root/autodl-tmp/code/ultravit-lane-b"))

from t4_bench_common import build_variant, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR

BASELINE = "yolo27x"
YAMLS = {
    "yolo27n": "yolo27n-detr.yaml",
    "yolo27s": "yolo27s-detr.yaml",
    "yolo27m": "yolo27m-deim-detr.yaml",
    "yolo27l": "yolo27l-deim-detr.yaml",
    "yolo27x": "yolo27x-vit-detr.yaml",
    "yolo27xxl": "yolo27xxl-vit-detr.yaml",
}
ENGINE_DIR = "/root/autodl-tmp/data/t4-laneb-baseline-engines"
CSV_PATH = "/root/autodl-tmp/data/t4_laneb_baselines_protocol0727.csv"

variants = [build_variant(tag, y, ENGINE_DIR, model_cls=RTDETR, engine_builder=pinned_fp32_attn) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
