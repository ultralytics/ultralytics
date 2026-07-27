# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 latency for every Lane A architecture that carries a registered number, remeasured in one session.

Lane A is the YOLO26 conv-head detector, so the YOLO facade resolves the task and the stock Ultralytics engine
export applies. The fp32 attention pin Lane B needs has nothing to pin here.

The registered numbers these replace were produced across many sessions under a timer and a weight state that are
both now known to be wrong, and two architectures carry two conflicting values each (2.6081 against 2.6641, 10.361
against 10.4053). Everything below is measured in one run against one baseline so the deltas are paired.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry
carries an explicit size letter.
"""

import sys

sys.path.append("/root/autodl-tmp/code")  # shared harness, appended so this checkout's ultralytics wins

from t4_bench_common import build_variant, run_benchmark

BASELINE = "conv-x"
YAMLS = {
    "conv-s": "yolo26s.yaml",
    "uvit-s-attn2-s480": "yolo26s-ultravit-repmixer-fastvitffn-attn2-s480.yaml",
    "uvit-l-attn2": "yolo26l-ultravit-repmixer-fastvitffn-attn2.yaml",
    "conv-x": "yolo26x.yaml",
    "uvit-x-plain": "yolo26x-ultravit.yaml",
    "uvit-x-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5.yaml",
    "uvit-x-attn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2.yaml",
    "uvit-x-attn2-dinoreg": "yolo26x-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml",
    "uvit-x-attn2-p4pooled": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml",
}
ENGINE_DIR = "/root/autodl-tmp/data/t4-lane-a-engines"
CSV_PATH = "/root/autodl-tmp/data/t4_lane_a_results.csv"

variants = [build_variant(tag, y, ENGINE_DIR) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
