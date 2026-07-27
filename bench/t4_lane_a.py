# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 latency for every Lane A architecture that carries a registered number, remeasured in one session.

Lane A is the YOLO26 conv-head detector, so the YOLO facade resolves the task and the stock Ultralytics engine
export applies. The fp32 attention pin Lane B needs has nothing to pin here.

Nine of these ten cover every phase2 run in the orc db, one per distinct (yaml, scale). The registered numbers they
replace were produced across many sessions under a timer and a weight state that are both now known to be wrong, and
two architectures carry two conflicting values each (2.6081 against 2.6641, 10.361 against 10.4053). Everything below
is measured in one run against one baseline so the deltas are paired. The tenth, depth-matched dinop5, has no
registered number and is the open row in the Lane A table.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry
carries an explicit size letter.

The baseline is conv at X, so `delta_vs_base_pct` is an architecture comparison for the six X rows only. The three
smaller rows are in the same run to share its thermal state, and their delta against an X baseline is scale, not
architecture. Compare those against each other.
"""

import sys
from pathlib import Path

# Repo root first, so the harness and ultralytics both come from this checkout. sys.path[0] is bench/, not the root,
# so without this the import resolves to whatever unversioned copy happens to sit on the shared path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t4_bench_common import build_variant, run_benchmark

BASELINE = "conv-x"
YAMLS = {
    "conv-s": "yolo26s.yaml",
    "uvit-s-attn2-s480": "yolo26s-ultravit-repmixer-fastvitffn-attn2-s480.yaml",
    "uvit-l-attn2": "yolo26l-ultravit-repmixer-fastvitffn-attn2.yaml",
    "conv-x": "yolo26x.yaml",
    "uvit-x-plain": "yolo26x-ultravit.yaml",
    "uvit-x-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5.yaml",
    "uvit-x-dinop5-depthmatched": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-depthmatched.yaml",
    "uvit-x-attn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2.yaml",
    "uvit-x-attn2-dinoreg": "yolo26x-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml",
    "uvit-x-attn2-p4pooled": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml",
    "uvit-x-attn2-p4win": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4win.yaml",
}
ENGINE_DIR = "/root/autodl-tmp/data/t4-lane-a-engines"
CSV_PATH = "/root/autodl-tmp/data/t4_lane_a_results.csv"

variants = [build_variant(tag, y, ENGINE_DIR) for tag, y in YAMLS.items()]
run_benchmark(variants, BASELINE, CSV_PATH)
