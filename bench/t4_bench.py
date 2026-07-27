# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 latency for one named suite of architectures, measured in a single paired session.

Run as ``python bench/t4_bench.py <suite>``. Every suite shares the protocol in t4_bench_common, which reads the
predictor's speed["inference"] exactly as profile_depth.py and ProfileModels do, so a new suite is a dict entry
rather than another runner. Suites naming yamls that this checkout does not carry simply are not run from it.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry
carries an explicit size letter. The two-letter xxl needs the widened scale regex in tasks.py to resolve at all.

`delta_vs_base_pct` is an architecture comparison only between rows at the baseline's scale. Rows at another scale
are in the same suite to share its thermal state, and their delta against it is scale, not architecture.
"""

import sys
from pathlib import Path

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from this commit. sys.path[0] is
# bench/, not the root. Ruff permits a sys.path mutation ahead of imports, so this needs no E402 suppression,
# which the formatter strips anyway.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t4_bench_common import build_variant, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR, YOLO

DATA = Path("/root/autodl-tmp/data")

# Lane A exports stock, Lane B with the fp32 attention pin DINOv3 needs to survive fp16. Six Lane A entries carry
# MHSA and would also move under a pin, so only within-lane ratios travel.
SUITES = {
    # Nine of the eleven cover every orc phase2 run, one per distinct (yaml, scale), replacing registered numbers
    # taken across sessions under a wrong timer and weight state, two of them conflicting. Two rows are new.
    "lane-a": (
        "conv-x",
        YOLO,
        None,
        {
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
        },
    ),
    # Depth-matching cut dinop5's win over attn2 from 3.66pp to 0.64pp at X. Does that collapse hold below X?
    "lane-a-ml": (
        "conv-m",
        YOLO,
        None,
        {
            "conv-m": "yolo26m.yaml",
            "uvit-m-attn2": "yolo26m-ultravit-repmixer-fastvitffn-attn2.yaml",
            "uvit-m-dinop5": "yolo26m-ultravit-repmixer-fastvitffn-dinop5.yaml",
            "uvit-m-dinop5-depthmatched": "yolo26m-ultravit-repmixer-fastvitffn-dinop5-depthmatched.yaml",
            "conv-l": "yolo26l.yaml",
            "uvit-l-attn2": "yolo26l-ultravit-repmixer-fastvitffn-attn2.yaml",
            "uvit-l-dinop5": "yolo26l-ultravit-repmixer-fastvitffn-dinop5.yaml",
            "uvit-l-dinop5-depthmatched": "yolo26l-ultravit-repmixer-fastvitffn-dinop5-depthmatched.yaml",
        },
    ),
    # The detr_decoder_clean2 reference arms every Lane B row is compared against at its own scale. The family
    # changes along the ladder: CSP trunk with RTDETRDecoderEfficient at n and s, DeimDecoder at m and l, plain ViT
    # trunk at x and xxl.
    "laneb-baselines": (
        "yolo27x",
        RTDETR,
        pinned_fp32_attn,
        {
            "yolo27n": "yolo27n-detr.yaml",
            "yolo27s": "yolo27s-detr.yaml",
            "yolo27m": "yolo27m-deim-detr.yaml",
            "yolo27l": "yolo27l-deim-detr.yaml",
            "yolo27x": "yolo27x-vit-detr.yaml",
            "yolo27xxl": "yolo27xxl-vit-detr.yaml",
        },
    ),
    # Five UltraViT backbones on one DEIMv2 neck. DINOv3-S+ is a row, not the baseline, so these stay readable
    # against both the yolo27 arm and every earlier run.
    "laneb-x": (
        "yolo27x",
        RTDETR,
        pinned_fp32_attn,
        {
            "yolo27x": "yolo27x-vit-detr.yaml",
            "dinov3splus": "deim_dinov3splus_sta_l6_xl.yaml",
            "ffnattn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml",
            "fastvitffn-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml",
            "repmixer-dinop5": "yolo26x-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml",
            "fastvitffn-dinop5-depthmatched": (
                "yolo26x-ultravit-repmixer-fastvitffn-dinop5-depthmatched-deim_mal_deimv2Neck.yaml"
            ),
            "ffnattn2-p4win": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4win-deim_mal_deimv2Neck.yaml",
        },
    ),
}

suite = sys.argv[1]
baseline, model_cls, engine_builder, yamls = SUITES[suite]
engines = DATA / f"t4-{suite}-engines"
variants = [build_variant(t, y, engines, model_cls=model_cls, engine_builder=engine_builder) for t, y in yamls.items()]
run_benchmark(variants, baseline, DATA / f"t4_{suite.replace('-', '_')}_protocol0727.csv")
