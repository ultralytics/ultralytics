# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 TensorRT latency for one lane at one scale, measured in a single paired session.

Run as ``python bench/t4_bench.py <lane>-<scale> [arm,arm] [warmup=N]``, e.g. `lane-a-x` or `lane-b-n repmixer`.
Given no arm list a session profiles that scale's baseline, its bridge and every arm that exists there. Only
TensorRT is timed, the other formats run once afterwards into a `.formats.csv` sidecar, see t4_bench_common for
why. `warmup=N` overrides the standard conditioning and marks the session exploratory, see run_benchmark.

The bridge is the one arm carried by every session at a scale. Deltas from two sessions are not directly
comparable, since re-measuring the baseline in each leaves 1.0 to 1.5pp of scatter, but the bridge anchors them:
an arm's ratio against the bridge inside its own session transfers exactly. A second argument, a comma list of
arm tags, spends that: `lane-a-x dinop5-mixedrope` times only that arm against the baseline and bridge, so one
new yaml costs three measurements rather than a rerun of the scale.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry has
its size letter substituted in. Arms absent at a scale are simply not in that session.

Every arm in a session runs at the baseline's deployed input size, from `IMGSZ` below, since a paired comparison
only holds at one size. Lane A deploys at 640 throughout, Lane B does not.
"""

import sys
from pathlib import Path

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from this commit. sys.path[0] is
# bench/, not the root. Ruff permits a sys.path mutation ahead of imports, so this needs no E402 suppression,
# which the formatter strips anyway.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t4_bench_common import build_variant, parse_session, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR, YOLO

DATA = Path("/root/autodl-tmp/data")

# The size each lane's baseline deploys at, per scale, defaulting to 640. yolo27-detr trains n at 480 and s and m
# at 512, so timing those at 640 measures an operating point nobody ships.
IMGSZ = {("lane-b", "n"): 480, ("lane-b", "s"): 512, ("lane-b", "m"): 512}

# lane -> (facade, engine builder, baseline tag prefix, bridge tag, {arm tag: (yaml template, scales it exists at)}).
#
# Lane A exports stock, Lane B with the fp32 attention pin DINOv3 needs to survive fp16, so only within-lane ratios
# travel.
#
# Retired and absent: fracrope, headdim64, attn2lite, the l640 family, and the dinop5, dinop5-mixedrope and
# dinop5-depthmatched variants the hybrid pair superseded. Their yamls stay, historical rows reference them.
LANES = {
    "lane-a": (
        YOLO,
        None,
        "conv",
        "ffnattn2",  # bridge, the incumbent every UltraViT arm is judged against
        {
            "conv": ("yolo26{s}.yaml", "nsmlx"),
            "attn2": ("yolo26{s}-ultravit-attn2.yaml", "nsmlx"),
            "base": ("yolo26{s}-ultravit.yaml", "nsmlx"),
            "dinoreg": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml", "smlx"),
            "dinorope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-dinorope.yaml", "smlx"),
            "fastvitffn": ("yolo26{s}-ultravit-repmixer-fastvitffn.yaml", "nsmlx"),
            "ffnattn2": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2.yaml", "nsmlx"),
            "mixedrope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-mixedrope.yaml", "smlx"),
            "p4pooled": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml", "nsmlx"),
            "p4win": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4win.yaml", "x"),
            "repmixer": ("yolo26{s}-ultravit-repmixer.yaml", "nsmlx"),
            "s480": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-s480.yaml", "s"),
            # Incumbent pair, the control the stage-balanced pair is judged against.
            "dinop5-hybrid": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-hybrid.yaml", "nsmlx"),
            "dinop5-mixedrope-hybrid": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-hybrid.yaml", "nsmlx"),
            # The incumbent trunk with one block moved from P4 to P3. s is the cell to watch, the incumbent's
            # thinnest margin over the conv baseline, and this spends MACs at four times the resolution it frees them.
            "dinop5-p3deep": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3deep.yaml", "nsmlx"),
            # p3deep won at n and s and lost at m and l, so the exchange it makes is priced by where the T4 sits on
            # the bandwidth to compute curve. p3deep2 takes a second step and belongs at s. p5lean drains P5, the
            # stage furthest over the conv floor at m, l and x, and the combined arm does both.
            "dinop5-p3deep2": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3deep2.yaml", "nsmlx"),
            "dinop5-p5lean": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p5lean.yaml", "mlx"),
            "dinop5-p3deep2-p5lean": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3deep2-p5lean.yaml", "mlx"),
            # The baseline itself with only C2PSA swapped for the dinop5 attention, so P2 and P3 sit exactly at the
            # conv floor. A whole-operator substitution, not an attention-type isolation: C2PSA attends inside a
            # half-width split while MHSABlock runs at full width.
            "convtrunk-dinop5": ("yolo26{s}-convtrunk-dinop5.yaml", "nsmlx"),
            # Stage-balanced pair, every P stage above both lanes' baselines at every scale. n is the cell to watch,
            # most added FLOPs against the thinnest measured margin.
            "dinop5-stagebal": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-stagebal.yaml", "nsmlx"),
            "dinop5-mixedrope-stagebal": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-stagebal.yaml",
                "nsmlx",
            ),
        },
    ),
    "lane-b": (
        RTDETR,
        pinned_fp32_attn,
        "yolo27",
        "ffnattn2",
        {
            # The detr_decoder_clean2 reference arms. The family changes along the ladder: CSP trunk with
            # RTDETRDecoderEfficient at n and s, DeimDecoder at m and l, plain ViT trunk at x.
            "yolo27": ("yolo27{s}-detr.yaml", "ns"),
            "yolo27-deim": ("yolo27{s}-deim-detr.yaml", "ml"),
            "yolo27-vit": ("yolo27{s}-vit-detr.yaml", "x"),
            "dinov3splus": ("deim_dinov3splus_sta_l6_xl.yaml", "x"),
            "attn2": ("yolo26{s}-ultravit-attn2-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "base": ("yolo26{s}-ultravit-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "dinop5": ("yolo26{s}-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml", "smlx"),
            "dinoreg": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-deim_mal_deimv2Neck.yaml", "x"),
            "dinorope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-dinorope-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "fastvitffn": ("yolo26{s}-ultravit-repmixer-fastvitffn-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "fastvitffn-dinop5": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml", "smlx"),
            "fastvitffn-dinop5-mixedrope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-deim_mal_deimv2Neck.yaml",
                "smlx",
            ),
            "fastvitffn-dinop5-depthmatched": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-depthmatched-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "ffnattn2": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "mixedrope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-mixedrope-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "p4deep": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4deep-deim_mal_deimv2Neck.yaml", "x"),
            "p4pooled": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4pooled-deim_mal_deimv2Neck.yaml", "x"),
            "p4win": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4win-deim_mal_deimv2Neck.yaml", "x"),
            "repmixer": ("yolo26{s}-ultravit-repmixer-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "s480": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-s480-deim_mal_deimv2Neck.yaml", "s"),
        },
    ),
}

lane, scale, session, only, warmup = parse_session(sys.argv[1:])
model_cls, engine_builder, base_prefix, bridge, arms = LANES[lane]
yamls = {tag: t.format(s=scale) for tag, (t, scales) in arms.items() if scale in scales}
# Lane B's baseline changes file along the ladder, so it is whichever arm carries the baseline prefix at this scale.
baseline = next(tag for tag in yamls if tag.startswith(base_prefix))
assert bridge in yamls, f"{session} has no bridge arm, so its ratios cannot be anchored to another session"
if only:  # append session, carrying the named arms plus the baseline and bridge that anchor them
    (arg,) = only  # unpack, so a stray extra argument fails here rather than being ignored
    want = set(arg.split(","))
    assert want <= set(yamls), f"{sorted(want - set(yamls))} absent from {session}, which would bench anchors alone"
    yamls = {t: y for t, y in yamls.items() if t in want | {baseline, bridge}}
    session += "-" + arg  # its own session id, since it is its own measurement occasion

imgsz = IMGSZ.get((lane, scale), 640)
engines = DATA / f"t4-{lane}-{scale}-engines"
variants = [
    build_variant(t, y, engines, imgsz, model_cls=model_cls, engine_builder=engine_builder) for t, y in yamls.items()
]
run_benchmark(
    variants,
    dict.fromkeys(yamls, baseline),
    DATA / f"t4_{session.replace('-', '_')}.csv",
    session,
    imgsz,
    warmup=warmup,
)
