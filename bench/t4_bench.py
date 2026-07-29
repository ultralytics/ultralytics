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
its size letter substituted in. Arms absent at a scale are simply not in that session. Both lanes now draw from one
arm table, so a full Lane B session carries every arm its scale has rather than the handful that had a hand-written
stem. The added cost is mostly the sidecar formats, around 2.5 min a variant against 1.5 for the entire TensorRT
pass, so prefer the arm list. The session header prints the count, which drifts as arms retire.

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
from ultralytics.nn.tasks import yaml_model_load

DATA = Path("/root/autodl-tmp/data")

# The size each lane's baseline deploys at, per scale, defaulting to 640. yolo27-detr trains n at 480 and s and m
# at 512, so timing those at 640 measures an operating point nobody ships.
IMGSZ = {("lane-b", "n"): 480, ("lane-b", "s"): 512, ("lane-b", "m"): 512}

# lane -> (facade, engine builder, baseline tag, bridge tag, baseline yaml template, arm yaml template).
#
# Lane A is the YOLO26 conv-head detector against its own conv baseline, exported stock. Lane B is the DETR
# detector against the yolo27 baseline of its scale's family, see FAMILY, exported with the fp32 attention pin
# DINOv3 needs to survive fp16. Six Lane A arms carry MHSA and would also move under a pin, so only within-lane ratios travel.
#
# A Lane B arm is its baseline with only the trunk swapped for the Lane A arm's, so its name is derivable from the
# tag and the scale's baseline family and it needs no second stem table. A `None` arm template means the lane uses
# the Lane A stem in ARMS.
LANES = {
    "lane-a": (YOLO, None, "conv", "ffnattn2", "yolo26{s}.yaml", None),
    "lane-b": (
        RTDETR,
        pinned_fp32_attn,
        "yolo27",  # baseline
        "ffnattn2",  # bridge, the incumbent every UltraViT arm is judged against
        "yolo27{s}-{f}.yaml",
        "yolo27{s}-ultravit-{tag}-{f}.yaml",
    ),
}

# The baseline family each scale deploys, which also names its yaml: CSP trunk with RTDETRDecoderEfficient at n and
# s, DeimDecoder at m and l, ViT + Spatial Tuning Adapter at x.
FAMILY = {"n": "detr", "s": "detr", "m": "deim-detr", "l": "deim-detr", "x": "vit-detr"}

# arm tag -> (Lane A yaml template, scales it exists at). Lane B carries exactly these trunks.
#
# Retired and absent: fracrope, headdim64, attn2lite, p4deep, which never had a Lane A trunk to swap in, the l640
# family, and the dinop5, dinop5-mixedrope and dinop5-depthmatched variants the hybrid pair superseded.
# dinov3splus was promoted rather than retired, its trunk is what yolo27-vit-detr spells out, so it is the x baseline.
ARMS = {
    "attn2": ("yolo26{s}-ultravit-attn2.yaml", "nsmlx"),
    "base": ("yolo26{s}-ultravit.yaml", "nsmlx"),
    "fastvitffn": ("yolo26{s}-ultravit-repmixer-fastvitffn.yaml", "nsmlx"),
    "ffnattn2": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2.yaml", "nsmlx"),
    "p4pooled": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml", "nsmlx"),
    "repmixer": ("yolo26{s}-ultravit-repmixer.yaml", "nsmlx"),
    "dinoreg": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml", "smlx"),
    "dinorope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-dinorope.yaml", "smlx"),
    "mixedrope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-mixedrope.yaml", "smlx"),
    "p4win": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4win.yaml", "x"),
    "s480": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-s480.yaml", "s"),
    # Incumbent pair, the control the stage-balanced pair is judged against. Both x arms are hand written,
    # gen_lane_b.py cannot remap the 27-row vit-detr trunk by row position.
    "dinop5-hybrid": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-hybrid.yaml", "nsmlx"),
    "dinop5-mixedrope-hybrid": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-hybrid.yaml", "nsmlx"),
    # P3-rebalanced trunks. In Lane A, p3deep beat the incumbent at n and s and lost at m, l and x, while draining
    # P5 recovered that loss and more, so p3deep2-p5lean leads at m and l. p3wide reaches a comparable P3 by
    # mlp_ratio rather than by block count. Only p3deep has an x arm, hand written like the other two above.
    "dinop5-p3deep": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3deep.yaml", "nsmlx"),
    "dinop5-p5lean": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p5lean.yaml", "ml"),
    "dinop5-p3deep2-p5lean": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3deep2-p5lean.yaml", "ml"),
    "dinop5-p3wide": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-p3wide.yaml", "nsml"),
    # Stage-balanced pair, every P stage above both lanes' baselines at every scale.
    "dinop5-stagebal": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-stagebal.yaml", "nsmlx"),
    "dinop5-mixedrope-stagebal": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-stagebal.yaml", "nsmlx"),
}

lane, scale, session, only, warmup = parse_session(sys.argv[1:])
model_cls, engine_builder, baseline, bridge, base_yaml, arm_yaml = LANES[lane]
sub = {"s": scale, "f": FAMILY[scale]}
yamls = {baseline: base_yaml.format(**sub)}
yamls |= {t: (arm_yaml or a).format(**sub, tag=t) for t, (a, scales) in ARMS.items() if scale in scales}
assert bridge in yamls, f"{session} has no bridge arm, so its ratios cannot be anchored to another session"
if only:  # append session, carrying the named arms plus the baseline and bridge that anchor them
    (arg,) = only  # unpack, so a stray extra argument fails here rather than being ignored
    want = set(arg.split(","))
    assert want <= set(yamls), f"{sorted(want - set(yamls))} absent from {session}, which would bench anchors alone"
    yamls = {t: y for t, y in yamls.items() if t in want | {baseline, bridge}}
    session += "-" + arg  # its own session id, since it is its own measurement occasion

imgsz = IMGSZ.get((lane, scale), 640)
# Resolve every name before building any of them, so a session carrying more arms than intended, or a name that does
# not fit the derived form, shows up now rather than after the exports ahead of it have already run.
print(f"=== {session} at {imgsz}px, {len(yamls)} variants: {' '.join(yamls)}", flush=True)
for y in yamls.values():
    yaml_model_load(y)
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
