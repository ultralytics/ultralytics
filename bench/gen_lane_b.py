# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generate a Lane B arm yaml: the scale's yolo27 baseline with only its trunk replaced by a Lane A arm's.

Run as ``python bench/gen_lane_b.py <tag> <lane-a-stem> [scales]``, e.g.

    python bench/gen_lane_b.py dinop5-l640 yolo26-ultravit-repmixer-fastvitffn-dinop5-l640 nsl

Repeats and channel args are emitted already resolved, under identity multipliers. The older hand-written arms
pre-divided each by the baseline's own depth and width, which breaks once the trunk wants a channel above the
baseline's max_channels, since `make_divisible(min(c, mx) * w, 8)` cannot exceed `mx`. An l trunk with a 640-wide
P5 under a 512 cap is that case. Everything else parse_model does, `scale_args` tokens above all, still happens at
load, so these files are not fully expanded.

Correctness is proved from built models: rows 0-8 must match the Lane A arm and its cls twin on parameter shapes,
`resolve` must reproduce parse_model on the untouched baseline, and the arm must run a forward pass.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from ultralytics.nn import tasks
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, yaml_model_load
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import get_num_params

ROOT = Path(__file__).resolve().parent.parent / "ultralytics/cfg/models"
FAMILY = {"n": "detr", "s": "detr", "m": "deim-detr", "l": "deim-detr", "x": "vit-detr"}
IMGSZ = {"n": 480, "s": 512, "m": 512, "l": 640, "x": 640}
TRUNK_ROWS = 9  # an UltraViT trunk is rows 0-8, where the CSP baseline's is 0-10


def _base_modules():
    """Module names parse_model width-scales, read from parse_model itself so this cannot drift from it."""
    tree = ast.parse(Path(tasks.__file__).read_text())
    fn = next((f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "parse_model"), None)
    for node in ast.walk(fn) if fn else ():
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "base_modules":
            return frozenset(ast.unparse(e).split(".")[-1] for e in node.value.args[0].elts)
    raise RuntimeError("parse_model no longer defines base_modules, so channel scaling cannot be mirrored")


BASE_MODULES = _base_modules()


def resolve(d, scale):
    """Resolve every row's repeats and channel arg at one scale.

    Only `base_modules` take an output-channel count first, so only those are width-scaled. A ViT block carries a kernel
    size there and must pass through untouched.
    """
    depth, width, mx = d["scales"][scale]
    rows = []
    for f, n, m, args in d["backbone"] + d["head"]:
        n = max(round(n * depth), 1) if n > 1 else n
        args = list(args)
        if m in BASE_MODULES and args and isinstance(args[0], int) and args[0] != d.get("nc"):
            args[0] = make_divisible(min(args[0], mx) * width, 8)
        rows.append([f, n, m, args])
    return rows


def remap(f, nb):
    """Rewrite a head `from` index off the baseline's `nb`-row trunk onto the UltraViT trunk.

    Not a uniform shift. Both trunks put P3 at row 4 and P4 at row 6, so those must not move. Only the baseline's P5
    tail collapses onto the last UltraViT row, and only rows past the trunk shift up. Shifting everything rewires P3 and
    P4 to the stems feeding them, which still builds and fails at the first forward pass.
    """
    if isinstance(f, list):
        return [remap(x, nb) for x in f]
    if not isinstance(f, int) or f < TRUNK_ROWS - 1:
        return f
    return TRUNK_ROWS - 1 if f < nb else f - (nb - TRUNK_ROWS)


def check_layout(base, scale):
    """Reject a baseline whose trunk is not the 11-row CSP layout `remap` assumes."""
    b = base["backbone"]
    if not (len(b) == 11 and b[4][2] == "C3k2" and b[6][2] == "C3k2" and b[10][2] == "C2PSA"):
        raise SystemExit(
            f"scale {scale} uses yolo27-{FAMILY[scale]}.yaml, a {len(b)}-row trunk, not the 11-row CSP layout with "
            f"P3 at row 4 and P4 at row 6. Its `from` indices cannot be remapped by row position, so an arm for it "
            f"must be derived by hand, as yolo27x-ultravit-*-vit-detr.yaml already is. Pass a scale list omitting {scale}."
        )


def emit(tag, stem, scale):
    """Write one Lane B arm yaml and return its path."""
    family = FAMILY[scale]
    base = yaml_model_load(ROOT / f"27/yolo27-{family}.yaml")
    check_layout(base, scale)
    arm_a = yaml_model_load(ROOT / f"26/{stem.replace('yolo26', 'yolo26' + scale, 1)}.yaml")

    nb = len(base["backbone"])
    trunk = [Flow(r) for r in resolve(arm_a, scale)[:TRUNK_ROWS]]  # resolve returns backbone + head, drop its head
    rest = [Flow([remap(f, nb), n, m, a]) for f, n, m, a in resolve(base, scale)[nb:]]
    # Only channel-bearing rows may set the cap, since a ViT block's first arg is a kernel size.
    mx = max(a[0] for _, _, m, a in trunk + rest if m in BASE_MODULES and a and isinstance(a[0], int))

    out = {k: v for k, v in base.items() if k in ("nc", "end2end", "reg_max")}
    out["scales"] = {scale: [1.0, 1.0, mx]}
    if "scale_args" in base:
        out["scale_args"] = {scale: base["scale_args"][scale]}
    out["backbone"], out["head"] = trunk, rest

    body = dump(out)
    assert yaml.safe_load(body) == out, "the emitted yaml does not round-trip back to the spec it was built from"
    path = ROOT / f"27/yolo27{scale}-ultravit-{tag}-{family}.yaml"
    path.write_text(
        "# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n\n"
        f"# Lane B arm, generated by bench/gen_lane_b.py from yolo27-{family}.yaml and {stem}.yaml at scale {scale}.\n"
        "# Neck, decoder and scale_args are the baseline's, the trunk is the Lane A arm's, every row already\n"
        "# resolved under identity multipliers. Do not hand-edit, regenerate.\n\n" + body
    )
    return path


def shapes(model, lo=0, hi=None):
    """Parameter names and shapes over a slice of a built model's rows."""
    return [(n, tuple(p.shape)) for blk in model.model[lo:hi] for n, p in blk.named_parameters()]


class Flow(list):
    """A row emitted inline, `- [-1, 1, Conv, [64, 3, 2]]`, the style every hand-written model yaml uses."""


_Dumper = type("_Dumper", (yaml.SafeDumper,), {})
_Dumper.add_representer(Flow, lambda d, r: d.represent_sequence("tag:yaml.org,2002:seq", r, flow_style=True))


def dump(spec):
    """Serialize a model spec the way the hand-written model yamls read.

    `default_flow_style=None` inlines `scales` and `scale_args` but not a row, which nests an args list. Marking rows
    `Flow` inlines them, since a flow sequence cannot contain a block one. `width` must be effectively unbounded or the
    longest decoder row folds mid-sequence.
    """
    return yaml.dump(spec, Dumper=_Dumper, sort_keys=False, default_flow_style=None, width=10**9)


def faithful(base_path, scale):
    """Check `resolve` against parse_model by re-emitting the baseline with no trunk swap and rebuilding it.

    This is the only check that can fail on a `resolve` bug rather than on serialization damage. It catches a
    max_channels rule `resolve` does not mirror, C2fAttn and Segment today, or a module whose yaml spelling does not
    match its scraped class name.
    """
    base = yaml_model_load(base_path)
    rows = resolve(base, scale)
    nb = len(base["backbone"])
    mx = max(a[0] for _, _, m, a in rows if m in BASE_MODULES and a and isinstance(a[0], int))
    spec = {k: v for k, v in base.items() if k in ("nc", "end2end", "reg_max")}
    spec["scales"] = {scale: [1.0, 1.0, mx]}
    if "scale_args" in base:
        spec["scale_args"] = {scale: base["scale_args"][scale]}
    spec["backbone"], spec["head"] = rows[:nb], rows[nb:]

    echo = base_path.with_name(f"yolo27{scale}-{FAMILY[scale]}-resolvecheck.yaml")
    echo.write_text(dump(spec))
    try:
        ref = base_path.with_name(f"yolo27{scale}-{FAMILY[scale]}.yaml")
        same = shapes(DetectionModel(str(echo), ch=3, nc=80, verbose=False)) == shapes(
            DetectionModel(str(ref), ch=3, nc=80, verbose=False)
        )
    finally:
        echo.unlink(missing_ok=True)
    assert same, f"resolve() no longer reproduces parse_model for yolo27-{FAMILY[scale]}.yaml at {scale}"


def prove(path, stem, scale):
    """Fail unless the arm carries the Lane A trunk, `resolve` still mirrors parse_model, and it runs at its size.

    Rechecking the emitted rows against `resolve` would compare it against itself, so the trunk claim rests on parameter
    shapes off independently built Lane A models and the head claim on `faithful`.
    """
    sized = stem.replace("yolo26", "yolo26" + scale, 1)
    arm = DetectionModel(str(path), ch=3, nc=80, verbose=False)
    ref = DetectionModel(str(ROOT / f"26/{sized}.yaml"), ch=3, nc=80, verbose=False)
    cls = ClassificationModel(str(ROOT / f"26/{sized}-cls.yaml"), ch=3, nc=1000, verbose=False)
    trunk = shapes(arm, 0, TRUNK_ROWS)
    assert trunk == shapes(ref, 0, TRUNK_ROWS), f"{path.name}: trunk parameter shapes differ from Lane A"
    assert trunk == shapes(cls, 0, TRUNK_ROWS), f"{path.name}: trunk parameter shapes differ from its cls twin"
    faithful(ROOT / f"27/yolo27-{FAMILY[scale]}.yaml", scale)

    arm.eval()
    with torch.no_grad():
        arm(torch.zeros(1, 3, IMGSZ[scale], IMGSZ[scale]))
    return get_num_params(arm)


def generate(tag, stem, scale):
    """Emit an arm and prove it, removing the file if the proof fails so no unproven yaml survives."""
    path = emit(tag, stem, scale)
    try:
        return path, prove(path, stem, scale)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    tag, stem, *rest = sys.argv[1:]
    for scale in rest[0] if rest else "nsml":
        path, n = generate(tag, stem, scale)
        print(f"{path.name}  {n / 1e6:.2f}M params, proved at {IMGSZ[scale]}px", flush=True)
