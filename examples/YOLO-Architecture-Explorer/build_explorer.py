# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build an interactive, scale-symbolic architecture explorer for the Ultralytics detection configs.

Every scale of every config is built and traced once at 640x640. Repeated units collapse to one "×n" node, block
internals are keyed by their real structure, and shapes are fitted across scales into expressions such as
"80×80×512×w×(1+r)" (top level) or "h×w×0.5c_out" (inside a block), so each config file yields one diagram.

Usage:
    python3 build_explorer.py                 # all configs -> yolo_architecture_explorer.html
    python3 build_explorer.py --only yolov8   # quick run on matching config stems
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))  # use this checkout's ultralytics package

import torch
from trace_graph import GraphTracer

from ultralytics.nn.tasks import BaseModel, DetectionModel, guess_model_task
from ultralytics.utils import LOGGER, YAML

CFG = REPO / "ultralytics/cfg/models"
IMGSZ = 640


def version_label(folder: str) -> str:
    """Display name of a config folder: 'v8' -> 'YOLOv8', '11' -> 'YOLO11', 'rt-detr' -> 'RT-DETR'."""
    return f"YOLO{folder}" if re.fullmatch(r"v?\d+", folder) else folder.upper()


def version_rank(label: str) -> float:
    """Chronological order from the number in the label; labels without one go last."""
    m = re.search(r"\d+", label)
    return int(m[0]) if m else float("inf")


def version_rank_finite(label: str) -> float:
    """version_rank with unnumbered labels first, for picking the newest numbered version."""
    r = version_rank(label)
    return -1 if r == float("inf") else r


def discover() -> list[tuple[str, Path]]:
    """Every detection config under cfg/models as (version, path), oldest version first.

    The task comes from Ultralytics' own guess_model_task. Configs whose head it does not know (None) are tried as well,
    so a new head type is picked up without changes here; anything that cannot be built and traced on an image alone
    (e.g. text-prompted heads) is skipped later and listed on the page.
    """
    found = []
    for path in sorted(CFG.rglob("*.yaml")):
        try:
            task = guess_model_task(YAML.load(path))
        except Exception:  # noqa: S112 - not a model config
            continue
        if task in {"detect", None}:
            found.append((version_label(path.parent.name), path))
    return sorted(found, key=lambda vp: (version_rank(vp[0]), vp[0], vp[1].stem))


FAMILIES = [  # torch-level primitives only; every other block gets its family from its traced structure
    ("attn", {"MultiheadAttention"}),
    ("pool", {"MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"}),
    ("up", {"Upsample", "ConvTranspose2d"}),
    ("concat", {"Concat"}),
    ("conv", {"Conv2d"}),
    ("norm", {"BatchNorm2d", "LayerNorm", "SiLU", "ReLU", "GELU", "Identity", "Dropout", "Linear", "Embedding"}),
]


# --------------------------------------------------------------------------------------------------------------------
# build + trace
# --------------------------------------------------------------------------------------------------------------------
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


ROW = re.compile(r"^\s*(\d+)\s+(-?\d+|\[[^\]]*\])\s+(\d+)\s+(\d+)\s{2}(\S+)\s+(.*?)\s*$")


def build_model(path: Path, scale: str | None):
    """Build a model at one scale, returning it with the per-layer resolved args parse_model logs."""
    d = YAML.load(path)
    d["yaml_file"] = str(path)
    if scale:
        d["scale"] = scale
    cap = _Capture()
    LOGGER.addHandler(cap)
    try:
        model = DetectionModel(d, verbose=True)
    finally:
        LOGGER.removeHandler(cap)
    rows = {}
    for line in cap.lines:
        m = ROW.match(line)
        if m:
            try:
                args = ast.literal_eval(m[6])
            except (ValueError, SyntaxError):
                args = m[6]
            rows[int(m[1])] = {"n": int(m[3]), "np": int(m[4]), "module": m[5], "args": args}
    return model.eval(), rows


# --------------------------------------------------------------------------------------------------------------------
# graph processing: collapse repeats, structural keys, samples for fitting
# --------------------------------------------------------------------------------------------------------------------
STRUCT: dict[str, dict] = {}  # struct key -> topology
SAMPLES: dict[tuple[str, str], list] = defaultdict(list)  # (struct key, config id) -> [(graph, ctx)]
STRUCT_HP = {"k", "s", "p", "d", "act"}
FNS: dict[str, dict] = {}  # helper methods that group ops, collected from every trace


def _hp_split(hp: dict) -> tuple[dict, dict]:
    """Split hyper-params into structural (shape-independent) and numeric (fitted) parts."""
    struct, num = {}, {}
    for k, v in hp.items():
        if k == "f":
            continue
        if k in STRUCT_HP or isinstance(v, (bool, str)):
            struct[k] = v
        elif isinstance(v, (int, float)):
            num[k] = v
    return struct, num


def _collapse(g: dict) -> dict:
    """Collapse chained homogeneous repeats (e.g. C2f.m.0..m.k) into one group node carrying n."""
    nodes, edges = g["nodes"], g["edges"]
    feeds = {(s, d) for s, _, d, _ in edges}
    groups, cur = [], None
    for n in nodes:
        r = n.get("rep")
        if r is None:
            continue
        last = nodes[cur[-1]] if cur else None
        if last and last.get("rep") == r and last.get("g") == n.get("g") and (last["id"], n["id"]) in feeds:
            cur.append(n["id"])
        else:
            cur = [n["id"]]
            groups.append(cur)
    per_rep = Counter(nodes[grp[0]]["rep"] for grp in groups)
    groups = [grp for grp in groups if per_rep[nodes[grp[0]]["rep"]] == 1]  # e.g. PSABlock.ffn: two different Convs
    head = {m: grp[0] for grp in groups for m in grp}
    for grp in groups:
        h = nodes[grp[0]]
        h["n"], h["name"] = len(grp), h["rep"]
    keep = [n for n in nodes if head.get(n["id"], n["id"]) == n["id"]]
    new = {n["id"]: i for i, n in enumerate(keep)}
    merged = {}
    for s, si, d, di in edges:
        if s in head and d in head and head[s] == head[d]:
            continue  # chain inside a group
        k = (new[head.get(s, s)], si, new[head.get(d, d)])
        prev = merged.get(k)
        merged[k] = (min(di, prev[0]), prev[1] + 1) if prev else (di, 1)
    for i, n in enumerate(keep):
        n["id"] = i
    return {"nodes": keep, "edges": [[s, si, d, di, c > 1] for (s, si, d), (di, c) in merged.items()]}


PHASES = ("train", "deploy")
PHASE_NAMES = {"train": "training", "deploy": "deployment"}


GLOSSARY = {"BatchNorm2d": "folded into the weights of the conv before it"}  # torch-level, not model-level


def phase_notes(stats: dict, diff: dict) -> list[str]:
    """Readable summary of the automatic training -> deployment diff (no per-model knowledge)."""
    d = diff["deploy"]
    rem, add = d["removed"], d["added"]
    parent = lambda pattern: pattern.rpartition(".")[0]
    leaf = lambda pattern: pattern.rpartition(".")[2]
    notes, used_r, used_a = [], set(), set()
    for ai, rs in d["replaced"]:
        pa, ta, ca, _ = add[ai]
        rs = [ri for ri in rs if ri not in used_r]
        if rs:
            used_a.add(ai)
            used_r.update(rs)
            old = " + ".join(sorted({f"{leaf(rem[ri][0])} ({rem[ri][1]})" for ri in rs}))
            notes.append(f"Replaced ×{ca}: {old} → {leaf(pa)} ({ta}) at {parent(pa)}")
    by_type = defaultdict(list)
    for ri, r in enumerate(rem):
        if ri not in used_r:
            by_type[r[1]].append(r)
    for t, rows in sorted(by_type.items(), key=lambda kv: -sum(r[3] for r in kv[1])):
        roots, size = sum(r[2] for r in rows), sum(r[3] for r in rows)
        where = ", ".join(r[0] for r in rows[:2]) + (f" and {len(rows) - 2} more patterns" if len(rows) > 2 else "")
        why = f", {GLOSSARY[t]}" if t in GLOSSARY else (f" ({size} modules)" if size > roots else "")
        notes.append(f"Removed {t} ×{roots}{why}: {where}")
    notes += [f"Added {ta} ×{ca}: {pa}" for ai, (pa, ta, ca, _) in enumerate(add) if ai not in used_a]
    for ph, calls in d["calls"].items():
        if calls:
            notes.append(f"Runs only in {PHASE_NAMES[ph]}: " + ", ".join(c + "()" for c in calls))
    for label, ops in (
        ("training", [o for o, _, b in d["ops"] if b == 0]),
        ("deployment", [o for o, a, _ in d["ops"] if a == 0]),
    ):
        if ops:
            notes.append(f"Tensor ops only in {label}: " + ", ".join(ops))
    notes.append(f"Parameters: {stats['train']['params']:,} (training) → {stats['deploy']['params']:,} (deployment).")
    return notes


def version_note(version: str, cfgs: list[dict], catalog: dict, rewired: dict) -> str:
    """Summary of a version from the data: its configs, new block types, new wirings of existing blocks, heads."""
    new = [k for k, c in catalog.items() if c.get("first") == version and c.get("file") and c["family"] != "fn"]
    heads = sorted({c["layers"][-1]["cls"] for c in cfgs})
    out = f"{len(cfgs)} config{'s' if len(cfgs) != 1 else ''}."
    if new:
        out += " First version to use: " + ", ".join(new) + "."
    if rewired.get(version):
        out += " New wiring of existing blocks: " + ", ".join(rewired[version]) + "."
    return out + (" Head: " + ", ".join(heads) + "." if heads else "")


def train_only(stats: dict) -> dict:
    """Top-level parts that run only while training, found by diffing the modules each mode called, for the layout.

    layers: whole layers that deployment drops (e.g. an auxiliary branch). branches: {layer: [[attribute, type,
    modules]]} for sub-branches a layer drops (e.g. a head's one-to-one branch); folded BatchNorms are not branches.
    """
    tr, dp = stats["train"]["names"], stats["deploy"]["names"]
    gone = {n for n, t in tr.items() if dp.get(n) != t}
    layers = sorted(int(n.split(".")[1]) for n in gone if re.fullmatch(r"model\.\d+", n))
    branches = defaultdict(list)
    for n in sorted(gone):
        m = re.fullmatch(r"model\.(\d+)\.([^.]+)", n)
        if m and int(m[1]) not in layers:
            size = sum(1 for x in gone if x == n or x.startswith(n + "."))
            if size > 2:
                branches[int(m[1])].append([m[2], tr[n], size])
    return {"layers": layers, "branches": dict(branches)}


def phase_diff(stats: dict) -> dict:
    """Differences between the training graph and each deployment graph, from the modules and helpers each one ran.

    Module paths are generalized by replacing indices with '*', so e.g. every 'model.4.m.0.cv1.bn' style path folds into
    the pattern 'model.*.m.*.cv1.bn'.
    """
    pat = lambda name: re.sub(r"\.\d+(?=\.|$)", ".*", name)
    tr = stats["train"]
    out = {}
    for ph, st in stats.items():
        if ph == "train":
            continue
        entry, parents = {}, {}
        for kind, a, b in (("removed", tr["names"], st["names"]), ("added", st["names"], tr["names"])):
            gone = {n for n, t in a.items() if b.get(n) != t}
            groups = defaultdict(lambda: [0, 0, set()])  # (pattern, type) -> [subtree roots, modules, parents]
            for n in gone:
                root = n
                while "." in root and root.rpartition(".")[0] in gone:
                    root = root.rpartition(".")[0]
                g = groups[(pat(root), a[root])]
                g[1] += 1
                g[0] += root == n
                if kind == "removed" or tr["registered"].get(root) != a[root]:  # only new modules replace others
                    g[2].add(root.rpartition(".")[0])
            rows = sorted(([p, t, c, size, par] for (p, t), (c, size, par) in groups.items()), key=lambda r: -r[3])
            entry[kind] = [r[:4] for r in rows]
            parents[kind] = [r[4] for r in rows]
        # a module built for deployment under the parent of removed ones replaces them, e.g. conv1 + conv2 -> conv
        entry["replaced"] = [
            [ai, rs]
            for ai, ap in enumerate(parents["added"])
            if (rs := [ri for ri, rp in enumerate(parents["removed"]) if rp & ap])
        ]
        entry["calls"] = {
            "train": sorted(set(tr["calls"]) - set(st["calls"])),
            ph: sorted(set(st["calls"]) - set(tr["calls"])),
        }
        ops = set(tr["ops"]) | set(st["ops"])
        entry["ops"] = sorted(
            [o, tr["ops"].get(o, 0), st["ops"].get(o, 0)] for o in ops if tr["ops"].get(o, 0) != st["ops"].get(o, 0)
        )
        out[ph] = entry
    return out


def _group_calls(g: dict, tag: str = "grp") -> dict:
    """Fold nodes sharing a tag into one node that opens as a sub-graph.

    tag 'grp': the nodes one helper-method call ran, e.g. Detect.forward_head. tag 'seq': the layers of one mixed
    nn.Sequential branch, e.g. Detect.cv2.0 = Conv -> Conv -> Conv2d.
    """
    nodes, edges = g["nodes"], g["edges"]
    member = {}
    for n in nodes:
        t = n.pop(tag, None)
        if t:
            member[n["id"]] = tuple(t) if isinstance(t, list) else (t,)
    if not member:
        return g
    new, idmap, pseudo = [], {}, {}
    for n in nodes:
        t = member.get(n["id"])
        if t is None:
            idmap[n["id"]] = len(new)
            new.append(n)
        elif t not in pseudo:
            pseudo[t] = len(new)
            new.append(
                {"kind": "module", "type": t[0] + "()", "hp": {"fn": t[0]}, "params": 0}
                if tag == "grp"
                else {"kind": "module", "type": "Sequential", "hp": {}, "params": 0}
            )
    ins = {t: [] for t in pseudo}  # (src, src_out) feeding each group, in order
    outs = {t: [] for t in pseudo}
    for s, si, d, _ in edges:
        ts, td = member.get(s), member.get(d)
        if ts != td:
            if td and (s, si) not in ins[td]:
                ins[td].append((s, si))
            if ts and (s, si) not in outs[ts]:
                outs[ts].append((s, si))
    byid = {n["id"]: n for n in nodes}
    for t, pid in pseudo.items():
        mem = [n for n in nodes if member.get(n["id"]) == t]
        sub = [{"kind": "input", "type": f"in{k}", "out": [byid[s]["out"][si]]} for k, (s, si) in enumerate(ins[t])]
        loc = {}
        for n in mem:
            loc[n["id"]] = len(sub)
            sub.append(n)
        sub += [{"kind": "output", "type": f"out{k}", "out": [byid[s]["out"][si]]} for k, (s, si) in enumerate(outs[t])]
        sedges = []
        for s, si, d, di in edges:
            if member.get(d) == t:
                src = (loc[s], si) if member.get(s) == t else (ins[t].index((s, si)), 0)
                sedges.append([*src, loc[d], di])
            if member.get(s) == t and member.get(d) != t:
                sedges.append([loc[s], si, len(ins[t]) + len(mem) + outs[t].index((s, si)), 0])
        for i, n in enumerate(sub):
            n["id"] = i
        roots = sorted({n["name"].split(".")[0] for n in mem if n["kind"] == "module" and n.get("name")})
        if tag == "seq":
            new[pid]["hp"]["layers"] = " → ".join(n["type"] for n in mem if n["kind"] == "module")
        new[pid].update(
            name=", ".join(roots[:3]) if tag == "grp" else t[0],
            params=sum(n.get("params", 0) for n in mem),
            out=[byid[s]["out"][si] for s, si in outs[t]],
            g={"nodes": sub, "edges": [list(e) for e in dict.fromkeys(map(tuple, sedges))]},
        )
    pedges = {}
    for s, si, d, di in edges:
        ts, td = member.get(s), member.get(d)
        if ts and ts == td:
            continue
        src = (pseudo[ts], outs[ts].index((s, si))) if ts else (idmap[s], si)
        dst = (pseudo[td], ins[td].index((s, si))) if td else (idmap[d], di)
        pedges[(*src, *dst)] = None
    for i, n in enumerate(new):
        n["id"] = i
    return {"nodes": new, "edges": [list(e) for e in pedges]}


def _attrs(node: dict) -> dict:
    """Integer attributes of the module that owns a graph, e.g. C2f.c (hidden channels)."""
    return {
        k: v
        for k, v in node.get("hp", {}).items()
        if isinstance(v, int) and not isinstance(v, bool) and k not in STRUCT_HP and k != "f"
    }


def _ctx(g: dict, base: dict, attrs: dict) -> dict:
    ins = [n for n in g["nodes"] if n["kind"] == "input"]
    outs = [n for n in g["nodes"] if n["kind"] == "output"]
    si = ins[0]["out"][0] if ins else []
    so = outs[0]["out"][0] if outs else []
    ns = [n["n"] for n in g["nodes"] if "n" in n]
    return {
        **base,
        "C": so[1] if len(so) > 1 else None,
        "Ci": si[1] if len(si) > 1 else None,
        "H": si[2] if len(si) == 4 else None,
        "W": si[3] if len(si) == 4 else None,
        "n": max(ns) if ns else None,
        "attrs": attrs,
    }


def process(g: dict, owner: str, cid: str, base: dict, attrs: dict) -> str:
    """Post-order: key children, collapse repeats, key this graph by structure and record it as a sample."""
    g = _group_calls(g)
    g = _group_calls(g, "seq") if len(g["nodes"]) > 16 else g  # only big graphs, e.g. Detect.forward_head
    for n in g["nodes"]:
        n.pop("seq", None)
        if "g" in n:
            n["g"] = process(n["g"], n["type"], cid, base, _attrs(n))
    g = _collapse(g)
    sig = []
    for n in g["nodes"]:
        n["hps"], n["hpn"] = _hp_split(n.pop("hp", {}))
        sig.append([n["kind"], n["type"], n.get("name", ""), n["hps"], n.get("g", ""), "n" in n])
    key = hashlib.sha1(json.dumps([owner, sig, sorted(e[:4] for e in g["edges"])]).encode()).hexdigest()[:10]
    topo = STRUCT.get(key)
    if topo is None:
        STRUCT[key] = {
            "owner": owner,
            "nodes": [
                {
                    k: v
                    for k, v in {
                        "kind": n["kind"],
                        "type": n["type"],
                        "name": n.get("name"),
                        "hp": n["hps"] or None,
                        "g": n.get("g"),
                        "group": "n" in n or None,
                    }.items()
                    if v is not None
                }
                for n in g["nodes"]
            ],
            "edges": g["edges"],
            "used": set(),
        }
    else:  # a repeat feeding a node from every member shows up only when n > 1
        for e, e2 in zip(sorted(topo["edges"]), sorted(g["edges"])):
            e[4] = e[4] or e2[4]
    STRUCT[key]["used"].add(cid)
    SAMPLES[(key, cid)].append((g, _ctx(g, base, attrs)))
    return key


# --------------------------------------------------------------------------------------------------------------------
# symbolic fitting
# --------------------------------------------------------------------------------------------------------------------
def _nice(x: float, maxden: int = 8) -> Fraction | None:
    if x <= 0:
        return None
    f = Fraction(x).limit_denominator(maxden)
    return f if abs(float(f) - x) < 1e-9 else None


def _coef(f: Fraction, sym: str, spatial: bool = False) -> str:
    if f == 1:
        return sym
    if spatial or f.denominator not in (1, 2, 4, 8):
        return f"{sym}/{f.denominator}" if f.numerator == 1 else f"{f.numerator}{sym}/{f.denominator}"
    return f"{float(f):g}{sym}"


BASES = {
    "c_out": lambda c: c["C"],
    "c_in": lambda c: c["Ci"],
    "h": lambda c: c["H"],
    "w": lambda c: c["W"],
    "h·w": lambda c: c["H"] * c["W"] if c["H"] else None,
}


def _base(sym: str, c: dict):
    """Value of a base symbol in one sample; '@name' reads the owning module's attribute."""
    return c["attrs"].get(sym[1:]) if sym.startswith("@") else BASES[sym](c)


def _fit_ratio(vals, ctxs, sym):
    bs = [_base(sym, c) for c in ctxs]
    if not all(bs):
        return None
    f = _nice(vals[0] / bs[0])
    if (
        f
        and (sym == "h·w" or Fraction(1, 8) <= f <= 8)
        and all(v * f.denominator == b * f.numerator for v, b in zip(vals, bs))
    ):
        return _coef(f, sym.lstrip("@"), spatial=sym in {"h", "w", "h·w"})
    return None


def _fit_nlin(vals, ctxs, sym="c_out"):
    """Value as (a + b·n)·base, e.g. the C2f concat 0.5(n+2)c_out."""
    label = sym.lstrip("@")
    pts = [(c["n"], v / b) for v, c in zip(vals, ctxs) if c["n"] and (b := _base(sym, c))]
    if len(pts) != len(vals) or len({p[0] for p in pts}) < 2:
        return None
    pts = sorted(set(pts))
    (n1, r1), (n2, r2) = pts[0], next(p for p in pts if p[0] != pts[0][0])
    b = Fraction(r2 - r1).limit_denominator(8) / (n2 - n1)
    a = Fraction(r1).limit_denominator(8) - b * n1
    if b <= 0 or a < 0 or not all(abs(float(a + b * n) - r) < 1e-9 for n, r in pts):
        return None
    if a == 0:
        return f"{_coef(b, 'n')}·{label}"
    k = a / b
    if k.denominator == 1:
        return f"(n+{k}){label}" if b == 1 else f"{float(b):g}(n+{k}){label}"
    return f"({float(a):g}+{float(b):g}n){label}"


def fit_dim(vals: list[int], ctxs: list[dict], kind: str) -> str:
    """Express one tensor dim across samples, preferring block-relative forms."""
    const = str(vals[0]) if len(set(vals)) == 1 else None
    named = lambda key, label: (
        label if all(v == c[key] * (4 if label == "4·reg_max" else 1) for v, c in zip(vals, ctxs)) else None
    )
    attrs = [f"@{a}" for a in ctxs[0]["attrs"] if all(a in c["attrs"] for c in ctxs)]
    owner = [x for a in attrs for x in (a, "nlin" + a)]  # block attributes, e.g. C3k2.c = e·c_out
    order = {
        "C": ["c_out", "c_in", "nlin", "nc", "4reg", *owner, "const", "h·w"],
        "Cin": ["c_in", "c_out", "nlin", "nc", "4reg", *owner, "const"],
        "H": ["h", "const"],
        "W": ["w", "const"],
        "X": ["const", "c_out", "c_in", "h·w", "h", "w", "nlin", "nc", "4reg", *owner],
    }[kind]
    for o in order:
        r = (
            const
            if o == "const"
            else _fit_nlin(vals, ctxs, o[4:] or "c_out")
            if o.startswith("nlin")
            else named("nc", "nc")
            if o == "nc"
            else named("reg_max", "4·reg_max")
            if o == "4reg"
            else _fit_ratio(vals, ctxs, o)
        )
        if r:
            return r
    u = sorted(set(vals))
    return "/".join(map(str, u)) if len(u) <= 4 else f"{u[0]}…{u[-1]}"


def fit_shape(shapes: list[list[int]], ctxs: list[dict], is_input: bool) -> str:
    """Symbolic block-relative shape of one tensor across scales, e.g. h×w×0.5c_out."""
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        return "·"
    dims = []
    for j in range(rank):
        kind = "X"
        if rank >= 3 and j == 1:
            kind = "Cin" if is_input else "C"
        elif rank == 4 and j in (2, 3):
            kind = "HW"[j - 2]
        dims.append(fit_dim([s[j] for s in shapes], ctxs, kind))
    if rank >= 2 and dims[0] == "1":
        dims = dims[1:]  # batch
        if rank == 4:
            return f"{dims[1]}×{dims[2]}×{dims[0]}"
    return "×".join(dims) if dims else "scalar"


def fit_struct(key: str, cid: str) -> dict:
    """Symbolic shapes and args of every node and edge of one block structure, fitted across its samples."""
    samples = SAMPLES[(key, cid)]
    graphs, ctxs = [s[0] for s in samples], [s[1] for s in samples]
    out, hp, args = [], [], []
    for j, n in enumerate(graphs[0]["nodes"]):
        nodes = [g["nodes"][j] for g in graphs]
        out.append(
            [fit_shape([m["out"][k] for m in nodes], ctxs, n["kind"] == "input") for k in range(len(n.get("out", [])))]
        )
        h = {}
        for k in n["hpn"]:
            vals = [m["hpn"].get(k) for m in nodes]
            if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                h[k] = fit_dim(vals, ctxs, "X")
            elif len(set(vals)) == 1:
                h[k] = f"{vals[0]:g}" if isinstance(vals[0], float) else str(vals[0])
        if "n" in n:
            h["n"] = "/".join(map(str, sorted({m["n"] for m in nodes})))
        hp.append(h or None)
        a = Counter(m.get("arg") for m in nodes)
        args.append(None if not n.get("arg") else (a.most_common(1)[0][0] + (" …" if len(a) > 1 else "")))
    return {"out": out, "hp": hp, "arg": args}


def fit_top(vals: list[int], scales: dict) -> str:
    """Express a top-level channel count with the scale constants w (width) and r (max_channels ratio)."""
    if len(set(vals)) == 1:
        return str(vals[0])
    ws = [s["w"] for s in scales.values()]
    rs = [s["r"] for s in scales.values()]
    a = vals[0] / ws[0]
    if all(abs(v - a * w) < 1e-6 for v, w in zip(vals, ws)):
        return f"{a:g}×w"
    b = vals[0] / (ws[0] * rs[0])
    if all(abs(v - b * w * r) < 1e-6 for v, w, r in zip(vals, ws, rs)):
        return f"{b:g}×w×r"
    for i in range(1, len(vals)):  # v/w = a + b·r
        if rs[i] != rs[0]:
            b = (vals[i] / ws[i] - vals[0] / ws[0]) / (rs[i] - rs[0])
            a = vals[0] / ws[0] - b * rs[0]
            if all(abs(v - w * (a + b * r)) < 1e-6 for v, w, r in zip(vals, ws, rs)):
                return f"{a:g}×w×(1+r)" if abs(a - b) < 1e-9 else f"({a:g}+{b:g}r)×w"
            break
    return "/".join(map(str, vals))


def fit_arg(name: str, vals: list, yaml_n: int, scales: dict) -> str:
    """Symbolic form of a layer argument across scales, e.g. n=3×d or 512×w."""
    if name == "n" and yaml_n > 1 and len(set(vals)) > 1:
        return f"{yaml_n}×d"
    if len(set(map(repr, vals))) == 1:
        return repr(vals[0]) if isinstance(vals[0], str) else str(vals[0])
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return fit_top(vals, scales)
    groups = defaultdict(list)
    for s, v in zip(scales, vals):
        groups[str(v)].append(s)
    return " · ".join(f"{v} ({','.join(ss)})" for v, ss in groups.items())


# --------------------------------------------------------------------------------------------------------------------
# block catalog
# --------------------------------------------------------------------------------------------------------------------
def structural_families(blocks: dict, configs: list[dict]) -> dict[str, str]:
    """Role of every block type from data alone, so new block types are classified without being named.

    head: what a config ends with. down: a top-level layer that halves the resolution (and is not a plain conv). pool:
    max/avg-pools concatenated (SPP style). attn: Q·K matmul, MultiheadAttention, sampled attention, or every variant
    built from attention children. csp: split/concat aggregation or repeated units. unit: a residual add.
    """
    feats = defaultdict(list)  # owner -> per-structure (ops, child module types, has repeats)
    for b in blocks.values():
        ops = {n["type"] for n in b["nodes"] if n["kind"] == "op"}
        kids = {n["type"] for n in b["nodes"] if n["kind"] == "module"}
        feats[b["owner"]].append((ops, kids, any(n.get("group") for n in b["nodes"])))
    fam = {c["layers"][-1]["cls"]: "head" for c in configs}
    for c in configs:
        for layer in c["layers"]:
            src = layer["f"] if isinstance(layer["f"], int) else layer["f"][0]
            prev = c["layers"][layer["i"] + src] if src < 0 else c["layers"][src]
            if (
                layer["hw"]
                and prev["hw"]
                and layer["hw"][0] * 2 == prev["hw"][0]
                and "conv" not in layer["cls"].lower()
            ):
                fam.setdefault(layer["cls"], "down")
    for owner, fs in feats.items():
        if owner in fam:
            continue
        ops = set().union(*(f[0] for f in fs))
        kids = set().union(*(f[1] for f in fs))
        if kids & {"MaxPool2d", "AvgPool2d"} and "cat" in ops:
            fam[owner] = "pool"
        elif "matmul" in ops or "MultiheadAttention" in kids or {"grid_sample", "softmax"} <= ops:
            fam[owner] = "attn"
        elif "cat" in ops and (ops & {"chunk", "split"} or any(f[2] for f in fs)):
            fam[owner] = "csp"
        elif "add" in ops:
            fam[owner] = "unit"
    for _ in range(4):  # blocks whose every variant is built from attention blocks are attention blocks too
        for owner, fs in feats.items():
            if fam.get(owner) not in {"head", "down", "pool", "attn"} and all(
                any(fam.get(k) == "attn" for k in f[1]) for f in fs
            ):
                fam[owner] = "attn"
    return fam


KNOWN_FAMILY = {t for _, members in FAMILIES for t in members}


def family(name: str) -> str:
    """Initial color family of a type: its torch-primitive family, else conv or csp until refined from structure."""
    for fam, members in FAMILIES:
        if name in members:
            return fam
    return "conv" if "conv" in name.lower() else "csp"  # refined from structure in structural_families()


def class_info(cls: type) -> dict:
    """Docstring summary, source file and line, and __init__/forward source of a block class."""
    doc = (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
    info = {"doc": doc, "family": family(cls.__name__), "module": cls.__module__}
    if not cls.__module__.startswith("ultralytics"):
        return info
    try:
        file = Path(inspect.getsourcefile(cls)).resolve().relative_to(REPO)
        _, line = inspect.getsourcelines(cls)
        info.update(file=str(file), line=line)
        parts = []
        for meth in ("__init__", "forward"):
            fn = cls.__dict__.get(meth)
            if fn is not None:
                code = inspect.getsource(fn)
                code = re.sub(r'(\n\s+)"""[\s\S]*?"""', "", code, count=1)  # strip docstring
                parts.append(inspect.cleandoc("\n" + code) if code.startswith(" ") else code)
        if not parts:  # inherited behavior, e.g. C3k2 -> C2f.forward
            parts.append(f"# inherits forward() from {cls.__mro__[1].__name__}")
        info["src"] = f"class {cls.__name__}({', '.join(b.__name__ for b in cls.__bases__)}):\n" + "\n\n".join(
            "    " + p.replace("\n", "\n    ") for p in parts
        )
    except (TypeError, OSError, ValueError):
        pass
    return info


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run(only: str | None):
    """Build and trace every discovered config and return the page data."""
    LOGGER.setLevel(logging.INFO)
    configs, classes, block_first = [], {}, {}
    x = torch.zeros(1, 3, IMGSZ, IMGSZ)

    skipped = []
    for version, path in discover():
        stem = path.stem
        if only and not re.search(only, stem):
            continue
        raw = YAML.load(path)
        scale_defs = raw.get("scales") or {}
        scales_list = list(scale_defs) if scale_defs else [None]
        base_max = min(v[2] for v in scale_defs.values()) if scale_defs else None
        cfg = {
            "id": stem,
            "version": version,
            "file": str(path.relative_to(REPO)),
            "yaml": path.read_text(),
            "nb": len(raw["backbone"]),
            "nc": raw.get("nc", 80),
            "scales": {},
            "layers": [],
        }
        per_scale = {}
        for s in scales_list:
            t0 = time.time()
            try:
                model, rows = build_model(path, s)
                det = model.model[-1]
                base = {"nc": getattr(det, "nc", raw.get("nc", 80)), "reg_max": getattr(det, "reg_max", 16)}
                traced, stats = {}, {}
                for phase in PHASES:
                    m = model if phase == "train" else copy.deepcopy(model)  # fuse() edits the model in place
                    if phase == "train":
                        m.train()
                    else:  # exactly what predict() / export() run by default
                        m.eval()
                        m.fuse(verbose=False)
                        m(x)  # warm-up so cached anchors are constants in the trace
                    tracer = GraphTracer()
                    root = tracer.trace(m, x)
                    FNS.update(tracer.fns)
                    nodes = {n.get("name"): n for n in root["nodes"] if n["kind"] == "module"}
                    keys = {}
                    for i in range(len(m.model)):
                        n = nodes[f"model.{i}"]
                        if "g" in n:
                            keys[i] = process(n["g"], n["type"], stem, base, _attrs(n))
                    stats[phase] = {
                        # modules that actually ran, so a branch kept in the model but not called counts as gone
                        "names": {n: type(mod).__name__ for n, mod in m.named_modules() if n in tracer.called},
                        "calls": sorted(tracer.fns),
                        "ops": dict(tracer.ops),
                        "params": sum(p.numel() for p in m.parameters()),
                        "gflops": round(2 * tracer.macs / 1e9, 1),
                        "registered": {n: type(mod).__name__ for n, mod in m.named_modules()},
                        "layer_params": [sum(p.numel() for p in layer.parameters()) for layer in m.model],
                    }
                    traced[phase] = (nodes, keys)
                    for mod in m.modules():
                        classes.setdefault(type(mod).__name__, type(mod))
                        block_first.setdefault(type(mod).__name__, version)
            except Exception as e:  # keep building the other configs, report this one
                print(f"  ! {stem}{s or ''}: {type(e).__name__}: {e}")
                skipped.append(
                    {
                        "id": stem,
                        "version": version,
                        "file": str(path.relative_to(REPO)),
                        "reason": f"{type(e).__name__}: {str(e)[:160]}",
                    }
                )
                break
            d, w, mx = scale_defs[s] if s else (1.0, 1.0, None)
            cfg["scales"][s or ""] = {
                "d": d,
                "w": w,
                "max": mx,
                "r": (mx / base_max) if mx else 1.0,
                **{ph: {"params": st["params"], "gflops": st["gflops"]} for ph, st in stats.items()},
            }
            if "phases" not in cfg:
                cfg["phase_diff"] = phase_diff(stats)
                cfg["train_only"] = train_only(stats)
                cfg["phases"] = phase_notes(stats, cfg["phase_diff"])
                cfg["phase_list"] = list(stats)
            per_scale[s or ""] = (model, rows, traced, stats)
            print(
                f"  {stem:18s} {s or '-':2s} train {stats['train']['gflops']:6.1f} / deploy "
                f"{stats['deploy']['gflops']:6.1f} GFLOPs  {time.time() - t0:5.1f}s"
            )
        if not per_scale:
            continue
        sc = cfg["scales"]
        first_model = next(iter(per_scale.values()))[0]
        entries = raw["backbone"] + raw["head"]
        for i, layer in enumerate(first_model.model):
            f, yaml_n, mod_name, yaml_args = entries[i]
            rows = [per_scale[s][1].get(i, {}) for s in sc]
            outs = [per_scale[s][2]["train"][0][f"model.{i}"]["out"][0] for s in sc]
            cls = classes.get(mod_name.split(".")[-1]) or type(layer)
            try:
                names = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
            except (TypeError, ValueError):
                names = []
            args_by_scale = [r.get("args") if isinstance(r.get("args"), list) else [] for r in rows]
            labeled = []
            for j in range(max(map(len, args_by_scale), default=0)):
                vals = [a[j] if j < len(a) else None for a in args_by_scale]
                nm = names[j] if j < len(names) else f"arg{j}"
                labeled.append([nm, fit_arg(nm, vals, yaml_n, sc)])
            variants = {}
            for ph in cfg["phase_list"]:
                groups = defaultdict(list)
                for s in sc:
                    k = per_scale[s][2][ph][1].get(i)
                    if k:
                        groups[k].append(s)
                variants[ph] = [[k, v] for k, v in groups.items()]
            shape = outs[0]
            ch = [o[1] for o in outs] if len(shape) == 4 else None
            cfg["layers"].append(
                {
                    "i": i,
                    "f": f,
                    "type": mod_name.replace("nn.", ""),
                    "cls": type(layer).__name__,
                    "section": "backbone" if i < cfg["nb"] else "head",
                    "yaml": json.dumps([f, yaml_n, mod_name, yaml_args]),
                    "n": [r.get("n", 1) for r in rows],
                    "n_sym": f"{yaml_n}×d" if yaml_n > 1 and len(sc) > 1 else (str(yaml_n) if yaml_n > 1 else ""),
                    "args": labeled,
                    "hw": [shape[2], shape[3]] if len(shape) == 4 else None,
                    "c_sym": fit_top(ch, sc) if ch else None,
                    "out": [o if len(o) != 4 else [o[2], o[3], o[1]] for o in outs],
                    "params": {ph: [per_scale[s][3][ph]["layer_params"][i] for s in sc] for ph in cfg["phase_list"]},
                    "variants": variants,
                    "hp": _hp_split(per_scale[next(iter(sc))][2]["train"][0][f"model.{i}"].get("hp", {}))[0],
                }
            )
        configs.append(cfg)
        del per_scale
    if not configs:
        raise SystemExit(f"No detection config was built{f' for --only {only!r}' if only else ''}.")

    blocks = {}
    for key, topo in STRUCT.items():
        syms, index = [], {}
        for cid in sorted(topo["used"]):
            s = fit_struct(key, cid)
            blob = json.dumps(s, sort_keys=True)
            if blob not in index:
                index[blob] = len(syms)
                syms.append(s)
            topo.setdefault("sym", {})[cid] = index[blob]
        topo["syms"] = syms
        topo["used"] = sorted(topo["used"])
        blocks[key] = topo

    order = sorted(
        {c["version"] for c in configs} | {k["version"] for k in skipped}, key=lambda v: (version_rank(v), v)
    )
    used_by = defaultdict(set)
    for key, b in blocks.items():
        for cid in b["used"]:
            used_by[b["owner"]].add(cid)
        for n in b["nodes"]:
            if n["kind"] == "module":
                used_by[n["type"]].update(b["used"])
    cfg_version = {c["id"]: c["version"] for c in configs}
    for c in configs:
        for layer in c["layers"]:
            used_by[layer["cls"]].add(c["id"])
    derived = structural_families(blocks, configs)
    catalog = {}
    for name, cls in classes.items():
        if name in {"Sequential", "ModuleList"} or issubclass(cls, BaseModel):
            continue
        versions = sorted({cfg_version[c] for c in used_by.get(name, ())}, key=order.index)
        info = class_info(cls)
        if name not in KNOWN_FAMILY and "conv" not in name.lower():
            info["family"] = derived.get(name, info["family"])
        info["versions"] = versions
        info["first"] = versions[0] if versions else block_first.get(name)
        info["configs"] = sorted(used_by.get(name, ()))
        info["structs"] = sorted(
            (k for k, b in blocks.items() if b["owner"] == name), key=lambda k: -len(blocks[k]["used"])
        )
        catalog[name] = info
    for qual, f in FNS.items():  # helper methods, e.g. Detect.forward_head(): shown as their own openable blocks
        name = qual + "()"
        if name not in used_by:
            continue
        try:
            file = str(Path(f["file"]).resolve().relative_to(REPO))
        except ValueError:
            file = None
        code = re.sub(r'(\n\s+)"""[\s\S]*?"""', "", f["src"], count=1)
        catalog[name] = {
            "doc": (f["doc"] or "").split("\n\n")[0].replace("\n", " "),
            "family": "fn",
            "module": "",
            "file": file,
            "line": f["line"],
            "src": inspect.cleandoc("\n" + code) if code.startswith(" ") else code,
            "versions": sorted({cfg_version[c] for c in used_by[name]}, key=order.index),
            "configs": sorted(used_by[name]),
            "structs": [k for k, b in blocks.items() if b["owner"] == name],
        }
        catalog[name]["first"] = catalog[name]["versions"][0]
    # existing block types whose internal wiring first appears in a version (e.g. a new flag), found from struct keys
    rewired = defaultdict(set)
    for b in blocks.values():
        first = min((cfg_version[c] for c in b["used"]), key=order.index)
        c = catalog.get(b["owner"])
        if c and c.get("file") and c["family"] != "fn" and order.index(c["first"]) < order.index(first):
            rewired[first].add(b["owner"])
    rewired = {v: sorted(o) for v, o in rewired.items()}
    return {
        "versions": [
            {
                "name": v,
                "note": version_note(v, [c for c in configs if c["version"] == v], catalog, rewired),
                "configs": [c["id"] for c in configs if c["version"] == v],
            }
            for v in order
        ],
        "skipped": skipped,
        "default": min(  # base config of the newest numbered version, e.g. yolo26 rather than yolo26-p6
            (c for c in configs if c["version"] == max((c["version"] for c in configs), key=version_rank_finite)),
            key=lambda c: len(c["id"]),
        )["id"],
        "configs": configs,
        "blocks": blocks,
        "catalog": catalog,
        "imgsz": IMGSZ,
        "ultralytics": __import__("ultralytics").__version__,
        "torch": torch.__version__,
    }


def main():
    """Command-line entry point: build the data and write the HTML page (and optionally the JSON)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="regex on config stems, e.g. 'yolov8$|yolo11'")
    ap.add_argument("--out", default=str(HERE / "yolo_architecture_explorer.html"))
    ap.add_argument("--json", help="also write the extracted data as JSON")
    opt = ap.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 4))
    data = run(opt.only)
    blob = json.dumps(data, separators=(",", ":"), default=list)
    if opt.json:
        Path(opt.json).write_text(blob)
    data_js = blob.replace("</", "<\\/")
    html, n = re.subn(r"/\*__DATA__\*/\s*null", lambda _: data_js, (HERE / "template.html").read_text(), count=1)
    if n != 1:
        raise SystemExit("template.html has no '/*__DATA__*/ null' placeholder")
    Path(opt.out).write_text(html)
    print(
        f"wrote {opt.out} ({len(html) / 1e6:.2f} MB, {len(data['configs'])} configs, {len(data['blocks'])} block structures)"
    )


if __name__ == "__main__":
    main()
