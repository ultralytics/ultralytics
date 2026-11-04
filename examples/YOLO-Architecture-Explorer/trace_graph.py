# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Record a hierarchical dataflow graph of a real forward pass.

Global module hooks open one graph scope per composite module call, and a TorchFunctionMode records every tensor op
(chunk, cat, add, view, matmul, ...) issued directly by that module's forward. Leaf modules (torch.nn built-ins and
custom modules without children) become single nodes. Tensor provenance is tracked by id() with weakref finalizers, so
activations are freed as usual and ids are never confused after reuse.
"""

from __future__ import annotations

import inspect
import itertools
import os
import re
import sys
import weakref

import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_flatten_with_path

OP_SYMBOLS = {
    "__add__": "add",
    "__radd__": "add",
    "__iadd__": "add_",
    "__mul__": "mul",
    "__rmul__": "mul",
    "__sub__": "sub",
    "__rsub__": "sub",
    "__truediv__": "div",
    "__rtruediv__": "div",
    "__matmul__": "matmul",
    "__getitem__": "slice",
    "__floordiv__": "floordiv",
    "__mod__": "mod",
    "__neg__": "neg",
}
_INTERNAL = (os.path.dirname(torch.__file__), os.path.abspath(__file__))


def _qualname(frame) -> tuple[str, object]:
    """'Class.method' of a frame, naming the class that defines the method (e.g. Detect.forward_head for v10Detect)."""
    code, obj = frame.f_code, frame.f_locals.get("self")
    if obj is not None:
        for c in type(obj).__mro__:
            fn = c.__dict__.get(code.co_name)
            if getattr(getattr(fn, "__func__", fn), "__code__", None) is code:
                return f"{c.__name__}.{code.co_name}", fn
    return code.co_name, frame.f_globals.get(code.co_name)


SKIP_OPS = {"__get__", "__set__", "size", "dim", "numel", "item", "tolist", "__len__", "__bool__", "__format__"}


def _tensors(obj) -> list[torch.Tensor]:
    """Flatten nested args/outputs (lists, tuples, dicts) into the tensors they contain."""
    return [t for t in tree_flatten(obj)[0] if isinstance(t, torch.Tensor)]


def _path_name(path) -> str:
    """'1.boxes' for the pytree path of out[1]['boxes'] in a module's return value."""
    return ".".join(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))) for k in path)


def _shape(t: torch.Tensor) -> list[int]:
    return list(t.shape)


def _arg_summary(args, kwargs) -> str:
    """Short text for non-tensor op arguments, e.g. 'dim=1' or '(2, 1)'."""

    def fmt(a):
        if isinstance(a, torch.Tensor):
            return None
        if isinstance(a, (list, tuple)):
            inner = [fmt(x) for x in a]
            if all(x is None for x in inner):
                return None
            return "(" + ", ".join("·" if x is None else x for x in inner) + ")"
        if isinstance(a, slice):
            return ":".join("" if v is None else str(v) for v in (a.start, a.stop, a.step)).rstrip(":") or ":"
        if a is Ellipsis:
            return "..."
        if isinstance(a, torch.dtype):
            return str(a).replace("torch.", "")
        if isinstance(a, float):
            return f"{a:.4g}"
        return repr(a) if isinstance(a, (int, bool, str, type(None))) else type(a).__name__

    parts = [x for x in (fmt(a) for a in args[1:]) if x is not None]
    parts += [f"{k}={v}" for k, v in ((k, fmt(v)) for k, v in kwargs.items()) if v is not None]
    s = ", ".join(parts)
    return s if len(s) <= 48 else s[:45] + "..."


def _hparams(mod: nn.Module) -> dict:
    """Readable hyper-parameters of a module (kernel/stride/padding, scalar attributes)."""
    hp = {}
    conv = mod if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)) else getattr(mod, "conv", None)
    if isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)):
        k, s, p, d = conv.kernel_size[0], conv.stride[0], conv.padding, conv.dilation[0]
        hp.update(k=k, s=s, p=p[0] if isinstance(p, tuple) else p)
        if conv.groups > 1:
            hp["g"] = conv.groups
        if d > 1:
            hp["d"] = d
        if conv is not mod and hasattr(mod, "act"):
            hp["act"] = type(mod.act).__name__
    elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
        hp.update(k=mod.kernel_size, s=mod.stride, p=mod.padding)
    elif isinstance(mod, nn.Upsample):
        hp.update(scale=mod.scale_factor, mode=mod.mode)
    elif isinstance(mod, nn.Linear):
        hp.update(in_f=mod.in_features, out_f=mod.out_features)
    if type(mod).__module__.startswith("torch."):
        return hp
    for k, v in vars(mod).items():
        if k.startswith("_") or k in {"training", "dynamic", "export", "format", "legacy", "i", "np", "type"}:
            continue
        if isinstance(v, (bool, int, float, str)) and k not in hp and len(hp) < 10:
            hp[k] = v
    return hp


class GraphTracer(TorchFunctionMode):
    """Trace a module forward into a table of deduplicated, nested graphs."""

    def __init__(self):
        """Create an empty tracer; trace() fills it."""
        super().__init__()
        self.macs = 0  # multiply-accumulates of conv/linear leaves
        self.ops: dict[str, int] = {}  # tensor-op counts outside leaf modules
        self.uid = itertools.count()
        self.prod: dict[int, dict[int, tuple[int, int]]] = {}  # id(tensor) -> {graph uid: (node, out index)}
        self.stack: list[tuple] = []
        self.names: dict[int, str] = {}
        self.mods: dict[str, nn.Module] = {}
        self.fns: dict[str, dict] = {}  # helper methods that group ops, e.g. Detect.forward_head -> source
        self.called: set[str] = set()  # paths of every module that ran and of its parents, e.g. a ModuleList

    # --- graph helpers -------------------------------------------------------------------------------------------
    def _new_graph(self, mod: nn.Module | None, path: str) -> dict:
        g = {"uid": next(self.uid), "nodes": [], "edges": [], "path": path, "consts": {}}
        if mod is not None:  # name tensors held as attributes so constants read as e.g. 'anchors'
            g["attrs"] = {id(v): k for k, v in vars(mod).items() if isinstance(v, torch.Tensor)}
            g["attrs"].update({id(v): k for k, v in mod.named_parameters(recurse=False)})
            g["attrs"].update({id(v): k for k, v in mod.named_buffers(recurse=False)})
        return g

    @staticmethod
    def _add(g: dict, **node) -> int:
        node["id"] = len(g["nodes"])
        g["nodes"].append(node)
        return node["id"]

    def _register(self, t: torch.Tensor, g: dict, nid: int, idx: int):
        k = id(t)
        d = self.prod.get(k)
        if d is None:
            d = self.prod[k] = {}
            weakref.finalize(t, self.prod.pop, k, None)
        d[g["uid"]] = (nid, idx)

    def _link(self, g: dict, tensors: list[torch.Tensor], dst: int):
        for slot, t in enumerate(tensors):
            src = self.prod.get(id(t), {}).get(g["uid"])
            if src is None:  # parameter, buffer or tensor created outside this scope
                c = g["consts"].get(id(t))
                if c is None:
                    label = g.get("attrs", {}).get(id(t), "tensor")
                    c = g["consts"][id(t)] = self._add(g, kind="const", type=label, out=[_shape(t)])
                src = (c, 0)
            g["edges"].append([src[0], src[1], dst, slot])

    def _group(self, g: dict) -> list | None:
        """Helper method of the scope module that issued the current call, e.g. Detect.forward_head (None if forward)."""
        mod = g.get("mod")
        if mod is None:
            return None
        chain, f = [], sys._getframe(2)
        while f is not None and not (f.f_code.co_name == "forward" and f.f_locals.get("self") is mod):
            if not f.f_code.co_filename.startswith(_INTERNAL) and not f.f_code.co_name.startswith("<"):
                chain.append(f)
            f = f.f_back
        if f is None or not chain:
            return None
        top = chain[-1]  # outermost helper called from forward()
        if g.get("gframe") is not top:
            g["gframe"], g["ginst"] = top, g.get("ginst", 0) + 1
        qual, fn = _qualname(top)
        if qual not in self.fns:
            try:
                src = inspect.getsource(fn or top.f_code)
                file, line = inspect.getsourcefile(fn or top.f_code), inspect.getsourcelines(fn or top.f_code)[1]
            except (OSError, TypeError):
                src, file, line = "", "", 0
            self.fns[qual] = {"src": src, "file": file, "line": line, "doc": inspect.getdoc(fn) if fn else ""}
        return [qual, g["ginst"]]

    def _cur(self) -> dict:
        for fr in reversed(self.stack):
            if fr[0] == "scope":
                return fr[3]
        raise RuntimeError("no open scope")

    @staticmethod
    def _finalize(g: dict) -> dict:
        """Drop trace-only bookkeeping from a finished graph."""
        for k in ("consts", "attrs", "uid", "path", "gframe", "ginst", "mod"):
            g.pop(k, None)
        return g

    # --- hooks ---------------------------------------------------------------------------------------------------
    def _rel(self, mod: nn.Module) -> str:
        full = self.names.get(id(mod), "")
        for fr in reversed(self.stack):
            if fr[0] == "scope":
                base = fr[4]
                if base and full == base:
                    return ""
                if not base or full.startswith(base + "."):
                    return full[len(base) + 1 :] if base else full
                # shared module (e.g. Conv.default_act) registered under its first owner: name it inside this scope
                return next((k for k, m in self.mods[base].named_modules() if m is mod), full)
        return full

    @staticmethod
    def _is_leaf(mod: nn.Module) -> bool:
        if isinstance(mod, (nn.Sequential, nn.ModuleList)):
            return False  # a repeated top-level layer, e.g. YOLOv6 'Conv x n'
        return type(mod).__module__.startswith("torch.") or not any(True for _ in mod.children())

    def pre_hook(self, mod, args):
        """Open a node for a module call, and a nested scope for it unless it is a torch leaf."""
        if not self.stack:
            return
        name = self.names.get(id(mod), "")
        while name and name not in self.called:  # a parent is in use when any child runs
            self.called.add(name)
            name = name.rpartition(".")[0]
        top = self.stack[-1][0]
        if top in {"leaf", "skip"}:
            self.stack.append(("skip",))
            return
        full = self.names.get(id(mod), "")
        if type(mod) is nn.Sequential and not re.fullmatch(r"model\.\d+", full):
            self.stack.append(("transparent",))
            return
        g = self._cur()
        ins = _tensors(args)
        node = {
            "kind": "module",
            "type": type(mod).__name__,
            "name": self._rel(mod),
            "params": sum(p.numel() for p in mod.parameters()),
            "hp": _hparams(mod),
        }
        parent, _, last = full.rpartition(".")
        pm = self.mods.get(parent)
        if last.isdigit() and isinstance(pm, (nn.Sequential, nn.ModuleList)) and len({type(c) for c in pm}) == 1:
            node["rep"] = self._rel(pm)  # homogeneous repeat container, e.g. C2f.m
        grp = self._group(g)
        if grp:
            node["grp"] = grp
        base = next(fr[4] for fr in reversed(self.stack) if fr[0] == "scope") or ""
        if full.startswith(base + ".") or not base:
            acc = base
            for part in full[len(base) + 1 if base else 0 :].split(".")[:-1]:
                acc = f"{acc}.{part}" if acc else part
                m = self.mods.get(acc)
                if type(m) is nn.Sequential and len({type(c) for c in m}) > 1:
                    node["seq"] = acc[len(base) + 1 :] if base else acc  # outermost mixed Sequential, e.g. Detect.cv2.0
                    break
        nid = self._add(g, **node)
        self._link(g, ins, nid)
        if self._is_leaf(mod):
            self.stack.append(("leaf", g, nid))
            return
        child = self._new_graph(mod, full)
        child["mod"] = mod
        for i, t in enumerate(ins):
            self._register(t, child, self._add(child, kind="input", type=f"in{i}", out=[_shape(t)]), 0)
        self.stack.append(("scope", g, nid, child, full))

    def post_hook(self, mod, args, out):
        """Close the module call: record its outputs, MACs and provenance in the parent graph."""
        if not self.stack:
            return
        fr = self.stack.pop()
        if fr[0] in {"skip", "transparent"}:
            return
        g, nid = fr[1], fr[2]
        outs = _tensors(out)
        if fr[0] == "leaf" and outs:
            if isinstance(mod, nn.Conv2d):
                self.macs += outs[0].numel() * mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
            elif isinstance(mod, nn.ConvTranspose2d):
                self.macs += (
                    _tensors(args)[0].numel() * mod.out_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
                )
            elif isinstance(mod, nn.Linear):
                self.macs += outs[0].numel() * mod.in_features
        if fr[0] == "scope":
            child = fr[3]
            names = [
                _path_name(path) for path, leaf in tree_flatten_with_path(out)[0] if isinstance(leaf, torch.Tensor)
            ]
            for i, t in enumerate(outs):
                src = self.prod.get(id(t), {}).get(child["uid"])
                if src is not None and child["nodes"][src[0]]["kind"] == "input" and len(outs) > 1:
                    continue  # pass-through of an input (e.g. Detect returning its feats), not a real output
                oid = self._add(child, kind="output", type=f"out{i}", out=[_shape(t)])
                if len(outs) > 1 and names[i]:  # name outputs after the returned structure, e.g. 'one2many.boxes'
                    child["nodes"][oid]["name"] = names[i]
                self._link(child, [t], oid)
            g["nodes"][nid]["g"] = self._finalize(child)
        g["nodes"][nid]["out"] = [_shape(t) for t in outs]
        for i, t in enumerate(outs):
            self._register(t, g, nid, i)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Record a tensor op called directly inside a module scope as an op node."""
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not self.stack or self.stack[-1][0] in {"leaf", "skip"}:
            return out
        name = getattr(func, "__name__", str(func))
        outs = _tensors(out)
        if not outs or name in SKIP_OPS:
            return out
        g = self._cur()
        op = {"kind": "op", "type": OP_SYMBOLS.get(name, name), "arg": _arg_summary(args, kwargs), "out": []}
        self.ops[op["type"]] = self.ops.get(op["type"], 0) + 1
        grp = self._group(g)
        if grp:
            op["grp"] = grp
        nid = self._add(g, **op)
        self._link(g, _tensors((args, kwargs)), nid)
        g["nodes"][nid]["out"] = [_shape(t) for t in outs]
        for i, t in enumerate(outs):
            self._register(t, g, nid, i)
        return out

    # --- entry ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, model: nn.Module, x: torch.Tensor) -> dict:
        """Run model(x) and return the model's nested internal graph."""
        self.names = {id(m): n for n, m in model.named_modules()}
        self.mods = dict(model.named_modules())
        self.macs = 0
        self.called = set()
        root = self._new_graph(None, "")
        self.stack = [("scope", None, None, root, None)]
        self._register(x, root, self._add(root, kind="input", type="image", out=[_shape(x)]), 0)
        h1 = nn.modules.module.register_module_forward_pre_hook(self.pre_hook)
        h2 = nn.modules.module.register_module_forward_hook(self.post_hook)
        try:
            with self:
                model(x)
        finally:
            h1.remove(), h2.remove()
        model_node = next(n for n in root["nodes"] if n["kind"] == "module")
        self.stack, self.prod = [], {}
        return model_node["g"]
