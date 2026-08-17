#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert Ultralytics YOLO detection and depth models to GGUF.

Usage:
    python scripts/convert_yolo_to_gguf.py --model yolov8n --dtype f16
    python scripts/convert_yolo_to_gguf.py --model yolo26n-depth --dtype f16

Supported --dtype: f32 | f16 | q8_0

The converter loads a .pt checkpoint via the Ultralytics Python API, fuses
Conv+BN, and flattens the module tree into a small op-graph vocabulary that the
C++ engine (cpp_ggml/src) interprets:

    conv, dwconv, maxpool, concat, upsample, add, slice,
    psa_attention (C2PSA / Attention), detect (v8 DFL head / end-to-end head),
    interpolate, conv_transpose, depth (YOLO26 metric-depth decoder)

Every op is stored as GGUF metadata keys `op.{i}.*` with weights stored as
tensors named `op.{i}.w` / `op.{i}.b` etc., so the C++ side is fully
metadata-driven — no per-architecture hardcoding.

Quantization rules (q8_0): a conv / dwconv weight [out, in*kh*kw] is quantized
to Q8_0 only when (in*kh*kw) % 32 == 0; everything else stays F16. Biases and
the DFL projection stay F32 in all modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import gguf
except ImportError:
    print("error: the 'gguf' package is required: pip install gguf", file=sys.stderr)
    raise

# Make the repo-local ultralytics importable when run from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

OP_GRAPH_VERSION = 2
CPP_ROOT = Path(__file__).resolve().parent.parent
PYTORCH_MODELS = CPP_ROOT / "models" / "pytorch"
GGUF_MODELS = CPP_ROOT / "models" / "gguf"


class Op:
    """One node of the flattened op graph plus its tensor payloads."""

    def __init__(self, op_type: str, inputs, **params):
        self.type = op_type
        self.inputs = list(inputs)
        self.params = params  # KV params (scalars / lists / strings)
        self.tensors = {}  # tensor name -> np.ndarray (torch layout)

    def __repr__(self):
        return f"Op({self.type}, in={self.inputs})"


class GraphBuilder:
    """Walk the fused Ultralytics model and emit flat ops."""

    def __init__(self):
        self.ops: list[Op] = []

    def add(self, op: Op) -> int:
        self.ops.append(op)
        return len(self.ops) - 1

    # ------------------------------------------------------------------
    # primitive emitters
    # ------------------------------------------------------------------
    def emit_conv2d(self, conv: nn.Conv2d, idx: int, act: str = "none") -> int:
        w = conv.weight.detach().cpu().float().numpy()  # [out, in, kh, kw]
        op = Op(
            "conv" if conv.groups == 1 else "dwconv",
            [idx],
            act=act,
            s=[conv.stride[0], conv.stride[1]],
            p=[conv.padding[0], conv.padding[1]],
            d=[conv.dilation[0], conv.dilation[1]],
            k=[conv.kernel_size[0], conv.kernel_size[1]],
            g=conv.groups,
        )
        op.tensors["w"] = np.ascontiguousarray(w)
        if conv.bias is not None:
            op.tensors["b"] = conv.bias.detach().cpu().float().numpy()
        return self.add(op)

    def emit_conv_module(self, m: nn.Module, idx: int) -> int:
        """Conv / DWConv wrapper (fused Conv+BN+SiLU)."""
        conv = m.conv if hasattr(m, "conv") else m
        act = "silu" if isinstance(getattr(m, "act", None), nn.SiLU) else "none"
        assert conv.groups == 1 or conv.groups == conv.in_channels, (
            f"grouped conv (groups={conv.groups}) not supported yet"
        )
        return self.emit_conv2d(conv, idx, act)

    def emit_sequential(self, seq: nn.Module, idx: int) -> int:
        for m in seq:
            if isinstance(m, (nn.Conv2d,)):
                idx = self.emit_conv2d(m, idx)
            elif hasattr(m, "conv"):  # Conv / DWConv wrappers
                idx = self.emit_conv_module(m, idx)
            elif isinstance(m, nn.Sequential):
                idx = self.emit_sequential(m, idx)
            else:
                raise TypeError(f"unsupported module inside sequential: {type(m).__name__}")
        return idx

    def emit_conv_transpose(self, conv: nn.ConvTranspose2d, idx: int) -> int:
        """Emit the stride-2, zero-padding transpose convolution used by the depth head."""
        assert conv.groups == 1 and conv.padding == (0, 0) and conv.output_padding == (0, 0)
        assert conv.dilation == (1, 1) and conv.stride[0] == conv.stride[1]
        op = Op("conv_transpose", [idx], s=conv.stride[0])
        # torch [in, out, kh, kw] is reversed by GGUF to ggml [kw, kh, out, in].
        op.tensors["w"] = np.ascontiguousarray(conv.weight.detach().cpu().float().numpy())
        if conv.bias is not None:
            op.tensors["b"] = conv.bias.detach().cpu().float().numpy()
        return self.add(op)

    # ------------------------------------------------------------------
    # compound emitters
    # ------------------------------------------------------------------
    def emit_bottleneck(self, m: nn.Module, idx: int) -> int:
        h = self.emit_conv_module(m.cv1, idx)
        h = self.emit_conv_module(m.cv2, h)
        if getattr(m, "add", False):
            h = self.add(Op("add", [idx, h]))
        return h

    def emit_c2f_like(self, m: nn.Module, idx: int) -> int:
        """C2f / C3k2 / C3 / C3k — cv1 chunk(2) + bottleneck chain + concat + cv2/cv3."""
        from ultralytics.nn.modules.block import C3k

        c = m.cv1.conv.out_channels // 2
        h = self.emit_conv_module(m.cv1, idx)
        a = self.add(Op("slice", [h], start=0, end=c))
        b = self.add(Op("slice", [h], start=c, end=2 * c))
        parts, cur = [a, b], b
        for wrapper in m.m:
            # C2f concatenates ONE output per m.m entry; a C3k2(attn=True) entry is a
            # Sequential(Bottleneck, PSABlock) whose final output is that one part.
            for blk in wrapper if isinstance(wrapper, nn.Sequential) else (wrapper,):
                if isinstance(blk, C3k):
                    cur = self.emit_c3k(blk, cur)
                elif hasattr(blk, "attn"):  # PSABlock (C3k2 with attn=True)
                    cur = self.emit_psablock(blk, cur)
                elif hasattr(blk, "cv1") and hasattr(blk, "cv2") and not hasattr(blk, "cv3"):
                    cur = self.emit_bottleneck(blk, cur)
                else:
                    raise TypeError(f"unsupported block inside C2f-like: {type(blk).__name__}")
            parts.append(cur)
        cat = self.add(Op("concat", parts))
        return self.emit_conv_module(m.cv2 if hasattr(m, "cv2") else m.cv3, cat)

    def emit_c3k(self, m: nn.Module, idx: int) -> int:
        """C3 / C3k — parallel branches: cv3(cat(m(cv1(x)), cv2(x)))."""
        h1 = self.emit_conv_module(m.cv1, idx)
        h2 = self.emit_conv_module(m.cv2, idx)  # parallel branch from the same input
        cur = h1
        for blk in m.m:
            cur = self.emit_bottleneck(blk, cur)
        cat = self.add(Op("concat", [cur, h2]))
        return self.emit_conv_module(m.cv3, cat)

    def emit_sppf(self, m: nn.Module, idx: int) -> int:
        k = m.m.kernel_size
        y = [self.emit_conv_module(m.cv1, idx)]
        cur = y[0]
        for _ in range(getattr(m, "n", 3)):
            cur = self.add(Op("maxpool", [cur], k=k, s=m.m.stride, p=m.m.padding))
            y.append(cur)
        cat = self.add(Op("concat", y))
        out = self.emit_conv_module(m.cv2, cat)
        if getattr(m, "add", False):
            out = self.add(Op("add", [idx, out]))
        return out

    def emit_attention(self, m: nn.Module, idx: int) -> int:
        nh, kd = m.num_heads, m.key_dim
        hd = m.head_dim
        op = Op("psa_attention", [idx], nh=nh, kd=kd, hd=hd, scale=m.scale)
        op.tensors["qkv_w"] = m.qkv.conv.weight.detach().cpu().float().numpy()
        if m.qkv.conv.bias is not None:
            op.tensors["qkv_b"] = m.qkv.conv.bias.detach().cpu().float().numpy()
        op.tensors["proj_w"] = m.proj.conv.weight.detach().cpu().float().numpy()
        if m.proj.conv.bias is not None:
            op.tensors["proj_b"] = m.proj.conv.bias.detach().cpu().float().numpy()
        op.tensors["pe_w"] = m.pe.conv.weight.detach().cpu().float().numpy()
        if m.pe.conv.bias is not None:
            op.tensors["pe_b"] = m.pe.conv.bias.detach().cpu().float().numpy()
        return self.add(op)

    def emit_psablock(self, m: nn.Module, idx: int) -> int:
        a = self.emit_attention(m.attn, idx)
        if m.add:
            a = self.add(Op("add", [idx, a]))
        f = self.emit_conv_module(m.ffn[0], a)
        f = self.emit_conv_module(m.ffn[1], f)
        if m.add:
            f = self.add(Op("add", [a, f]))
        return f

    def emit_c2psa(self, m: nn.Module, idx: int) -> int:
        h = self.emit_conv_module(m.cv1, idx)
        a = self.add(Op("slice", [h], start=0, end=m.c))
        b = self.add(Op("slice", [h], start=m.c, end=2 * m.c))
        for blk in m.m:
            b = self.emit_psablock(blk, b)
        cat = self.add(Op("concat", [a, b]))
        return self.emit_conv_module(m.cv2, cat)

    def emit_detect(self, m: nn.Module, layer_outputs: list[int]) -> int:
        """Detect head: per-level box+cls conv stacks, then one detect op.

        end2end models are fused first (cv2/cv3 removed) so only the one2one
        head remains. Non-end2end models keep the shared cv2/cv3.
        """
        cv2_list = m.cv2 if m.cv2 is not None else m.one2one_cv2
        cv3_list = m.cv3 if m.cv3 is not None else m.one2one_cv3
        feats = []
        for i, src in enumerate(layer_outputs):
            box = self.emit_sequential(cv2_list[i], src)
            cls = self.emit_sequential(cv3_list[i], src)
            feats.append(self.add(Op("concat", [box, cls])))
        op = Op("detect", feats, reg_max=m.reg_max, nc=m.nc, end2end=int(m.end2end), max_det=getattr(m, "max_det", 300))
        if m.reg_max > 1:
            op.tensors["dfl_w"] = m.dfl.conv.weight.detach().cpu().float().numpy()  # [1, reg_max, 1, 1]
        return self.add(op)

    def emit_depth(self, m: nn.Module, layer_outputs: list[int]) -> int:
        """Depth head: project P3-P5, coarse-to-fine fusion, then decode calibrated metric depth."""
        feats = [self.emit_conv_module(proj, src) for proj, src in zip(m.proj, layer_outputs)]
        out = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            out = self.add(Op("interpolate", [out], sf=2, align_corners=1))
            out = self.add(Op("add", [out, feats[i]]))
            out = self.emit_sequential(m.refine[i], out)

        out = self.emit_conv_module(m.head[0], out)
        out = self.emit_conv_transpose(m.head[1], out)
        out = self.emit_conv_module(m.head[2], out)
        out = self.emit_conv2d(m.head[3], out)
        return self.add(Op("depth", [out], cal_a=float(m.cal_a.item()), cal_b=float(m.cal_b.item())))

    # ------------------------------------------------------------------
    # top-level walk over the DetectionModel layer list
    # ------------------------------------------------------------------
    def build(self, model: nn.Module) -> None:
        from ultralytics.nn.modules import C2PSA, SPPF, C2f, C3k2, Concat
        from ultralytics.nn.modules.head import Depth, Detect

        seq = model.model  # nn.Sequential of layers
        save_idx: list[int] = []  # layer i output op index
        self.layers: list[dict] = []  # torch layer -> last op index (for parity tests)
        for layer in seq:
            i = len(save_idx)
            f = layer.f  # from: -1 = previous layer, int index, or list (Detect)
            src = -1 if i == 0 else (save_idx[i - 1] if f == -1 else (None if isinstance(f, list) else save_idx[f]))
            if isinstance(layer, Detect):
                idx = self.emit_detect(layer, [save_idx[j] for j in f])
            elif isinstance(layer, Depth):
                idx = self.emit_depth(layer, [save_idx[j] for j in f])
            elif isinstance(layer, C2PSA):
                idx = self.emit_c2psa(layer, src)
            elif isinstance(layer, (C3k2, C2f)):
                idx = self.emit_c2f_like(layer, src)
            elif isinstance(layer, SPPF):
                idx = self.emit_sppf(layer, src)
            elif isinstance(layer, nn.Upsample):
                sf = layer.scale_factor
                idx = self.add(Op("upsample", [src], sf=int(sf) if isinstance(sf, (int, float)) else int(sf[0])))
            elif isinstance(layer, Concat):
                idx = self.add(Op("concat", [save_idx[j] for j in f]))
            elif isinstance(layer, nn.Conv2d) or hasattr(layer, "conv"):
                idx = self.emit_conv_module(layer, src)
            else:
                raise TypeError(f"unsupported top-level layer {i}: {type(layer).__name__}")
            save_idx.append(idx)
            self.layers.append({"idx": i, "type": type(layer).__name__, "op": idx})


# ----------------------------------------------------------------------
# quantization + GGUF writing
# ----------------------------------------------------------------------
Q8_BLOCK = 32


def should_quantize(name: str, arr: np.ndarray) -> bool:
    """Quantize a conv weight [out, in, kh, kw] when K = in*kh*kw is 32-aligned.

    Depthwise kernels (in == 1) stay f16: the CPU conv_2d_dw path requires an
    f16 kernel next to its f16 im2col, and their K is tiny anyway.
    """
    if arr.ndim != 4 or arr.shape[1] == 1:
        return False
    k = arr.shape[1] * arr.shape[2] * arr.shape[3]
    return k % Q8_BLOCK == 0 and arr.shape[0] >= 1


def write_gguf(path: str, name: str, builder: GraphBuilder, meta: dict, dtype: str):
    writer = gguf.GGUFWriter(path, "yolo")
    writer.add_string("general.name", name)
    writer.add_uint32("yolo.op_graph_version", OP_GRAPH_VERSION)
    writer.add_string("yolo.task", meta["task"])
    writer.add_uint32("yolo.nc", meta["nc"])
    writer.add_uint32("yolo.nl", meta["nl"])
    writer.add_uint32("yolo.imgsz", meta["imgsz"])
    writer.add_array("yolo.strides", [float(s) for s in meta["strides"]])
    writer.add_string("yolo.dtype", dtype)
    writer.add_array("yolo.class_names", [str(meta["names"][k]) for k in sorted(meta["names"])])

    writer.add_uint32("yolo.op.count", len(builder.ops))
    for i, op in enumerate(builder.ops):
        prefix = f"op.{i}"
        writer.add_string(f"{prefix}.type", op.type)
        writer.add_array(f"{prefix}.inputs", [int(j) for j in op.inputs])
        for key, val in op.params.items():
            fk = f"{prefix}.{key}"
            if isinstance(val, str):
                writer.add_string(fk, val)
            elif isinstance(val, (int, np.integer)):
                writer.add_uint32(fk, int(val))
            elif isinstance(val, float):
                writer.add_float32(fk, float(val))
            elif isinstance(val, (list, tuple, np.ndarray)):
                writer.add_array(fk, [int(v) for v in val])
            else:
                raise TypeError(f"unsupported op param {key}={val!r}")

    quant_type = {"q8_0": gguf.GGMLQuantizationType.Q8_0}.get(dtype)
    n_quant = n_f16 = n_f32 = 0
    for i, op in enumerate(builder.ops):
        for tname, arr in op.tensors.items():
            full = f"op.{i}.{tname}"
            if (
                quant_type is not None
                and op.type != "conv_transpose"
                and tname in ("w", "qkv_w", "proj_w", "pe_w")
                and should_quantize(full, arr)
            ):
                # Quantize in the logical [out, K] layout; K is 32-aligned.
                flat = np.ascontiguousarray(arr.reshape(arr.shape[0], -1))
                q = gguf.quants.quantize(flat, quant_type)
                writer.add_tensor(full, q, raw_dtype=quant_type)
                n_quant += 1
            elif dtype in ("f16", "q8_0") and arr.ndim >= 2:
                writer.add_tensor(full, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
                n_f16 += 1
            else:
                writer.add_tensor(full, arr.astype(np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
                n_f32 += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[write] {path}: {len(builder.ops)} ops, tensors q8_0={n_quant} f16={n_f16} f32={n_f32}")


def convert(model_path: str, dtype: str, output: str, opmap: str | None = None) -> None:
    yolo = YOLO(model_path)
    model = yolo.model
    task = yolo.task or "detect"
    if task not in {"detect", "depth"}:
        raise SystemExit(f"error: task '{task}' not supported (expected detect or depth)")

    model.eval()
    model.fuse()
    imgsz = 768 if task == "depth" else 640

    # One dummy forward so Detect.stride / .names settle (lazy init on load).
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))

    head = model.model[-1]
    stride = head.stride if task == "detect" and head.stride is not None else torch.tensor([8.0, 16.0, 32.0])

    builder = GraphBuilder()
    builder.build(model)

    meta = {
        "task": task,
        "nc": int(head.nc) if task == "detect" else 1,
        "nl": int(head.nl),
        "imgsz": imgsz,
        "strides": [float(s) for s in stride.tolist()],
        "names": model.names,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_gguf(output, Path(model_path).stem, builder, meta, dtype)
    if opmap:
        import json

        with open(opmap, "w") as f:
            json.dump({"n_ops": len(builder.ops), "layers": builder.layers}, f, indent=1)
        print(f"[opmap] {opmap}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".pt checkpoint path or alias (yolov8n, yolo26n-depth)")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--output", help="output .gguf path (default: models/gguf/<model>-<dtype>.gguf)")
    ap.add_argument("--opmap", help="optional JSON: torch layer -> op index map (parity tests)")
    args = ap.parse_args()

    PYTORCH_MODELS.mkdir(parents=True, exist_ok=True)
    requested = Path(args.model)
    if requested.exists() or requested.parent != Path("."):
        src = requested
    else:
        src = PYTORCH_MODELS / (requested.name if requested.suffix == ".pt" else f"{requested.name}.pt")
    output = args.output or GGUF_MODELS / f"{src.stem}-{args.dtype}.gguf"
    convert(str(src), args.dtype, str(output), args.opmap)


if __name__ == "__main__":
    main()
