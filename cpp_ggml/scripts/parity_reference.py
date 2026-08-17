#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tensor-level parity reference for the cpp_ggml engine.

Modes:
  prep   <pt> <img> <out.bin>        dump the torch LetterBox input (YINP0001)
  raw    <pt> <in.bin> <out.bin>     run the fused model on a dumped input and
                                     dump the pre-DFL raw head output (YRAW0001)
  diff   <torch.bin> <cpp.bin> [rm]  compare two YRAW0001 dumps
  depth <pt> <img> <out.bin> [dev]  dump official metric depth (YDEP0001)
  ddiff <torch.bin> <cpp.bin>        compare two metric-depth maps
  layers <pt> <in.bin> <outdir>      dump every torch layer output (YLYR0001)
  ldiff  <opmap.json> <torch_dir> <cpp_dir>
                                     compare torch layer outputs vs cpp op outputs
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
IMG = "ultralytics/assets/bus.jpg"


def load_fused(pt: str):
    from ultralytics import YOLO

    model = YOLO(pt)
    model.fuse()
    return model


NDIMS = {b"YINP0001": 3, b"YRAW0001": 2, b"YDEP0001": 2, b"YLYR0001": 4}


def read_bin(path: str):
    with open(path, "rb") as f:
        magic = f.read(8)
        nd = NDIMS.get(magic, 3)
        dims = struct.unpack(f"<{nd}i", f.read(4 * nd))
        data = np.fromfile(f, dtype=np.float32)
    return magic, dims, data


def write_bin(path: str, magic: bytes, dims, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(magic)
        nd = NDIMS.get(magic, len(dims))
        dims = tuple(dims) + (1,) * (nd - len(dims))
        f.write(struct.pack(f"<{nd}i", *dims))
        data.astype(np.float32).tofile(f)


def main():
    mode = sys.argv[1]
    if mode == "prep":
        import cv2

        from ultralytics.data.augment import LetterBox

        model = load_fused(sys.argv[2])
        stride = int(model.model.stride.max())
        im = cv2.imread(sys.argv[3])[:, :, ::-1]  # BGR -> RGB
        lb = LetterBox(640, auto=True, stride=stride)
        im_lb = lb(image=im)
        chw = np.ascontiguousarray(im_lb.transpose(2, 0, 1)) / 255.0
        print(f"canvas={im_lb.shape} input={chw.shape} stride={stride}")
        write_bin(sys.argv[4], b"YINP0001", (3, *chw.shape[1:]), chw)

    elif mode == "raw":
        model = load_fused(sys.argv[2])
        magic, dims, data = read_bin(sys.argv[3])
        assert magic == b"YINP0001", magic
        chw = data.reshape(dims).copy()
        x = torch.from_numpy(chw)[None]
        with torch.no_grad():
            out = model.model(x)
        # Detect returns (y, preds); preds holds the pre-DFL raw head output.
        preds = out[1]
        head = preds.get("one2one", preds)
        if "boxes" not in head:
            head = preds.get("one2many", head)
        raw = torch.cat([head["boxes"], head["scores"]], dim=1)[0].numpy()  # [no, A]
        print(f"raw={raw.shape} canvas={dims}")
        write_bin(sys.argv[4], b"YRAW0001", raw.shape, raw)

    elif mode == "diff":
        _, d1, a = read_bin(sys.argv[2])
        _, d2, b = read_bin(sys.argv[3])
        assert d1 == d2 and a.shape == b.shape, (d1, d2, a.shape, b.shape)
        a, b = a.reshape(d1), b.reshape(d2)
        diff = np.abs(a - b)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"shape={a.shape} max={diff.max():.6f} mean={diff.mean():.8f}")
        print(f"argmax(ch,a)={idx} torch={a[idx]:.6f} cpp={b[idx]:.6f}")
        # Top-5 anchors by torch cls max: are they preserved? (optional argv[4]=reg_max)
        rm = int(sys.argv[4]) if len(sys.argv) > 4 else 16
        nc = a.shape[0] - (4 * rm if rm > 1 else 4)
        cls = a[-nc:]
        top = np.argsort(cls.max(0))[::-1][:5]
        for t in top:
            print(f"anchor {t}: torch_max={cls[:, t].max():.4f} cpp_max={b[-nc:][:, t].max():.4f}")

    elif mode == "depth":
        model = load_fused(sys.argv[2])
        device = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"
        result = model.predict(sys.argv[3], imgsz=768, device=device, verbose=False)[0]
        depth = result.depth.data.float().cpu().numpy()
        print(f"depth={depth.shape} min={depth.min():.6f} mean={depth.mean():.6f} max={depth.max():.6f}")
        write_bin(sys.argv[4], b"YDEP0001", depth.shape, depth)

    elif mode == "ddiff":
        magic_a, dims_a, a = read_bin(sys.argv[2])
        magic_b, dims_b, b = read_bin(sys.argv[3])
        assert magic_a == magic_b == b"YDEP0001"
        assert dims_a == dims_b and a.shape == b.shape, (dims_a, dims_b, a.shape, b.shape)
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(a), 1e-3)
        print(
            f"shape={dims_a} abs_mean={diff.mean():.6f} abs_p99={np.quantile(diff, 0.99):.6f} "
            f"abs_max={diff.max():.6f} rel_mean={rel.mean():.6f} rel_p99={np.quantile(rel, 0.99):.6f}"
        )

    elif mode == "layers":
        model = load_fused(sys.argv[2])
        magic, dims, data = read_bin(sys.argv[3])
        assert magic == b"YINP0001", magic
        x = torch.from_numpy(data.reshape(dims).copy())[None]
        outdir = sys.argv[4]
        os.makedirs(outdir, exist_ok=True)
        feats = {}

        def hook(name):
            def fn(m, i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                if torch.is_tensor(t):
                    feats[name] = t.detach().float().cpu().numpy()

            return fn

        hs = [m.register_forward_hook(hook(f"{i:02d}_{type(m).__name__}")) for i, m in enumerate(model.model.model)]
        with torch.no_grad():
            model.model(x)
        for h in hs:
            h.remove()
        for name, a in feats.items():
            if a.ndim == 4:  # BCHW -> CHW
                a = a[0]
            write_bin(f"{outdir}/{name}.bin", b"YLYR0001", a.shape, a.reshape(-1))
        print(f"layers={len(feats)} -> {outdir}")

    elif mode == "ldiff":
        # argv: opmap.json torch_dir cpp_dir [opdump_prefix]
        with open(sys.argv[2]) as file:
            opmap = json.load(file)
        tdir, cdir = sys.argv[3], sys.argv[4]
        first = True
        for layer in opmap["layers"]:
            tpath = f"{tdir}/{layer['idx']:02d}_{layer['type']}.bin"
            if layer["type"] == "Detect" or not os.path.exists(tpath):
                continue
            try:
                cpath = f"{cdir}/op{layer['op']:03d}_*.bin"
                import glob

                cpath = glob.glob(cpath)[0]
            except IndexError:
                print(f"layer {layer['idx']:02d} {layer['type']:<8}: no cpp op dump")
                continue
            _, td, a = read_bin(tpath)
            _, cd, b = read_bin(cpath)
            # torch dumps (C,H,W,1); ggml dumps ne=[W,H,C,N] whose f32 memory order
            # is x-innermost, i.e. numpy (N,C,H,W) — same CHW planes, no transpose.
            a = a.reshape(td)[..., 0]
            b = b.reshape(cd[::-1])[0]
            if a.shape != b.shape:
                print(f"layer {layer['idx']:02d} {layer['type']:<8}: SHAPE torch={a.shape} cpp={b.shape} <<<<")
                first = False
                continue
            d = np.abs(a - b)
            mark = ""
            if d.max() > 1e-3 and first:
                mark = " <<<< FIRST DIVERGENCE"
                first = False
            print(f"layer {layer['idx']:02d} {layer['type']:<8}: max={d.max():.6f} mean={d.mean():.8f}{mark}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
