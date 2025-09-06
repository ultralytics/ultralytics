# ultralytics/utils/profile.py
from __future__ import annotations
import json, time
from contextlib import contextmanager
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

def de_parallel(m):
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m

@contextmanager
def _eval_no_grad_restore(model: nn.Module):
    modes = [x.training for x in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for x, t in zip(model.modules(), modes):
            x.train(t)

def _input_from_model(model: nn.Module, imgsz=640, try_stride=True) -> Tuple[torch.Tensor, float]:
    m = de_parallel(model)
    p = next(m.parameters())
    ch = int(p.shape[1]); device = p.device
    if not isinstance(imgsz, (list, tuple)): imgsz = [int(imgsz), int(imgsz)]
    if try_stride and hasattr(m, "stride"):
        try: s = max(int(torch.as_tensor(m.stride).max().item()), 32)
        except Exception: s = 32
        return torch.empty((1, ch, s, s), device=device), (imgsz[0] / s) * (imgsz[1] / s)
    return torch.empty((1, ch, imgsz[0], imgsz[1]), device=device), 1.0

def _count_conv2d(B, Cin, H, W, Cout, Kh, Kw, G) -> int:
    return B * H * W * Cout * ((Cin // G) * Kh * Kw)  # MACs

def _count_linear(B, in_f, out_f) -> int:
    return B * in_f * out_f  # MACs

def get_flops(model: nn.Module, imgsz=640) -> float:
    """Return GFLOPs via forward hooks (Conv/Linear + bias + AvgPool/AdaptiveAvgPool)."""
    m = de_parallel(model)
    macs_muladd = 0  # Conv/Linear MACs (each -> 2 FLOPs)
    adds_only = 0    # bias/pooling adds (optionally add 1 div per output if desired)

    def fh(module, inp, out):
        nonlocal macs_muladd, adds_only
        x = inp[0]; B = int(x.shape[0])
        if isinstance(module, nn.Conv2d):
            Cin = int(x.shape[1]); Cout = module.out_channels
            Kh, Kw = module.kernel_size; G = module.groups
            H, W = int(out.shape[-2]), int(out.shape[-1])
            macs_muladd += _count_conv2d(B, Cin, H, W, Cout, Kh, Kw, G)
            if module.bias is not None: adds_only += B * Cout * H * W
        elif isinstance(module, nn.Linear):
            macs_muladd += _count_linear(B, module.in_features, module.out_features)
            if module.bias is not None: adds_only += B * module.out_features
        elif isinstance(module, nn.AvgPool2d):
            Hout, Wout = int(out.shape[-2]), int(out.shape[-1]); C = int(out.shape[1])
            k = module.kernel_size; Kh, Kw = (k if isinstance(k, tuple) else (k, k))
            adds_only += B * C * Hout * Wout * (Kh * Kw - 1)
            # adds_only += B * C * Hout * Wout  # +1 per output for division if you want to count it
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            Hin, Win = int(x.shape[-2]), int(x.shape[-1])
            Hout, Wout = int(out.shape[-2]), int(out.shape[-1]); C = int(out.shape[1])
            Kh = max(1, Hin // Hout); Kw = max(1, Win // Wout)
            adds_only += B * C * Hout * Wout * (Kh * Kw - 1)
            # adds_only += B * C * Hout * Wout  # optional division

    handles = []
    try:
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                handles.append(mod.register_forward_hook(fh))
        with _eval_no_grad_restore(m):
            try:
                im, scale = _input_from_model(m, imgsz, try_stride=True); m(im)
            except Exception:
                macs_muladd = adds_only = 0
                im, scale = _input_from_model(m, imgsz, try_stride=False); m(im)
        gflops = ((macs_muladd * 2) + adds_only) * scale / 1e9
        return float(gflops)
    finally:
        for h in handles: h.remove()

# ---------- thop comparator (deepcopy is part of time) ----------
def _flops_thop_once(model: nn.Module, imgsz=640) -> float:
    import thop
    from copy import deepcopy
    m = deepcopy(de_parallel(model)).eval()
    im, scale = _input_from_model(m, imgsz, try_stride=True)
    try:
        macs = thop.profile(m, inputs=[im], verbose=False)[0]
    except Exception:
        im, scale = _input_from_model(m, imgsz, try_stride=False)
        macs = thop.profile(m, inputs=[im], verbose=False)[0]
    return float(macs) * 2.0 * scale / 1e9

def _run_n(fn, n=10) -> List[float]:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts

def benchmark(model: nn.Module, imgsz=640, iters=10) -> Dict[str, Dict[str, float]]:
    """Compare native hooks vs thop over N runs; thop timing includes deepcopy."""
    # hooks
    _ = get_flops(model, imgsz)  # warm
    ts_hooks = _run_n(lambda: get_flops(model, imgsz), iters)
    g_hooks = get_flops(model, imgsz)
    avg_h = sum(ts_hooks) / iters; var_h = sum((t - avg_h) ** 2 for t in ts_hooks) / (iters - 1) if iters > 1 else 0.0
    out = {"hooks": {"GFLOPs": g_hooks, "avg_s": avg_h, "std_s": var_h ** 0.5, "total_s": sum(ts_hooks)}}

    # thop (optional)
    try:
        import thop  # noqa: F401
        _ = _flops_thop_once(model, imgsz)  # warm
        ts_thop = _run_n(lambda: _flops_thop_once(model, imgsz), iters)
        g_thop = _flops_thop_once(model, imgsz)
        avg_t = sum(ts_thop) / iters; var_t = sum((t - avg_t) ** 2 for t in ts_thop) / (iters - 1) if iters > 1 else 0.0
        out["thop"] = {"GFLOPs": g_thop, "avg_s": avg_t, "std_s": var_t ** 0.5, "total_s": sum(ts_thop)}
        if avg_t > 0:
            out["speedup_vs_thop"] = avg_t / avg_h if avg_h > 0 else float("inf")
    except Exception:
        pass
    return out

if __name__ == "__main__":

    # load YOLO11n (CPU is fine)
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt").model.eval()

    # run FLOPs benchmark at 640
    res = benchmark(model, imgsz=640, iters=10)
    print(json.dumps(res, indent=2))
