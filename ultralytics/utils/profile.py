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
    if not isinstance(imgsz, (list, tuple)):
        imgsz = [int(imgsz), int(imgsz)]
    if try_stride and hasattr(m, "stride"):
        try:
            s = max(int(torch.as_tensor(m.stride).max().item()), 32)
        except Exception:
            s = 32
        return torch.empty((1, ch, s, s), device=device), (imgsz[0] / s) * (imgsz[1] / s)
    return torch.empty((1, ch, imgsz[0], imgsz[1]), device=device), 1.0


def _count_conv2d(B, Cin, H, W, Cout, Kh, Kw, G) -> int:
    return B * H * W * Cout * ((Cin // G) * Kh * Kw)  # MACs


def _count_linear(B, in_f, out_f) -> int:
    return B * in_f * out_f  # MACs


def get_flops(model: nn.Module, imgsz=640) -> float:
    """Hook-based FLOPs to match torch.profiler(with_flops=True); stride-scaled; no caching."""
    m = de_parallel(model)
    macs = 0; adds = 0

    def fh(module, inp, out):
        nonlocal macs, adds
        x = inp[0]; B = int(x.shape[0])
        if isinstance(module, nn.Conv2d):
            Cin = int(x.shape[1]); Cout = int(module.out_channels)
            k = module.kernel_size; Kh, Kw = (k if isinstance(k, tuple) else (k, k))
            G = int(module.groups)
            H, W = int(out.shape[-2]), int(out.shape[-1])
            macs += _count_conv2d(B, Cin, H, W, Cout, Kh, Kw, G)
            if module.bias is not None:
                adds += B * Cout * H * W
        elif isinstance(module, nn.Linear):
            macs += _count_linear(B, int(module.in_features), int(module.out_features))
            if module.bias is not None:
                adds += B * int(module.out_features)

    handles = []
    try:
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                handles.append(mod.register_forward_hook(fh))
        with _eval_no_grad_restore(m):
            try:
                im, scale = _input_from_model(m, imgsz, try_stride=True); m(im)
            except Exception:
                macs = adds = 0
                im, scale = _input_from_model(m, imgsz, try_stride=False); m(im)
        flops = ((macs * 2) + adds) * float(scale) / 1e9
        return float(flops)
    finally:
        for h in handles: h.remove()


# ---------- thop comparator ----------
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


# ---------- torch.profiler ----------
def get_flops_profiler(model: nn.Module, imgsz=640) -> float:
    """FLOPs from torch.profiler with_flops=True, run at stride and upscale to imgsz."""
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return float("nan")
    m = de_parallel(model)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with _eval_no_grad_restore(m):
        try:
            im, scale = _input_from_model(m, imgsz, try_stride=True)
            with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                m(im)
        except Exception:
            try:
                im, scale = _input_from_model(m, imgsz, try_stride=False)
                with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                    m(im)
            except Exception:
                return float("nan")
    flops = 0
    for e in prof.key_averages():
        f = getattr(e, "flops", None)
        if f is not None:
            flops += int(f)
    return float(flops) * float(scale) / 1e9


def _run_n(fn, n=10) -> List[float]:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts


def benchmark(model: nn.Module, imgsz=640, iters=10) -> Dict[str, Dict[str, float]]:
    """Benchmark FLOPs calculators; report GFLOPs and avg speed (ms) only."""
    out: Dict[str, Dict[str, float]] = {}

    # hooks
    ts = _run_n(lambda: get_flops(model, imgsz), iters)
    out["hooks"] = {"GFLOPs": get_flops(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}

    # thop
    try:
        import thop  # noqa: F401
        ts = _run_n(lambda: _flops_thop_once(model, imgsz), iters)
        out["thop"] = {"GFLOPs": _flops_thop_once(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    # profiler
    try:
        ts = _run_n(lambda: get_flops_profiler(model, imgsz), iters)
        out["profiler"] = {"GFLOPs": get_flops_profiler(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    return out


if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=10)
    print(json.dumps(res, indent=2))
