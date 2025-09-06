# ultralytics/utils/profile.py
from __future__ import annotations
import json, time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn


# ----- flags -----
TORCH_2_0 = hasattr(torch, "profiler") and hasattr(torch.profiler, "profile")


# ----- shared helpers -----
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
    """Return dummy input and scale; prefer stride-sized square, else imgsz."""
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


def _try_profile(m: nn.Module, imgsz: int | Tuple[int, int], runner: Callable[[torch.Tensor], Any]) -> Tuple[Any, float]:
    """Run `runner(im)` first at stride, then fallback to imgsz on error; returns (result, scale)."""
    try:
        im, scale = _input_from_model(m, imgsz, try_stride=True)
        return runner(im), scale
    except Exception:
        im, scale = _input_from_model(m, imgsz, try_stride=False)
        return runner(im), scale


# ----- method 1: hooks -----
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
            G = int(module.groups); H, W = int(out.shape[-2]), int(out.shape[-1])
            macs += B * H * W * Cout * ((Cin // G) * Kh * Kw)  # conv MACs
            if module.bias is not None:
                adds += B * Cout * H * W
        elif isinstance(module, nn.Linear):
            macs += B * int(module.in_features) * int(module.out_features)  # linear MACs
            if module.bias is not None:
                adds += B * int(module.out_features)

    handles = []
    try:
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                handles.append(mod.register_forward_hook(fh))
        with _eval_no_grad_restore(m):
            try:
                _, scale = _try_profile(m, imgsz, lambda im: m(im))
            except Exception:
                return float("nan")
        return float(((macs * 2) + adds) * float(scale) / 1e9)
    finally:
        for h in handles: h.remove()


# ----- method 2: thop -----
def get_flops_thop(model: nn.Module, imgsz=640) -> float:
    """FLOPs via thop; stride→imgsz; stride-scaled."""
    import thop
    from copy import deepcopy
    m = deepcopy(de_parallel(model)).eval()

    def run(im: torch.Tensor):
        return thop.profile(m, inputs=[im], verbose=False)[0]  # MACs

    with _eval_no_grad_restore(m):
        try:
            macs, scale = _try_profile(m, imgsz, run)
        except Exception:
            return float("nan")
    return float(macs) * 2.0 * float(scale) / 1e9


# ----- method 3: torch.profiler (standalone, stride→imgsz fallback) -----
def get_flops_with_torch_profiler(model: nn.Module, imgsz=640) -> float:
    """
    Compute GFLOPs using torch.profiler only; prefer stride-sized input then fall back to imgsz-sized input.
    """
    if not TORCH_2_0:
        return 0.0  # profiler not available in older torch

    m = de_parallel(model)
    p = next(m.parameters())
    if not isinstance(imgsz, (list, tuple)):
        imgsz = [int(imgsz), int(imgsz)]

    with _eval_no_grad_restore(m):
        # Try stride-sized input first
        try:
            s = (max(int(torch.as_tensor(m.stride).max().item()), 32) if hasattr(m, "stride") else 32) * 2
            im = torch.empty((1, int(p.shape[1]), s, s), device=p.device)
            with torch.profiler.profile(with_flops=True) as prof:
                m(im)
            flops = sum((getattr(x, "flops", 0) or 0) for x in prof.key_averages()) / 1e9
            return float(flops) * (imgsz[0] / s) * (imgsz[1] / s)
        except Exception:
            # Fall back to actual imgsz input
            im = torch.empty((1, int(p.shape[1]), imgsz[0], imgsz[1]), device=p.device)
            with torch.profiler.profile(with_flops=True) as prof:
                m(im)
            flops = sum((getattr(x, "flops", 0) or 0) for x in prof.key_averages()) / 1e9
            return float(flops)


# ----- utils -----
def _run_n(fn, n=10) -> List[float]:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts


def benchmark(model: nn.Module, imgsz=640, iters=10) -> Dict[str, Dict[str, float]]:
    """Benchmark FLOPs calculators; report GFLOPs and avg speed (ms) only."""
    out: Dict[str, Dict[str, float]] = {}

    # hooks
    try:
        ts = _run_n(lambda: get_flops(model, imgsz), iters)
        out["hooks"] = {"GFLOPs": get_flops(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    # thop
    try:
        import thop  # noqa: F401
        ts = _run_n(lambda: get_flops_thop(model, imgsz), iters)
        out["thop"] = {"GFLOPs": get_flops_thop(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    # torch.profiler
    try:
        ts = _run_n(lambda: get_flops_with_torch_profiler(model, imgsz), iters)
        out["profiler"] = {"GFLOPs": get_flops_with_torch_profiler(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    return out


if __name__ == "__main__":
    from ultralytics import YOLO, RTDETR, FastSAM

    # YOLO11n
    model = YOLO("yolo11n.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("YOLO11n:", json.dumps(res, indent=2))

    # RT-DETR
    model = RTDETR("rtdetr-l.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("RT-DETR-L:", json.dumps(res, indent=2))

    # SAM
    model = FastSAM("FastSAM-s.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("SAM:", json.dumps(res, indent=2))
