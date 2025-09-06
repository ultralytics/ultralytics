# ultralytics/utils/profile.py
from __future__ import annotations
import json, time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Tuple
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


def _try_profile(m: nn.Module, imgsz: int | Tuple[int, int], runner: Callable[[torch.Tensor], Any]) -> Tuple[Any, float]:
    """Run `runner(im)` first at stride, then fallback to imgsz on error; returns (result, scale)."""
    try:
        im, scale = _input_from_model(m, imgsz, try_stride=True)
        return runner(im), scale
    except Exception:
        im, scale = _input_from_model(m, imgsz, try_stride=False)
        return runner(im), scale


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
            # conv MACs
            macs += B * H * W * Cout * ((Cin // G) * Kh * Kw)
            if module.bias is not None:
                adds += B * Cout * H * W
        elif isinstance(module, nn.Linear):
            # linear MACs
            macs += B * int(module.in_features) * int(module.out_features)
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
        flops = ((macs * 2) + adds) * float(scale) / 1e9
        return float(flops)
    finally:
        for h in handles:
            h.remove()


def _flops_thop_once(model: nn.Module, imgsz=640) -> float:
    """FLOPs via thop; run at stride then fallback to imgsz; stride-scaled."""
    import thop
    from copy import deepcopy
    m = deepcopy(de_parallel(model)).eval()
    with _eval_no_grad_restore(m):
        macs, scale = _try_profile(m, imgsz, lambda im: thop.profile(m, inputs=[im], verbose=False)[0])
    return float(macs) * 2.0 * float(scale) / 1e9


def get_flops_profiler(model: nn.Module, imgsz=640) -> float:
    """FLOPs from torch.profiler with_flops=True; stride→imgsz; fallback to hooks if profiler returns 0."""
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return float("nan")

    m = de_parallel(model)
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    def run_with_prof(im: torch.Tensor):
        with profile(activities=acts, with_flops=True, record_shapes=True) as prof:
            m(im)
            flops = sum(int(getattr(e, "flops", 0) or 0) for e in prof.events())
            if not flops:
                flops = sum(int(getattr(e, "flops", 0) or 0) for e in prof.key_averages())
            return flops

    with _eval_no_grad_restore(m):
        try:
            flops, scale = _try_profile(m, imgsz, run_with_prof)
        except Exception:
            return float("nan")

    if flops == 0:
        try:
            return get_flops(model, imgsz)
        except Exception:
            return float("nan")
    return float(flops) * float(scale) / 1e9


def _run_n(fn, n=10) -> List[float]:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def benchmark(model: nn.Module, imgsz=640, iters=10) -> Dict[str, Dict[str, float]]:
    """Benchmark FLOPs calculators; report GFLOPs and avg speed (ms) only."""
    out: Dict[str, Dict[str, float]] = {}

    try:
        ts = _run_n(lambda: get_flops(model, imgsz), iters)
        out["hooks"] = {"GFLOPs": get_flops(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    try:
        import thop  # noqa: F401
        ts = _run_n(lambda: _flops_thop_once(model, imgsz), iters)
        out["thop"] = {"GFLOPs": _flops_thop_once(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    try:
        ts = _run_n(lambda: get_flops_profiler(model, imgsz), iters)
        out["profiler"] = {"GFLOPs": get_flops_profiler(model, imgsz), "speed_ms": (sum(ts) / iters) * 1e3}
    except Exception:
        pass

    return out


if __name__ == "__main__":
    from ultralytics import YOLO, RTDETR, FastSAM

    model = YOLO("yolo11n.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("YOLO11n:", json.dumps(res, indent=2))

    model = RTDETR("rtdetr-l.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("RT-DETR-L:", json.dumps(res, indent=2))

    model = FastSAM("FastSAM-s.pt").model.eval()
    res = benchmark(model, imgsz=640, iters=3)
    print("SAM:", json.dumps(res, indent=2))
