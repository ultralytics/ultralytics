# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Profile YOLO26 depth models (n..x) for CPU inference speed on the T4 box's CPU.

Formats, all on CPU (batch=1, imgsz configurable), timed via the predictor's
isolated speed["inference"] (pre/post excluded):
  - PyTorch fp32
  - ONNX Runtime, CPUExecutionProvider, fp32
  - OpenVINO, fp32 (the CPU-optimised backend; CPU analog of TensorRT on GPU)
fp16/TensorRT are GPU-only and intentionally omitted. Fewer iterations than the
GPU run (CPU latency is high but low-variance). Models built from yolo26-depth.yaml
with random weights (speed is weight-independent).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER

WARMUP = 5
RUNS = 30


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def env_info(imgsz):
    import platform

    import torch

    import ultralytics

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "ultralytics": ultralytics.__version__,
        "device": "cpu",
        "imgsz": imgsz,
        "batch": 1,
        "warmup": WARMUP,
        "timed_runs": RUNS,
    }
    for mod, key in (("onnxruntime", "onnxruntime"), ("openvino", "openvino"), ("onnx", "onnx")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            info[key] = None
    return info


def sigma_clip(a, sigma=2, iters=3):
    a = np.asarray(a, dtype=float)
    for _ in range(iters):
        m, s = a.mean(), a.std()
        c = a[(a > m - sigma * s) & (a < m + sigma * s)]
        if len(c) == len(a):
            break
        a = c
    return a


def profile(model, imgsz, **kw):
    """Warmup + timed CPU predictions; return per-stage timings (ms)."""
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        model(img, imgsz=imgsz, verbose=False, **kw)
    pre, inf, post = [], [], []
    for _ in range(RUNS):
        r = model(img, imgsz=imgsz, verbose=False, **kw)
        s = r[0].speed
        pre.append(s["preprocess"])
        inf.append(s["inference"])
        post.append(s["postprocess"])
    inf_c = sigma_clip(inf)
    pre_m, inf_m, post_m = float(np.mean(pre)), float(inf_c.mean()), float(np.mean(post))
    return {
        "pre": round(pre_m, 3),
        "inf": round(inf_m, 3),
        "inf_std": round(float(inf_c.std()), 3),
        "post": round(post_m, 3),
        "total": round(pre_m + inf_m + post_m, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--sizes", default="nsmlx")
    ap.add_argument("--workdir", default="/root/autodl-tmp/cpu_profile")
    ap.add_argument("--out", default="depth_cpu.json")
    args = ap.parse_args()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)  # export artifacts land here, tagged per imgsz below

    rows = []
    for s in args.sizes:
        name = f"yolo26{s}-depth"
        LOGGER.info(f"\n{'=' * 70}\nCPU profiling {name} @ imgsz={args.imgsz}\n{'=' * 70}")

        y = YOLO(f"{name}.yaml")
        y.fuse()
        info = y.info(imgsz=args.imgsz)  # (layers, params, grads, flops)
        params_m, gflops = info[1] / 1e6, info[3]

        # imgsz-tagged artifacts so 768 and 640 don't collide
        onnx_f = Path(f"{name}-{args.imgsz}.onnx")
        ov_dir = Path(f"{name}-{args.imgsz}_openvino_model")
        if not onnx_f.is_file():
            exp = y.export(format="onnx", imgsz=args.imgsz, device="cpu", batch=1, verbose=False)
            Path(exp).rename(onnx_f)
        if not ov_dir.is_dir():
            exp = y.export(format="openvino", imgsz=args.imgsz, device="cpu", batch=1, verbose=False)
            Path(exp).rename(ov_dir)

        pt32 = profile(YOLO(f"{name}.yaml"), args.imgsz, device="cpu", quantize=32)

        onnx_model = YOLO(str(onnx_f))
        ort_t = profile(onnx_model, args.imgsz, device="cpu")
        providers = onnx_model.predictor.model.session.get_providers()

        ov_t = profile(YOLO(str(ov_dir)), args.imgsz, device="cpu")

        rows.append(
            {
                "model": name,
                "params_M": round(params_m, 2),
                "GFLOPs": round(gflops, 1),
                "pt_fp32": pt32,
                "onnx_cpu_fp32": ort_t,
                "onnx_providers": providers,
                "openvino_fp32": ov_t,
            }
        )
        with open(args.out, "w") as f:
            json.dump({"env": env_info(args.imgsz), "rows": rows}, f, indent=2)

    def fps(ms):
        return 1000.0 / ms if ms else 0.0

    for stage, key in (("INFERENCE-ONLY (ms)", "inf"), ("WITH PRE+POST (ms)", "total")):
        print(f"\n== CPU {stage} ==")
        print(
            f"{'model':<16}{'params(M)':>10}{'GFLOPs':>9}{'pt32':>10}{'FPS':>7}{'onnx':>10}{'FPS':>7}{'ov':>10}{'FPS':>7}"
        )
        for r in rows:
            print(
                f"{r['model']:<16}{r['params_M']:>10.2f}{r['GFLOPs']:>9.1f}"
                f"{r['pt_fp32'][key]:>10.2f}{fps(r['pt_fp32'][key]):>7.1f}"
                f"{r['onnx_cpu_fp32'][key]:>10.2f}{fps(r['onnx_cpu_fp32'][key]):>7.1f}"
                f"{r['openvino_fp32'][key]:>10.2f}{fps(r['openvino_fp32'][key]):>7.1f}"
            )
    print(f"\nONNX providers used: {rows[0]['onnx_providers'] if rows else 'n/a'}")


if __name__ == "__main__":
    main()
