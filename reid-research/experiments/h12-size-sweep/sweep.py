"""H12 — ReID size sweep (n/s/m/l/x), 8-GPU DDP, both stages.

For each size: yolo26{n,s,m,l,x}-reid-2psa
  Stage A: MSMT17 pretrain, 8-GPU DDP, recipe mirrors h11/pretrain.py
  Stage B: Market-1501 FT, 8-GPU DDP, recipe lifted from ty2_msmt448_then_ft448 to 8-GPU
  Eval:    val + val(reid_tta=True) for headline numbers

Designed to run on ultra1 (8x RTX PRO 6000 Blackwell). Background nvidia-smi logger
runs as a sidecar (see mem_logger.sh) — sweep.py prints H12_SIZE_{START,END,FAIL}
markers so the continuous CSV can be sliced post-hoc.

Env knobs:
  SIZES=n,s,m,l,x        which sizes to run (comma-separated, default all 5)
  SMOKE=1                tiny sanity: 1 epoch each stage, fraction=0.01, single GPU
  EPOCHS_A=NNN           override Stage A epochs
  EPOCHS_B=NNN           override Stage B epochs
  PROJECT=/path          override output dir (default /home/rick/runs/reid/h12)

Run on ultra1:
  PYTHONPATH=/home/rick/ultralytics /home/rick/ultralytics/.venv/bin/python \\
      reid-research/experiments/h12-size-sweep/sweep.py 2>&1 | tee /home/rick/runs/reid/h12/sweep.log
"""

from __future__ import annotations

import os
import time
import traceback

os.chdir("/home/rick/ultralytics")

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

SIZES = [s.strip() for s in os.environ.get("SIZES", "n,s,m,l,x").split(",") if s.strip()]
PROJECT = os.environ.get("PROJECT", "/home/rick/runs/reid/h12")
SMOKE = os.environ.get("SMOKE") == "1"

if SMOKE:
    DEVICE = os.environ.get("DEVICE", "0")
    EPOCHS_A = int(os.environ.get("EPOCHS_A", 1))
    EPOCHS_B = int(os.environ.get("EPOCHS_B", 1))
    FRACTION = float(os.environ.get("FRACTION", 0.01))
    WORKERS_A = WORKERS_B = 2
else:
    DEVICE = os.environ.get("DEVICE", "0,1,2,3,4,5,6,7")
    EPOCHS_A = int(os.environ.get("EPOCHS_A", 1277))
    EPOCHS_B = int(os.environ.get("EPOCHS_B", 635))
    FRACTION = float(os.environ.get("FRACTION", 1.0))
    WORKERS_A, WORKERS_B = 12, 8

os.makedirs(PROJECT, exist_ok=True)


def stage_a_kwargs(size: str) -> dict:
    return dict(
        data="ultralytics/cfg/datasets/MSMT17-v1.yaml",
        pretrained=f"yolo26{size}-cls.pt",
        epochs=EPOCHS_A, imgsz=288, device=DEVICE, workers=WORKERS_A, fraction=FRACTION,
        reid_p=128, reid_k=4,
        optimizer="AdamW", lr0=3e-3, lrf=0.001, momentum=0.9, weight_decay=0.01,
        warmup_epochs=0.5, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=False,
        label_smoothing=0.1, erasing=0.5, scale=0.5, auto_augment="randaugment",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.0, dropout=0.0,
        triplet_margin=0.2, triplet_weight=1.0, ce_weight=2.0,
        center_weight=0.005, center_momentum=0.9, focal_gamma=2.0, supcon_temp=0.05,
        amp=False, val=False, verbose=True,
        project=PROJECT, name=f"h12_{size}_msmt_pretrain", exist_ok=True,
    )


def stage_b_kwargs(size: str, ckpt: str) -> dict:
    return dict(
        data="ultralytics/cfg/datasets/Market-1501.yaml",
        pretrained=ckpt,
        epochs=EPOCHS_B, imgsz=448, device=DEVICE, workers=WORKERS_B, fraction=FRACTION,
        reid_p=8, reid_k=8,
        optimizer="AdamW", lr0=3.5e-3, lrf=0.001, momentum=0.937, weight_decay=0.01,
        warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=True,
        label_smoothing=0.1, erasing=0.7, scale=0.5, auto_augment="randaugment",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.0, dropout=0.1,
        triplet_margin=0.5, triplet_weight=1.0, ce_weight=2.0,
        center_weight=0.005, center_momentum=0.9, focal_gamma=0.0, supcon_temp=0.02,
        amp=False, val=False, verbose=True,
        project=PROJECT, name=f"h12_{size}_market_ft", exist_ok=True,
    )


def run_size(size: str) -> tuple:
    """Return (status, r1, mAP, r1_tta, mAP_tta)."""
    cfg = f"ultralytics/cfg/models/26/yolo26{size}-reid-2psa.yaml"

    t0 = time.time()
    print(f"=== H12_SIZE_START size={size} stage=A ts={t0:.0f} cfg={cfg} ===", flush=True)
    mA = YOLO(cfg, task="reid")
    mA.train(**stage_a_kwargs(size))
    ckpt = str(mA.trainer.save_dir / "weights" / "best.pt")
    print(f"=== H12_SIZE_END size={size} stage=A ckpt={ckpt} elapsed={time.time() - t0:.0f}s ===", flush=True)

    t1 = time.time()
    print(f"=== H12_SIZE_START size={size} stage=B ts={t1:.0f} pretrained={ckpt} ===", flush=True)
    mB = YOLO(cfg, task="reid")
    mB.train(**stage_b_kwargs(size, ckpt))
    print(f"=== H12_FT_TRAIN_DONE size={size} save_dir={mB.trainer.save_dir} ===", flush=True)

    std = mB.val(data="ultralytics/cfg/datasets/Market-1501.yaml", imgsz=448, split="val")
    r1, m_ap = getattr(std, "rank1", None), getattr(std, "map", None)
    print(f"=== H12_EVAL_STD size={size} R1={r1} mAP={m_ap} ===", flush=True)

    try:
        tta = mB.val(data="ultralytics/cfg/datasets/Market-1501.yaml", imgsz=448, split="val", reid_tta=True)
        r1_tta, map_tta = getattr(tta, "rank1", None), getattr(tta, "map", None)
        print(f"=== H12_EVAL_TTA size={size} R1={r1_tta} mAP={map_tta} ===", flush=True)
    except Exception as e:
        r1_tta = map_tta = None
        print(f"=== H12_EVAL_TTA size={size} skipped err={e} ===", flush=True)

    print(f"=== H12_SIZE_END size={size} stage=B elapsed={time.time() - t1:.0f}s total={time.time() - t0:.0f}s ===", flush=True)
    return ("ok", r1, m_ap, r1_tta, map_tta)


def main():
    print(f"=== H12_SWEEP_START sizes={SIZES} device={DEVICE} epochs_a={EPOCHS_A} epochs_b={EPOCHS_B} smoke={SMOKE} project={PROJECT} ===", flush=True)

    # Pre-warm cls checkpoints so a network blip mid-sweep doesn't kill a multi-hour pretrain.
    for sz in SIZES:
        asset = f"yolo26{sz}-cls.pt"
        try:
            path = attempt_download_asset(asset)
            print(f"=== H12_PREWARM ok {asset} -> {path} ===", flush=True)
        except Exception as e:
            print(f"=== H12_PREWARM FAIL {asset} err={e} ===", flush=True)

    results = []
    consecutive_fails = 0
    for sz in SIZES:
        try:
            row = (sz, *run_size(sz))
            results.append(row)
            consecutive_fails = 0
        except Exception as e:
            tb = traceback.format_exc()
            print(f"=== H12_SIZE_FAIL size={sz} err={e} ===\n{tb}", flush=True)
            results.append((sz, "fail", None, None, None, None))
            consecutive_fails += 1
            if consecutive_fails >= 2:
                print("=== H12_ABORT consecutive_failures>=2 ===", flush=True)
                break

    print("=== H12_SUMMARY ===", flush=True)
    print(f"{'size':>5}  {'status':<6}  {'R1':>7}  {'mAP':>7}  {'R1_TTA':>7}  {'mAP_TTA':>7}", flush=True)
    for r in results:
        sz, status, r1, m_ap, r1_tta, map_tta = r
        fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else "—"
        print(f"{sz:>5}  {status:<6}  {fmt(r1):>7}  {fmt(m_ap):>7}  {fmt(r1_tta):>7}  {fmt(map_tta):>7}", flush=True)
    print("=== H12_SWEEP_DONE ===", flush=True)


if __name__ == "__main__":
    main()
