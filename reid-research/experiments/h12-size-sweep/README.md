# H12 — ReID size sweep (n→x), 8-GPU DDP, on `ultra1`

Train the Ultralytics ReID person model at every YOLO scale (`n`, `s`, `m`, `l`, `x`) with the
champion 2-stage recipe (MSMT17 pretrain → Market-1501 FT), sequentially, all 8 GPUs DDP for
both stages. Background `nvidia-smi` sampler captures GPU memory throughout.

## Files

- `sweep.py` — single-process orchestrator. Loops over sizes; per size runs Stage A then B.
- `mem_logger.sh` — `nvidia-smi` sampler, 5s cadence, single CSV for the whole sweep.
- `../../remote_ultra1.py` — `ssh ultra1 -- …` / `scp` / `rsync` wrapper. Honors `~/.ssh/config`.

## Launch

From the local machine (this repo):

```bash
# 1. Push code to ultra1 (preferred path: git)
git push
python reid-research/remote_ultra1.py run "cd /home/rick/ultralytics && git pull"

# 2. Smoke test (size n, 1 epoch each stage, single GPU, fraction=0.01) — ~5 min
python reid-research/remote_ultra1.py run \
  "cd /home/rick/ultralytics && SMOKE=1 SIZES=n /home/rick/ultralytics/.venv/bin/python reid-research/experiments/h12-size-sweep/sweep.py 2>&1 | tee /home/rick/runs/reid/h12/smoke.log"

# 3. Start memory logger as a sidecar (detached screen)
python reid-research/remote_ultra1.py run \
  "screen -dmS h12mem bash /home/rick/ultralytics/reid-research/experiments/h12-size-sweep/mem_logger.sh /home/rick/runs/reid/h12/mem.csv 5"

# 4. Launch full sweep in a detached screen
python reid-research/remote_ultra1.py run \
  "screen -dmS h12 bash -lc 'cd /home/rick/ultralytics && PYTHONPATH=/home/rick/ultralytics /home/rick/ultralytics/.venv/bin/python reid-research/experiments/h12-size-sweep/sweep.py 2>&1 | tee /home/rick/runs/reid/h12/sweep.log'"

# 5. Watch (Ctrl-C to detach from tail; the screen keeps running)
python reid-research/remote_ultra1.py run "tail -F /home/rick/runs/reid/h12/sweep.log"

# 6. When done — kill logger, pull artifacts
python reid-research/remote_ultra1.py run "screen -X -S h12mem quit"
python reid-research/remote_ultra1.py pull /home/rick/runs/reid/h12/ /home/rick/runs/reid/h12/
```

## Recipe (locked, both stages 8-GPU DDP)

Stage A — MSMT17 pretrain, `imgsz=288`, `reid_p=128, reid_k=4`, `epochs=1277`, `lr0=3e-3`,
mirrors `h11-luperson-nl/pretrain.py`.

Stage B — Market-1501 FT, `imgsz=448`, `reid_p=8, reid_k=8`, `epochs=635`, `lr0=3.5e-3,
cos_lr=True`, lifted from `h8-imgsz-followups/ty2_msmt448_then_ft448.py` Stage B to 8-GPU.

⚠️  Per-rank PK sampling (`ultralytics/data/build.py:241-274`): user-supplied `batch=` is
silently overridden by `p*k` and applied per rank. Stage A on 8 ranks with `reid_p=128,
reid_k=4` yields 1 batch/epoch/rank (mirrors h11 quirk). Stage B yields global batch=512
(8× champion). Headline regression check at size `l` is the safety net.

## Env knobs

| var       | default     | effect                                       |
|-----------|-------------|----------------------------------------------|
| SIZES     | `n,s,m,l,x` | comma-separated list of scales to train      |
| SMOKE     | `0`         | `1`: 1 epoch / stage, `fraction=0.01`, 1 GPU |
| EPOCHS_A  | 1277        | Stage A epochs                               |
| EPOCHS_B  | 635         | Stage B epochs                               |
| PROJECT   | `/home/rick/runs/reid/h12` | output dir                  |
| DEVICE    | `0,1,2,3,4,5,6,7` | DDP device string                      |

## Wall-clock estimate (ultra1, 8× RTX PRO 6000)

- Stage A: ~6 h × 5 sizes = 30 h
- Stage B: ~1 h × 5 sizes = 5 h
- **Total: ~35 h** end-to-end. Plan to start on a Friday or split into two passes
  (`SIZES=n,s,m` first, then `SIZES=l,x`).

## Outputs (on ultra1)

```
/home/rick/runs/reid/h12/
├── mem.csv                      # continuous nvidia-smi log, 5s cadence
├── sweep.log                    # stdout/stderr with H12_* markers for slicing
├── h12_n_msmt_pretrain/weights/best.pt
├── h12_n_market_ft/weights/best.pt
├── h12_s_…/…
└── h12_x_market_ft/weights/best.pt
```

## Post-run plotting

Slice `mem.csv` by stage markers, then plot peak/avg memory per (size, stage):

```python
import re, pandas as pd
log = open("sweep.log").read()
markers = re.findall(r"H12_SIZE_(START|END) size=(\w+) stage=(\w+) ts=(\d+)", log)
mem = pd.read_csv("mem.csv", parse_dates=["timestamp"])
# split into (size, stage) buckets using markers, summarize memory_used_mib...
```

The `H12_SUMMARY` block at the end of `sweep.log` gives the R1 / mAP table for every size.
Cross-check `h12_l_market_ft` TTA R1 against the documented champion R1≈0.929 — large gap
implicates the 8× batch deviation on Stage B.
