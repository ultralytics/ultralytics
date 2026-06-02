# Anomaly v2 — Full Segmentation Adaptation

**Date:** 2026-06-03
**Branch (impl):** `yoloa_v2_seg` (off `yoloa_v2_softhint`, AFTER softhint runs report)
**Status:** Spec — awaiting implementation plan

---

## 1. Problem

YOLOA v2 (`task=anomaly_v2`, detection-style) cannot consume the new v6 datasets:
- `merge_data_v6_binary` (`task: segment`, `nc=1`, ~128K train images)
- `merge_data_v6_multiclass` (`task: segment`, `nc=64`)

Both ship YOLO polygon labels (`class x1 y1 x2 y2 ... xn yn`), not bbox labels. Current `YOLOAnomalyV2Model` outputs detection only; its mask prior is a rect/gauss bbox-rasterization (`BboxMaskRenderer`) — coarser than the actual polygon when polygons are available.

Goal: a parallel task `anomaly_v2_seg` that does **full segmentation** (output = bbox + per-instance mask) while preserving the softhint heatmap-bias mechanism that was validated on `anomaly_v2`. Initial deployment target = `v6_binary`; `v6_multiclass` (nc=64) follows once the architecture is proven on nc=1.

## 2. Goal

- Read YOLO polygon labels via Ultralytics' existing segmentation dataloader (no custom label parsing needed — the dataloader already rasterizes polygons into `batch["masks"]`).
- Use the **rasterized GT mask** (union of per-image instance masks) as the heatmap prior, fed into `HeatmapBiasFusion` exactly as on `yoloa_v2_softhint`. This is strictly more informative than the bbox-rendered rect/gauss prior used by `anomaly_v2`.
- Output instance segmentation (bbox + mask coefficient + prototype) via Ultralytics' `Segment` head.
- Preserve **everything else** that worked on softhint: bounded additive bias on PAN features, β init 0 (vanilla on launch), mask dropout (`p_drop`), external-mask injection for inference, optional SegBranch (deferred — see §7).
- New task name: **`anomaly_v2_seg`**, alongside (not replacing) `anomaly_v2`.

**Not required (deferred):**
- Polygon-aware prior renderer (Ultralytics' rasterizer is already polygon-faithful).
- nc=64 multiclass training in the first run — binary first to validate the architecture.

## 3. Design

### 3.1 Model — `YOLOAnomalyV2SegModel`

New class in `ultralytics/nn/tasks.py`, extending Ultralytics' `SegmentationModel` (NOT `DetectionModel`). Mirrors the structure of `YOLOAnomalyV2Model` but with three differences:

1. **Base class:** `SegmentationModel` → inherits the `Segment` head construction, proto net, and `v8SegmentationLoss`.
2. **No `BboxMaskRenderer`:** the prior comes from `batch["masks"]` (Ultralytics-rasterized polygon → mask), combined per-image into a single `(B, 1, H, W)` tensor by union (logical OR / max-reduce across instances belonging to the same image).
3. **Loss path:** delegates to `v8SegmentationLoss` via `init_criterion`. No custom seg loss needed; the only change is feeding the GT-mask-derived prior into `HeatmapBiasFusion` during forward.

Reused unchanged from softhint:
- `HeatmapBiasFusion` (1→8→1 conv + per-scale β init 0, tanh-bounded).
- The `_predict_once` PAN-bias loop: `P + bias` broadcast over channels.
- `set_mask_input` / `disable_mask_once` / `set_external_mask_once` APIs.
- Mask dropout (`p_drop`) zeroing per-sample bias.

### 3.2 Per-image prior construction

Ultralytics' seg dataloader produces `batch["masks"]` shaped `(N, H, W)` where `N = sum of instances across the batch`; `batch["batch_idx"]` gives the image index per instance.

`YOLOAnomalyV2SegModel.loss()` reduces this to a `(B, 1, H, W)` per-image prior:

```python
def _build_mask_prior(self, batch):
    masks = batch["masks"]  # (N, H_lbl, W_lbl), 0/1 floats
    batch_idx = batch["batch_idx"].long()  # (N,)
    B = batch["img"].shape[0]
    out = torch.zeros(B, 1, masks.shape[-2], masks.shape[-1], device=masks.device, dtype=masks.dtype)
    if masks.numel() == 0:
        return out
    for b in range(B):
        sel = batch_idx == b
        if sel.any():
            out[b, 0] = masks[sel].amax(dim=0)   # union across instances
    return out
```

For inference the existing `set_external_mask_once(mask)` API is reused — caller passes a `(B, 1, H, W)` mask directly (rasterized polygon or user-drawn).

### 3.3 New task plumbing

| File | Action |
| --- | --- |
| `ultralytics/nn/tasks.py` | Add `YOLOAnomalyV2SegModel(SegmentationModel)`. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml` | New YAML — same backbone+PAN as softhint; `Segment` head instead of `Detect`; `anomaly_v2:` config block. |
| `ultralytics/models/yolo/anomaly_v2_seg/__init__.py` | Package marker. |
| `ultralytics/models/yolo/anomaly_v2_seg/train.py` | `AnomalyV2SegTrainer(SegmentationTrainer)` — overrides `get_model` to return `YOLOAnomalyV2SegModel`; otherwise inherits. |
| `ultralytics/models/yolo/anomaly_v2_seg/val.py` | `AnomalyV2SegValidator(SegmentationValidator)` — adds B-on / B-off val passes, same pattern as `AnomalyV2Validator`. |
| `ultralytics/models/yolo/anomaly_v2_seg/predict.py` | `AnomalyV2SegPredictor(SegmentationPredictor)` — adds the `set_external_mask_once` glue, same pattern as `AnomalyV2Predictor`. |
| `ultralytics/models/yolo/model.py` `YOLO.task_map` | Register `"anomaly_v2_seg"` entry. |
| `ultralytics/cfg/__init__.py` | Register task name in CLI accept list. |
| `ultralytics/cfg/default.yaml` | Add `"anomaly_v2_seg"` to allowed `task:` values if there's an enum. |

### 3.4 What does NOT change

- `HeatmapBiasFusion` (`anomaly_v2.py`) — reused as-is.
- `BboxMaskRenderer` — still present for the original detection task (`anomaly_v2`); not used by `anomaly_v2_seg`.
- `Detect` head — untouched.
- Existing `anomaly_v2` task (detection) — unaffected; runs on `yoloa_v2_softhint` continue to work.

## 4. Training

Same MuSGD recipe as softhint; only data and YAML differ.

| Setting | Value |
| --- | --- |
| backbone init | `yolo26m.pt` (transfer from detection pretrain) |
| task | `anomaly_v2_seg` |
| dataset | `merge_data_v6_binary` first; `merge_data_v6_multiclass` follows |
| epochs | **20** for first run |
| batch / device | 96 / 3 GPUs (mirrors softhint hparams) |
| optimizer / lr | MuSGD, lr0=0.00125, lrf=0.5, momentum=0.9, wd=0.0005 |
| close_mosaic | 20 |
| p_drop | 0.5 |
| mask_shuffle_p / mask_noise_std | 0.0 / 0.0 |

**Sanity gate:** β=0 forward must equal vanilla `yolo26-seg` (same shape, same logits). Adapt `scripts/softhint_sanity.py` → `scripts/softhint_seg_sanity.py`.

## 5. First-batch comparison runs

| Run | Branch | Dataset | Purpose |
| --- | --- | --- | --- |
| `26m_yoloav2seg_v6binary_pd50_v1` | `yoloa_v2_seg` | v6_binary | Main result. Compare against vanilla `yolo26m-seg` on the same dataset. |
| `26m_yolo26mseg_v6binary_v1` | (any) | v6_binary | Vanilla seg baseline — no anomaly knobs, no bias. Required for apples-to-apples mAP-mask comparison. |

`v6_multiclass` (nc=64) is queued for run-2 after the binary architecture is proven.

## 6. Evaluation

Same metrics as Ultralytics seg validator + a seg-aware false-prompt benchmark.

1. **Standard seg metrics:** mAP50, mAP50-95, mask AP (auto-emitted by `SegmentationValidator`).
2. **False-prompt for seg:** new script `scripts/false_prompt_seg_eval.py`. For each *good* image (no annotated anomaly), feed a synthetic random rect mask as the external prior and record `max(mask_logit)` across predicted masks. AUROC vs (anomaly + GT mask) cohort.
3. **β values at e20:** logged from the checkpoint — finite + bounded.

**Pass criteria:**
- mAP50-mask within 0.02 of vanilla `yolo26m-seg` baseline on `v6_binary` (proves architecture is non-destructive).
- False-prompt AUROC ≥ 0.7 (proves the bias actually suppresses false positives on clean images).

## 7. Out of scope

- Polygon-faithful prior renderer (instead of Ultralytics' rasterizer). Defer — the rasterizer is already polygon-faithful.
- SegBranch in `anomaly_v2_seg`. Skip — the prior IS the GT mask during training and external during inference; an internal predictor adds no value and Phase 2 already showed SegBranch fails.
- nc=64 multiclass first. Run after binary architecture validates.
- bbox-only mode on v6 data. Skip — full-seg is the whole point.
- Anything from the softhint §7 deferred list (backbone-mid injection, cross-attn) — same deferrals apply.

## 8. File touch list (for plan)

| File | Action |
| --- | --- |
| `ultralytics/nn/tasks.py` | Add `YOLOAnomalyV2SegModel`. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml` | New YAML, Segment head. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg-multiclass.yaml` | (Optional, lands later) Same YAML, override nc=64 in args. |
| `ultralytics/models/yolo/anomaly_v2_seg/{__init__,train,val,predict}.py` | New package, ~50 lines per file. |
| `ultralytics/models/yolo/model.py` | Register `anomaly_v2_seg` in `YOLO.task_map`. |
| `ultralytics/cfg/__init__.py` | Allow `task=anomaly_v2_seg`. |
| `scripts/softhint_seg_sanity.py` | β=0 equivalence test for seg. |
| `scripts/false_prompt_seg_eval.py` | Seg-flavored false-prompt benchmark. |
| `docs_yoloa_v2/v6_seg_commands.md` | Runbook. |

## 9. Decision summary

- Full seg adaptation (option C from brainstorming), not bbox-only or hybrid.
- Branch off `yoloa_v2_softhint` (option Q2-B). Wait for softhint runs to report before branching — if softhint regresses, we may want to branch off a different point.
- Use Ultralytics' rasterized `batch["masks"]` as the prior (no custom polygon renderer).
- Reuse `HeatmapBiasFusion` unmodified.
- Drop SegBranch in seg task (the prior IS the GT mask).
- Binary first, multiclass after.
