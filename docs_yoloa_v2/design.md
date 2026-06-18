# YOLOA v2 — Design Doc

**Branch:** `yoloa_v2` · **Status:** Phase 0 implemented, ready for ultra6 runs · **Last updated:** 2026-05-29

> Runbook for the 4 Phase 0 experiments: see [phase0_commands.md](phase0_commands.md).

---

## 0. Goal

Build a YOLO-based anomaly detector with a **Heatmap-Guided Feature Fusion** module that can consume an external anomaly heatmap (from various sources) and inject it into the PAN features to refine detection.

Core insight:
```
Anomaly prior is NOT the final prediction.
Anomaly prior is a feature modulation signal for the YOLO head.
```

This doc covers **Phase 0** — a minimal validation experiment. SegBranch / MemoryBank / SemSeg priors are deferred to later phases, only if Phase 0 succeeds.

---

## 1. Phase 0 — Minimal Validation Experiment

### Why Phase 0 first

Before investing in SegBranch / MemoryBank / manual mask generators, we need to validate the **core hypothesis**:

> Injecting an anomaly heatmap into PAN features improves YOLO detection.

If fusion module itself doesn't help, all downstream prior generators are wasted effort. So Phase 0 strips everything down to the minimum: feed a **GT-derived mask** as heatmap (cheating, using bbox labels), and check if fusion contributes vs vanilla YOLO.

### Phase 0 Architecture

```
Image + bbox labels (.txt)
   │
   ├──→ Backbone (yolo26m, trainable)
   │         │
   │         ▼
   │      PAN Neck ──→ P3_1 (256×80×80)
   │                   P4_1 (512×40×40)
   │                   P5_1 (512×20×20)
   │                       │
   │                       ▼
   └──→ bbox→mask render ──→ HeatmapEncoder ──→ {AF_P3, AF_P4, AF_P5}
                                                       │
                              Heatmap-Guided Fusion (per-scale)
                                                       ↓
                                                  YOLO Head
                                                       ↓
                                                det loss only
```

### What's IN for Phase 0

- ✅ `bbox → mask` renderer (rectangle and Gaussian, configurable)
- ✅ HeatmapEncoder (per-scale: 1ch → 256 / 512 / 512)
- ✅ HeatmapGuidedFusion (multiplicative attention, revised formula — see §3.4)
- ✅ Mask dropout during training (anti-shortcut, see §3.5)
- ✅ Standard YOLO detection loss only

### What's OUT for Phase 0

- ❌ SegBranch (deferred to v2.x if needed)
- ❌ Segmentation loss
- ❌ α curriculum
- ❌ GT mask noise augmentation
- ❌ MemoryBank inference path (v2.1+)

---

## 2. Locked Decisions

| Decision | Choice | Note |
|---|---|---|
| Base model | `yolo26m.pt` | Pretrained init |
| Image size | 640 | Standard |
| Backbone | Trainable end-to-end | No freezing |
| Detection only loss | YOLO det (box + cls + dfl) | No seg loss |
| Mask render | Rectangle + Gaussian, config switchable | Two options, ablation friendly |
| HeatmapEncoder | Per-scale, 1ch → 256 / 512 / 512 | Three independent encoders, no resize |
| Fusion op | `P_out = P * 2 * sigmoid(AnomalyFeat)` | Passthrough at sigmoid(0)=0.5 → mult=1.0 |
| Fusion location | Post-Neck | Most stable; In-Neck / Dual deferred |
| Mask dropout | p=0.5 per-sample, fixed schedule | Anti-shortcut + built-in mask-off val |
| Baseline A (vanilla YOLO) | Skip — already known | User provides existing numbers |
| Dataset | User's own YOLO-format dataset | Single class `anomaly` |
| Mask source (Phase 0) | Always bbox-derived | No SegBranch/MB yet |

---

## 3. Module Design

### 3.1 bbox → mask renderer

Input: list of bboxes per image (XYWH normalized, YOLO format)
Output: `1 × H × W` mask (H=W=80 to match P3 scale)

Two modes, config switchable:

**(a) Rectangle (default)**
```
mask[bbox_region] = 1
mask[else] = 0
```

**(b) Gaussian**
```
For each bbox center (cx, cy) with size (w, h):
  σ_x = w / 4, σ_y = h / 4
  G(x, y) = exp(-((x-cx)² / (2σ_x²) + (y-cy)² / (2σ_y²)))
mask = max over all bboxes
```

Both render at 80×80 directly (image res 640 → mask res 80, stride 8).

### 3.2 HeatmapEncoder (per-scale, 3 independent)

For each PAN level (P3 / P4 / P5):

```
mask (1×H×W) at level scale (80/40/20)
       │
       ▼
   Conv(1, C, k=3, s=1) → GELU
       │
       ▼
   Conv(C, C, k=3, s=1) → GELU
       │
       ▼
   AnomalyFeat (C×H×W)
```

Three encoders:
- P3: 1 → 256
- P4: 1 → 512
- P5: 1 → 512

Mask is first downsampled to each level's spatial size (80→80 / 80→40 / 80→20) before feeding to its encoder.

### 3.3 HeatmapGuidedFusion (per-scale)

At each PAN level:
```python
P_out = P * 2 * torch.sigmoid(AnomalyFeat)
```

Properties of `2 * sigmoid(x)`:
- x = 0  →  multiplier = 1.0  →  **passthrough** (equivalent to no-fusion)
- x → +∞ →  multiplier = 2.0  →  emphasize feature
- x → -∞ →  multiplier = 0.0  →  suppress feature

This is the key revision from the original `1 + sigmoid(x)` formulation, which had multiplier=1.5 at x=0 (not passthrough). Passthrough at x=0 is critical for §3.5 mask dropout to work cleanly.

### 3.4 Mask dropout (anti-shortcut, the critical piece)

**Problem:** During training, heatmap comes from GT bbox. Model may shortcut: just read mask, ignore image features. Then at inference (where mask comes from a noisier source like MemoryBank or manual), performance collapses.

**Solution:** Randomly drop the mask during training. Force model to also work without it.

```python
# pseudocode in forward (per sample)
if training and random() < p_drop:
    AnomalyFeat = zeros  # or replace mask input with zeros before encoder
    # with our revised fusion: P_out = P * 2 * sigmoid(0) = P * 1.0 (passthrough)

P_out = P * 2 * sigmoid(AnomalyFeat)
```

Settings:
- **p_drop = 0.5**, per-sample independent, fixed (no schedule)
- Same det loss for both mask-on and mask-off branches
- No special handling — just a random switch per sample

**Bonus:** Built-in evaluation. Run val twice — with mask, without mask. Difference = fusion module's contribution. Replaces the originally-planned Experiment C (zero-mask leak check).

### 3.5 Where mask zeros are injected

Two implementation options — pick (a) for Phase 0:

| Option | Where | Trade-off |
|---|---|---|
| (a) **Zero AnomalyFeat directly** before fusion (skip encoder) | Cleanest passthrough, exact zero | Encoder doesn't see "null" examples |
| (b) Feed zero-mask through encoder | Encoder learns "no anomaly" representation | Conv biases → non-zero AnomalyFeat → not pure passthrough |

**Phase 0 uses (a).** Encoder sees only real masks; fusion is exact passthrough when dropped.

---

## 4. Experimental Protocol (Phase 0)

| Experiment | Train heatmap | Val heatmap | Purpose |
|---|---|---|---|
| **B-on**  | GT bbox-derived mask (p=0.5 dropout) | GT bbox-derived mask | **Upper bound** — does fusion help when prior is perfect? |
| **B-off** | (same model as B-on) | All-zero mask (mask dropped) | **No-mask baseline** — should approximate vanilla YOLO |
| **Existing A** | (user's prior YOLO baseline) | — | External reference |

**Success criteria for Phase 0**:
- **B-on > B-off**: fusion module contributes (the prior actually helps)
- **B-off ≈ A**: model didn't degrade when mask is removed; fusion is a pure add-on, not a crutch
- **B-on > A**: end-to-end improvement over vanilla YOLO when prior is available

If any criterion fails, the design needs rethinking before adding SegBranch / MemoryBank.

---

## 5. Loss

```
L_total = L_yolo_det = L_box + L_cls + L_dfl
```

Standard ultralytics detection loss. No seg loss in Phase 0.

---

## 6. Module Layout

| Module | Path | Status |
|---|---|---|
| `YOLOAnomalyV2Model` | `ultralytics/nn/tasks.py` | new (extend `DetectionModel`) |
| `BboxMaskRenderer` | `ultralytics/nn/modules/anomaly_v2.py` | new |
| `HeatmapEncoder` (per-scale) | `ultralytics/nn/modules/anomaly_v2.py` | new |
| `HeatmapGuidedFusion` | `ultralytics/nn/modules/anomaly_v2.py` | new |
| YAML config | `ultralytics/cfg/models/v2/yolo26-anomaly-v2.yaml` | new |
| Trainer | `ultralytics/models/yolo/anomaly_v2/train.py` | new |
| Validator | `ultralytics/models/yolo/anomaly_v2/val.py` | new (run val twice: w/ and w/o mask) |
| Predictor | `ultralytics/models/yolo/anomaly_v2/predict.py` | new |
| Public API | `ultralytics/models/yolo/model.py` (extend) | new `YOLOAnomalyV2` class |

Reuses existing ultralytics dataset loaders — no custom dataloader needed (only bbox labels, mask is rendered on-the-fly inside the model).

---

## 7. Implementation Order

1. **Modules**: `BboxMaskRenderer`, `HeatmapEncoder`, `HeatmapGuidedFusion` — pure nn.Module, no training logic, unit-testable with dummy tensors
2. **Model**: `YOLOAnomalyV2Model` — wire backbone + neck + fusion + head, dummy forward pass works
3. **YAML config**: assemble the architecture, `model.info()` and a dry forward run
4. **Mask dropout**: integrate into `YOLOAnomalyV2Model.forward()` (sample-level switch)
5. **Trainer**: thin extension of `DetectionTrainer`, no special loss
6. **Validator**: extends `DetectionValidator`, runs val twice (mask-on / mask-off)
7. **Public API**: `YOLOAnomalyV2` class in `model.py`
8. **Sanity run**: 10 epochs on user dataset, check loss curves, B-on vs B-off mAP gap
9. **Full run on ultra6**

---

## 8. Alternative methods (defer to follow-up experiments if Phase 0 succeeds)

These were considered but **not in Phase 0** — only worth trying if Phase 0 validates fusion is useful:

| Alternative | When to try | Notes |
|---|---|---|
| **GT mask noise injection** (dilation / blur) | If B-on > A but inference with real noisy mask underperforms | Train on imperfect mask, robust to noisy inference priors |
| **Gated residual** with learnable gate init=0 | If B-on/B-off gap is too small | fusion contribution ramps up gradually |
| **Two-stage training** (freeze backbone after vanilla, then train fusion) | If full e2e training is unstable | Slower, but cleaner separation |
| **In-Neck fusion** (§6.2 of original design) | If Post-Neck plateau | More powerful, harder to optimize |
| **Dual fusion** (In-Neck + Post-Neck) | After In-Neck baseline | Two leverage points |

---

## 9. Deferred Phases

| Phase | Adds | Triggered when |
|---|---|---|
| **v2.0** | Phase 0 design (this doc) | Now |
| **v2.1** | MemoryBank inference path; frozen DINOv2 encoder; `max` / β fusion of SegPred + MB | Phase 0 validates |
| **v2.2** | SegBranch (supervised), needs pixel mask data — α curriculum + GT mask noise | When pixel-mask dataset is available |

---

## 10. Training Setup (locked)

**Dataset (ultra6 only, no local copy):**
```
/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml
```

**Hyperparameters — identical to existing YOLO baseline:**
```
model        = yolo26m.pt
epochs       = 50
batch        = 96
device       = 0,1,2,3 (DDP)
optimizer    = MuSGD
lr0          = 0.00125
lrf          = 0.5
momentum     = 0.9
weight_decay = 0.0005
scale        = 0.1
copy_paste   = 0.1
mixup        = 0.0
close_mosaic = 20
save_json    = True
```

**Project / naming convention:**
- Project: `yoloa_v2` (separate from baseline's `yoloa`)
- Naming: `<size>_<arch>_<dataset>_<key_hparams>_v<n>`
- `<arch>` = `yoloav2`
- `<key_hparams>` extends with `rect`/`gauss` (mask render), `pd<NN>` (p_drop × 100)

**Phase 0 runs:**

| Run | Name | Mask render | p_drop |
|---|---|---|---|
| Primary B-on (rect) | `26m_yoloav2_v5_binary_cm20_rect_pd50_v1` | rectangle | 0.5 |
| Primary B-on (gauss) | `26m_yoloav2_v5_binary_cm20_gauss_pd50_v1` | Gaussian | 0.5 |
| Ablation: full shortcut | `26m_yoloav2_v5_binary_cm20_rect_pd0_v1` | rectangle | 0.0 |
| Sanity: full no-mask | `26m_yoloav2_v5_binary_cm20_rect_pd100_v1` | rectangle | 1.0 (mask always dropped → should ≈ baseline) |

**Baseline reference for comparison:**
```
nohupyolo train model=yolo26m.pt data=.../merge_data_v5_binary/data.yaml \
  epochs=50 batch=96 close_mosaic=20 device=0,1,2,3 optimizer=MuSGD \
  lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa name=26m_yolo_v5_binary_cm20_v1
```

**Sanity-check subset:** None. Full data + 50 epochs from the start (same as baseline, fair comparison).

**Local dev:** Code only — no local training. Push branch → run on ultra6.
