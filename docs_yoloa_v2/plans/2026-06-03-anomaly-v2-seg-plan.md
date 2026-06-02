# Anomaly v2 Segmentation Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `task=anomaly_v2_seg`, a segmentation variant of YOLOA v2 that consumes YOLO polygon labels and outputs instance segmentation, while keeping the softhint heatmap-bias mechanism. Target dataset: `merge_data_v6_binary`.

**Architecture:** New `YOLOAnomalyV2SegModel(SegmentationModel)` reuses the segment head + proto + `v8SegmentationLoss` from Ultralytics. The per-image GT mask prior comes from `batch["masks"]` (union of per-instance masks). `HeatmapBiasFusion` from softhint is reused unmodified to inject a bounded bias onto PAN features.

**Tech Stack:** Same as softhint — PyTorch, Ultralytics 8.x, Python 3.10, ultra6 (3-GPU DDP).

**Spec:** [`docs_yoloa_v2/specs/2026-06-03-anomaly-v2-seg-design.md`](../specs/2026-06-03-anomaly-v2-seg-design.md)

**Precondition:** softhint runs (`26m_yoloav2_softhint_rect_pd50_v1` and `..._seg_a1_v1`) have completed and reported. If softhint shows a regression, revisit branch base before continuing.

---

## File touch summary

| File | Action |
| --- | --- |
| `ultralytics/nn/tasks.py` | Add `YOLOAnomalyV2SegModel` class (after the existing `YOLOAnomalyV2Model`). |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml` | New YAML with `Segment` head. |
| `ultralytics/models/yolo/anomaly_v2_seg/__init__.py` | Package marker; exports trainer/validator/predictor. |
| `ultralytics/models/yolo/anomaly_v2_seg/train.py` | `AnomalyV2SegTrainer(SegmentationTrainer)`. |
| `ultralytics/models/yolo/anomaly_v2_seg/val.py` | `AnomalyV2SegValidator(SegmentationValidator)` with B-on/B-off passes. |
| `ultralytics/models/yolo/anomaly_v2_seg/predict.py` | `AnomalyV2SegPredictor(SegmentationPredictor)` with `set_external_mask_once` glue. |
| `ultralytics/models/yolo/model.py` | Register `anomaly_v2_seg` in `YOLO.task_map`. |
| `ultralytics/cfg/__init__.py` | Allow `task=anomaly_v2_seg` in CLI. |
| `scripts/softhint_seg_sanity.py` | β=0 equivalence smoke. |
| `scripts/false_prompt_seg_eval.py` | Seg-flavored false-prompt AUROC. |
| `docs_yoloa_v2/v6_seg_commands.md` | Runbook. |

**Not touched:** `ultralytics/nn/modules/anomaly_v2.py` (HeatmapBiasFusion reused as-is), `Detect` head, existing `anomaly_v2` task.

---

## Task 1: Branch off `yoloa_v2_softhint`

**Files:** none (git only)

- [ ] **Step 1: Confirm softhint runs are evaluated**
  - `26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt` exists and false-prompt AUROC measured.
  - If softhint regressed (mAP50-95 < baseline - 0.02), STOP and re-spec; don't branch yet.

- [ ] **Step 2: Create and switch branch**

```bash
cd /Users/louis/workspace/ultra_louis_work/ultralytics
git checkout yoloa_v2_softhint
git pull
git checkout -b yoloa_v2_seg
git push -u origin yoloa_v2_seg
```

---

## Task 2: Add `YOLOAnomalyV2SegModel` to tasks.py

**File:** `ultralytics/nn/tasks.py` (append near `YOLOAnomalyV2Model`)

- [ ] **Step 1: Read the existing `YOLOAnomalyV2Model` class (lines ~522–880)** to understand the existing fusion forward path. The new class mirrors it but inherits from `SegmentationModel`.

- [ ] **Step 2: Add the class. Place it right after `YOLOAnomalyV2Model` ends.**

```python
class YOLOAnomalyV2SegModel(SegmentationModel):
    """YOLO Anomaly v2 Segmentation — softhint heatmap-bias fusion on a Segment head.

    Extends SegmentationModel with the same heatmap-bias fusion mechanism as
    YOLOAnomalyV2Model (softhint branch): a bounded, low-bandwidth bias is added
    (broadcast over channels) to each PAN P3/P4/P5 feature before the Segment head.
    The Segment head, proto net, and v8SegmentationLoss are reused from upstream.

    Mask prior:
      - Training: union of per-instance masks from ``batch["masks"]`` (rasterized
        from polygon labels by the Ultralytics dataloader), combined per-image
        into a single ``(B, 1, H, W)`` tensor.
      - Inference: external mask provided via ``set_external_mask_once``.
      - Validation B-off: bias is None -> exact passthrough -> vanilla Segment.

    Mask dropout (anti-shortcut, see design.md §3.4): with probability ``p_drop``
    per sample, the bias is zeroed -> the model is forced to also perform without
    a mask.

    Spec: docs_yoloa_v2/specs/2026-06-03-anomaly-v2-seg-design.md.
    """

    def __init__(
        self,
        cfg="yolo26-anomaly-v2-seg.yaml",
        ch=3,
        nc=None,
        verbose=True,
        p_drop: float | None = None,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        v2_cfg = self.yaml.get("anomaly_v2", {}) if isinstance(self.yaml, dict) else {}
        p_drop = float(v2_cfg.get("p_drop", 0.5) if p_drop is None else p_drop)

        seg_head = self.model[-1]
        if not isinstance(seg_head, Segment):
            raise TypeError(f"YOLOAnomalyV2SegModel expects last layer to be Segment, got {type(seg_head).__name__}")

        pan_channels = []
        for cv2_seq in seg_head.cv2:
            first = cv2_seq[0]
            if hasattr(first, "conv") and isinstance(first.conv, torch.nn.Conv2d):
                pan_channels.append(first.conv.in_channels)
            else:
                raise RuntimeError(f"Unable to infer PAN channel from Segment.cv2[0]={type(first).__name__}")

        self.pan_from_indices = list(seg_head.f)
        self.pan_channels = pan_channels

        # Soft-hint fusion (reused unmodified from YOLOAnomalyV2Model).
        self.heatmap_bias_fusion = HeatmapBiasFusion(num_scales=seg_head.nl)

        self.p_drop = float(p_drop)

        # Transient mask state (mirrors YOLOAnomalyV2Model).
        self._mask_prior_buf = None  # (B, 1, H, W), set by loss() / set_external_mask_once
        self._mask_disabled_once = False

    # -----------------------------------------------------------------
    # Mask input API (mirrors YOLOAnomalyV2Model)
    # -----------------------------------------------------------------
    def set_mask_prior(self, mask: torch.Tensor):
        """Provide the (B, 1, H, W) mask prior for the next forward (training path)."""
        self._mask_prior_buf = mask
        self._mask_disabled_once = False

    def disable_mask_once(self):
        self._mask_prior_buf = None
        self._mask_disabled_once = True

    def set_external_mask_once(self, mask: torch.Tensor):
        if mask.dim() != 4 or mask.shape[1] != 1:
            raise ValueError(f"external mask must be (B, 1, H, W), got {tuple(mask.shape)}")
        self._mask_prior_buf = mask
        self._mask_disabled_once = False

    def _consume_mask_input(self):
        m = getattr(self, "_mask_prior_buf", None)
        disabled = getattr(self, "_mask_disabled_once", False)
        if hasattr(self, "_mask_prior_buf"):
            self._mask_prior_buf = None
            self._mask_disabled_once = False
        return m, disabled

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------
    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            # Build the per-image union-of-instance-masks prior from the seg dataloader output.
            prior = self._build_mask_prior(batch)
            self.set_mask_prior(prior)
            try:
                preds = self.forward(batch["img"])
            finally:
                self._mask_prior_buf = None
        return self.criterion(preds, batch)

    def _build_mask_prior(self, batch):
        """Per-image union of instance masks. Output shape (B, 1, H, W)."""
        masks = batch.get("masks", None)
        if masks is None or masks.numel() == 0:
            B = batch["img"].shape[0]
            # No instances anywhere -> all-zero prior.
            return torch.zeros(B, 1, batch["img"].shape[-2] // 4, batch["img"].shape[-1] // 4, device=batch["img"].device)
        batch_idx = batch["batch_idx"].long()
        B = batch["img"].shape[0]
        H, W = masks.shape[-2], masks.shape[-1]
        out = torch.zeros(B, 1, H, W, device=masks.device, dtype=masks.dtype)
        for b in range(B):
            sel = batch_idx == b
            if sel.any():
                out[b, 0] = masks[sel].amax(dim=0)
        return out

    # -----------------------------------------------------------------
    # Forward with heatmap bias inserted before the Segment head
    # -----------------------------------------------------------------
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        batch_size = x.shape[0]
        device = x.device

        mask, mask_disabled = self._consume_mask_input()

        # Per-sample mask dropout: zero the bias for some samples to force prior-free competence.
        p_drop = getattr(self, "p_drop", 0.0)
        keep = torch.ones(batch_size, device=device)
        if mask is not None and self.training and p_drop > 0.0:
            keep = (torch.rand(batch_size, device=device) > p_drop).to(keep.dtype)

        if mask_disabled:
            mask = None

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        last = self.model[-1]
        for m in self.model:
            if m is last:
                pan_inputs = [y[j] for j in m.f]
                fused = []
                for i, p in enumerate(pan_inputs):
                    if mask is None:
                        fused.append(p)
                        continue
                    target_h, target_w = p.shape[2], p.shape[3]
                    if mask.shape[-2] != target_h or mask.shape[-1] != target_w:
                        m_scale = torch.nn.functional.interpolate(
                            mask, size=(target_h, target_w), mode="bilinear", align_corners=False
                        )
                    else:
                        m_scale = mask
                    bias = self.heatmap_bias_fusion(m_scale, i)
                    bias = bias * keep.to(bias.dtype).view(-1, 1, 1, 1)
                    fused.append(p + bias)
                x = m(fused)
            else:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
```

Note: `Segment` class needs to be imported at the top of tasks.py (likely already imported indirectly; add if not).

- [ ] **Step 3: Smoke-test build (no forward yet)**

```bash
python -c "
import torch
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel
m = YOLOAnomalyV2SegModel('ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml', ch=3, nc=1, verbose=False)
print('built OK, params:', sum(p.numel() for p in m.parameters()))
print('bias_fusion beta:', m.heatmap_bias_fusion.beta.detach().tolist())
"
```

(Task 3 creates the YAML; this will fail until then. That's expected — run after Task 3.)

- [ ] **Step 4: Commit**

```bash
git add ultralytics/nn/tasks.py
git commit -m "yoloa_v2_seg: add YOLOAnomalyV2SegModel (softhint bias-fusion on SegmentationModel)"
```

---

## Task 3: Add segment YAML

**File:** `ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml`

- [ ] **Step 1: Create the YAML**

Base it on the existing `yolo26-seg.yaml` (backbone + head), but change `nc` source via `anomaly_v2` block:

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLO Anomaly v2 Segmentation — softhint heatmap-bias fusion + Segment head.
# Reads YOLO polygon labels via the standard Ultralytics seg dataloader.
# Branch: yoloa_v2_seg. Spec: docs_yoloa_v2/specs/2026-06-03-anomaly-v2-seg-design.md.

nc: 80
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

anomaly_v2:
  p_drop: 0.5

# Backbone (identical to yolo26-seg.yaml)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]

# Head — Segment instead of Detect
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]

  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]   # Segment(nc, nm=32, npr=256)
```

(Cross-reference `yolo26-seg.yaml` for exact Segment args; copy unchanged.)

- [ ] **Step 2: Build + Task-2 smoke**

```bash
python -c "
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel
m = YOLOAnomalyV2SegModel('ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml', ch=3, nc=1, verbose=False)
print('Segment head:', type(m.model[-1]).__name__)
print('bias_fusion params:', sum(p.numel() for p in m.heatmap_bias_fusion.parameters()))
"
```

Expected: `Segment head: Segment`, `bias_fusion params: 156`.

- [ ] **Step 3: Commit**

```bash
git add ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml
git commit -m "yoloa_v2_seg: add yolo26-anomaly-v2-seg.yaml (Segment head)"
```

---

## Task 4: Create `anomaly_v2_seg` package (trainer / validator / predictor)

**Files:** `ultralytics/models/yolo/anomaly_v2_seg/{__init__.py, train.py, val.py, predict.py}`

Mirror the existing `ultralytics/models/yolo/anomaly_v2/` package, which is the detection-task version. Read it first to understand the pattern.

- [ ] **Step 1: Read the existing pattern**

```bash
ls ultralytics/models/yolo/anomaly_v2/
cat ultralytics/models/yolo/anomaly_v2/__init__.py
cat ultralytics/models/yolo/anomaly_v2/train.py
cat ultralytics/models/yolo/anomaly_v2/val.py
cat ultralytics/models/yolo/anomaly_v2/predict.py
```

- [ ] **Step 2: Create the new package**

`__init__.py`:
```python
from .predict import AnomalyV2SegPredictor
from .train import AnomalyV2SegTrainer
from .val import AnomalyV2SegValidator

__all__ = "AnomalyV2SegPredictor", "AnomalyV2SegTrainer", "AnomalyV2SegValidator"
```

`train.py`:
```python
from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel


class AnomalyV2SegTrainer(yolo.segment.SegmentationTrainer):
    """Trainer for the anomaly_v2_seg task. Mirrors SegmentationTrainer but yields
    a YOLOAnomalyV2SegModel."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOAnomalyV2SegModel(cfg, nc=self.data["nc"], verbose=verbose and self.args.rank == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "seg_loss"
        return AnomalyV2SegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
```

`val.py`:
```python
from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel


class AnomalyV2SegValidator(yolo.segment.SegmentationValidator):
    """Validator that does B-on (mask prior provided) and B-off (no mask) passes."""

    # NOTE: Look at ultralytics/models/yolo/anomaly_v2/val.py for the B-on/B-off
    # double-pass implementation pattern. Same approach here, but the prior is
    # built via YOLOAnomalyV2SegModel._build_mask_prior(batch).

    def _build_prior(self, batch):
        # Defer to the model — same logic.
        return self.model._build_mask_prior(batch) if hasattr(self.model, "_build_mask_prior") else None
```

(NOTE for implementer: the actual B-on/B-off val loop pattern needs to be copied from `anomaly_v2/val.py`. The current sketch above just illustrates the structure; fill in the actual `__call__` / `update_metrics` overrides by reading the existing detection validator. Budget extra time for this — it's the most involved part.)

`predict.py`:
```python
from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel


class AnomalyV2SegPredictor(yolo.segment.SegmentationPredictor):
    """Segment predictor with set_external_mask_once glue."""
    # Mirrors anomaly_v2/predict.py. The base SegmentationPredictor handles
    # mask postprocessing; we only need to ensure the model exposes
    # set_external_mask_once (it does, via YOLOAnomalyV2SegModel).
```

- [ ] **Step 3: Commit**

```bash
git add ultralytics/models/yolo/anomaly_v2_seg/
git commit -m "yoloa_v2_seg: add trainer/validator/predictor package (mirrors anomaly_v2/)"
```

---

## Task 5: Register `anomaly_v2_seg` task

**Files:**
- `ultralytics/models/yolo/model.py` — extend `YOLO.task_map`
- `ultralytics/cfg/__init__.py` — extend the CLI's accepted-tasks list

- [ ] **Step 1: Add task_map entry in `model.py`**

Find the existing `"anomaly_v2": {...}` entry in `YOLO.task_map` (around line 119). After it, add:

```python
"anomaly_v2_seg": {
    "model": YOLOAnomalyV2SegModel,
    "trainer": yolo.anomaly_v2_seg.AnomalyV2SegTrainer,
    "validator": yolo.anomaly_v2_seg.AnomalyV2SegValidator,
    "predictor": yolo.anomaly_v2_seg.AnomalyV2SegPredictor,
},
```

Add `from ultralytics.nn.tasks import YOLOAnomalyV2SegModel` to imports.
Add `anomaly_v2_seg` to the `from ultralytics.models import yolo` namespace (it should auto-resolve via package import).

- [ ] **Step 2: Update CLI in `cfg/__init__.py`**

Search for where `"anomaly_v2"` appears (likely an allowed-tasks set or similar) and add `"anomaly_v2_seg"` next to it.

- [ ] **Step 3: Verify CLI**

```bash
python -c "from ultralytics import YOLO; m = YOLO('yolo26m-anomaly-v2-seg.yaml', task='anomaly_v2_seg'); print(type(m.model).__name__)"
```

Expected: `YOLOAnomalyV2SegModel`.

- [ ] **Step 4: Commit**

```bash
git add ultralytics/models/yolo/model.py ultralytics/cfg/__init__.py
git commit -m "yoloa_v2_seg: register anomaly_v2_seg task in YOLO.task_map and CLI"
```

---

## Task 6: Sanity script

**File:** `scripts/softhint_seg_sanity.py`

- [ ] **Step 1: Adapt `softhint_sanity.py`**

Same structure, but with the seg YAML and a `batch["masks"]`-style prior. Key checks:

1. Build `YOLOAnomalyV2SegModel` from the seg YAML.
2. Eval-mode forward with NO mask → record output.
3. Inject a non-trivial prior via `set_external_mask_once`, β still 0 → output MUST equal the no-mask forward exactly.
4. Set β=1.0, prior still on → output should differ.
5. PASS / FAIL exit code.

(Use the same `to_tensor` helper for tuple/raw output. For Segment models, `out` is a tuple `(decoded, proto)` or `(detections, proto)` depending on mode; compare both elements.)

- [ ] **Step 2: Run it**

```bash
cd /Users/louis/workspace/ultra_louis_work/ultralytics
python scripts/softhint_seg_sanity.py
```

Expected last line: `PASS softhint-seg sanity.`

- [ ] **Step 3: Commit**

```bash
git add scripts/softhint_seg_sanity.py
git commit -m "yoloa_v2_seg: add softhint_seg_sanity.py (beta=0 equivalence gate)"
```

---

## Task 7: False-prompt eval script for seg

**File:** `scripts/false_prompt_seg_eval.py`

- [ ] **Step 1: Adapt `false_prompt_eval.py`**

Differences from the detection version:
- Output to inspect is the predicted mask, not the bbox confidence.
- For each image, after prediction, take `result.masks.data` (the predicted instance masks) and record `max(mask_logit)` or `max(area of high-confidence pixels)`.
- AUROC computed over (anomaly + GT mask) vs (good + random mask) cohorts.

(Refer to Ultralytics' `Results.masks` API for the exact tensor shapes.)

- [ ] **Step 2: Commit**

```bash
git add scripts/false_prompt_seg_eval.py
git commit -m "yoloa_v2_seg: add false_prompt_seg_eval.py"
```

---

## Task 8: Runbook

**File:** `docs_yoloa_v2/v6_seg_commands.md`

- [ ] **Step 1: Write**

```markdown
# Anomaly v2 Seg — ultra6 Training Commands

Preconditions:
1. `conda activate ultra`
2. `set_wandb_true`
3. Branch `yoloa_v2_seg` checked out, clean
4. Softhint runs completed and false-prompt evaluated

## 1. Softhint-seg main on v6_binary

```
nohupyolo train task=anomaly_v2_seg \
  model=yolo26m-anomaly-v2-seg.yaml \
  pretrained=yolo26m.pt \
  data=/data/shared-datasets/louis_data/AnomalyDataset/merge_data_v6_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=0,1,2 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2_seg name=26m_yoloav2seg_v6binary_pd50_v1
```

## 2. Vanilla yolo26m-seg baseline on v6_binary (for comparison)

```
nohupyolo train task=segment \
  model=yolo26m-seg.yaml \
  pretrained=yolo26m.pt \
  data=/data/shared-datasets/louis_data/AnomalyDataset/merge_data_v6_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=3,4,5 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2_seg name=26m_seg_v6binary_v1
```

## 3. Pass criteria

- Softhint-seg mAP50-mask within 0.02 of vanilla `yolo26m-seg` baseline (architecture non-destructive).
- False-prompt AUROC ≥ 0.7 (bias suppresses false positives on clean images).
- β values at e20 are finite.

## 4. v6_multiclass (later)

Once v6_binary is validated, swap `data=...v6_multiclass/data.yaml` and remove the YAML's hard-coded nc — let dataloader pick up nc=64 from data.yaml.
```

- [ ] **Step 2: Commit**

```bash
git add docs_yoloa_v2/v6_seg_commands.md
git commit -m "yoloa_v2_seg: add v6_seg runbook"
```

---

## Task 9: Push, update memory

- [ ] **Step 1: Push**

```bash
git push
```

- [ ] **Step 2: Update `~/.claude/projects/.../memory/project_yoloa_v2.md`**

Append a paragraph noting:
- Branch `yoloa_v2_seg` exists off `yoloa_v2_softhint`.
- New task `anomaly_v2_seg` registered.
- Pending: launch the two runs (softhint-seg + vanilla seg baseline) per runbook once softhint AUROC results are in.

---

## Self-review checklist

Plan was self-reviewed against the spec:

- **Spec §3.1 (`YOLOAnomalyV2SegModel(SegmentationModel)`)** — Task 2.
- **Spec §3.2 (per-image union of `batch["masks"]`)** — Task 2 step 2, `_build_mask_prior`.
- **Spec §3.3 (new task plumbing)** — Tasks 3, 4, 5.
- **Spec §3.4 (HeatmapBiasFusion reused unmodified)** — explicit in Task 2 (imports from softhint), `anomaly_v2.py` NOT touched.
- **Spec §4 (training recipe)** — Task 8 runbook.
- **Spec §4 sanity gate** — Task 6.
- **Spec §5 (two comparison runs)** — Task 8.
- **Spec §6 (false-prompt seg eval)** — Task 7.
- **Spec §7 (out of scope)** — respected.
- **Spec §8 file touch list** — all 10 items covered.

Open caveats:
- Task 4 validator (B-on/B-off pattern) is the riskiest piece — implementer must read `anomaly_v2/val.py` carefully and replicate the double-pass logic.
- Task 5 task registration may need additional auto-import wiring in `ultralytics/models/yolo/__init__.py` (mirror how `anomaly_v2` is exposed).
- Spec assumes `batch["masks"]` is `(N, H, W)` with `batch_idx` indexing — verify against actual seg dataloader output in step 1 of Task 2 (read existing seg trainer code).
