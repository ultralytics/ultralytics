# Soft-Hint Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace YOLOA v2's per-channel multiplicative gate (`P * 2·sigmoid(AF)`) with a bounded, 1-channel additive bias on the PAN features (`P + bias`, broadcast to all C channels) on a new branch `yoloa_v2_softhint` for cross-branch comparison.

**Architecture:** New `HeatmapBiasFusion` module (1→8→1 conv + 3 learnable per-scale β scalars, init 0) produces a bounded per-pixel bias added directly to PAN P3/P4/P5 features before the (unchanged) Detect head. `BboxMaskRenderer`, `SegBranch`, mask dropout, and shuffle/noise augments remain intact.

**Tech Stack:** PyTorch, Ultralytics 8.x, Python 3.10, ultra6 (3-GPU DDP) for training, MVTec-style anomaly dataset (`merge_data_v5_binary`).

**Spec:** [`docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md`](../specs/2026-06-02-softhint-fusion-design.md)

---

## File touch summary

| File | Action |
| --- | --- |
| `ultralytics/nn/modules/anomaly_v2.py` | Delete `HeatmapEncoder`, `HeatmapGuidedFusion`; add `HeatmapBiasFusion`. Update `__all__` and `__main__` smoke. |
| `ultralytics/nn/modules/__init__.py` | Replace `HeatmapEncoder` / `HeatmapGuidedFusion` import+export with `HeatmapBiasFusion`. |
| `ultralytics/nn/tasks.py` `YOLOAnomalyV2Model` | Remove `heatmap_encoders` / `heatmap_fusions` ModuleLists; add single `heatmap_bias_fusion`; in `_predict_once`, replace per-scale multiplicative gate with `P + bias`. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml` | New YAML. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml` | New YAML, same as above + `seg_branch: true`, `seg_alpha_mode: pinned_one`. |
| `scripts/softhint_sanity.py` | β=0 forward equivalence test. |
| `scripts/false_prompt_eval.py` | Max-conf AUROC over (real anomaly + GT mask) vs (good image + random mask). |
| `docs_yoloa_v2/phase0_softhint_commands.md` | Runbook: train + eval commands for both runs. |

**Not touched:** `ultralytics/nn/modules/head.py`. The Detect head is reused as-is.

---

## Task 1: Branch off `yoloa_v2`

**Files:** none (git only)

- [ ] **Step 1: Verify clean tree (or accept pre-existing diffs)**

```bash
cd /Users/louis/workspace/ultra_louis_work/ultralytics
git status --short
```

Expected: only the pre-existing `M docs_yoloa_v2/demo_mask_prompt.py`, `?? CLAUDE.md`, `?? tools/` lines. Those are unrelated; leave them.

- [ ] **Step 2: Create and switch to the new branch**

```bash
git checkout -b yoloa_v2_softhint
git branch --show-current
```

Expected output: `yoloa_v2_softhint`

- [ ] **Step 3: Push to origin (no commits yet — empty branch tracking)**

```bash
git push -u origin yoloa_v2_softhint
```

Expected: branch published, upstream set.

---

## Task 2: Implement `HeatmapBiasFusion` and remove old fusion classes

**Files:**
- Modify: `ultralytics/nn/modules/anomaly_v2.py` (entire file — delete two classes, add one, update `__all__` and `__main__`)

- [ ] **Step 1: Replace the file**

Overwrite `ultralytics/nn/modules/anomaly_v2.py` with:

```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap bias fusion, optional SegBranch.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("BboxMaskRenderer", "HeatmapBiasFusion", "SegBranch", "binary_seg_loss")


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        self.sigma_factor = float(sigma_factor)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        mask = torch.zeros(batch_size, 1, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return mask

        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)

        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H

        if self.mode == "rect":
            x1 = (cx - w / 2)[:, None, None]
            x2 = (cx + w / 2)[:, None, None]
            y1 = (cy - h / 2)[:, None, None]
            y2 = (cy + h / 2)[:, None, None]
            inside = (
                (self.grid_x[None] >= x1)
                & (self.grid_x[None] < x2)
                & (self.grid_y[None] >= y1)
                & (self.grid_y[None] < y2)
            ).to(dtype)
        else:  # gauss
            sigma_x = (w * self.sigma_factor).clamp(min=0.5)
            sigma_y = (h * self.sigma_factor).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(
                -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
            )

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor={self.sigma_factor}"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask -> bounded per-pixel bias broadcast onto PAN features.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The conv stack is SHARED across PAN scales; the caller
    is responsible for resizing the mask to each scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap.

    Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        return self.beta[scale_idx] * torch.tanh(self.conv(mask))


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Consumes the P3 and P4 PAN features and emits per-pixel logits at P3 resolution
    (e.g. 80x80 for 640 input). A P4 auxiliary head provides deep supervision during
    training.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None):
        super().__init__()
        self.nc = nc
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0], c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor]):
        logits = self.classifier(x[0])
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(x[1])
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap."""
    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    card = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
    loss = bce + dice
    if aux_logits is not None:
        aux_t = target
        if aux_t.shape[2:] != aux_logits.shape[2:]:
            aux_t = F.interpolate(target, size=aux_logits.shape[2:], mode="nearest")
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits, aux_t)
    return loss


if __name__ == "__main__":
    # Smoke test: render, fusion init=0 passthrough, fusion with learned beta bounded,
    # broadcast onto a C-channel PAN feature.
    B, H = 4, 80
    renderer = BboxMaskRenderer(mask_size=H, mode="rect")
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)
    mask = renderer(bboxes, batch_idx, B)
    assert mask.shape == (B, 1, H, H)
    assert mask[1].sum().item() == 0.0  # image with no bboxes

    fusion = HeatmapBiasFusion(num_scales=3)
    # beta init=0 -> bias is exactly zero for every scale
    for s in range(3):
        bias = fusion(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} bias not zero at init"
    print("HeatmapBiasFusion init=0 passthrough OK.")

    # With beta set non-zero, output is bounded in [-beta, +beta]
    with torch.no_grad():
        fusion.beta.fill_(1.5)
    bias = fusion(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "bias exceeded beta after tanh"
    print(f"HeatmapBiasFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    # Broadcast onto a C-channel PAN feature: (B, 1, H, W) + (B, C, H, W) -> (B, C, H, W).
    p = torch.randn(B, 256, H, H)
    p_fused = p + fusion(mask, 0)
    assert p_fused.shape == p.shape
    print(f"Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> {tuple(p_fused.shape)}.")

    # Resize-per-scale smoke
    mask_p4 = F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False)
    mask_p5 = F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False)
    assert fusion(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapBiasFusion multi-scale OK.")

    print("\nAll smoke tests passed.")
```

- [ ] **Step 2: Run the smoke test**

```bash
cd /Users/louis/workspace/ultra_louis_work/ultralytics
python -m ultralytics.nn.modules.anomaly_v2
```

Expected (all four lines must appear, no asserts fire):
```
HeatmapBiasFusion init=0 passthrough OK.
HeatmapBiasFusion beta=1.5 bounded OK (max abs = ...).
Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> (4, 256, 80, 80).
HeatmapBiasFusion multi-scale OK.

All smoke tests passed.
```

- [ ] **Step 3: Commit**

```bash
git add ultralytics/nn/modules/anomaly_v2.py
git commit -m "yoloa_v2_softhint: replace HeatmapEncoder+HeatmapGuidedFusion with HeatmapBiasFusion

Per-channel multiplicative gate (P * 2*sigmoid(AF)) was rewriting PAN features
and forcing detections wherever a heatmap was provided (observed in demo with
wrong-location prompts).

HeatmapBiasFusion is a single 1->8->1 conv shared across scales plus three
learnable scalars (beta init 0) that scale a tanh-bounded per-pixel bias. The
bias is added (broadcast over channels) to PAN features before the Detect head;
Detect itself is unchanged."
```

---

## Task 3: Update `nn/modules/__init__.py`

**Files:**
- Modify: `ultralytics/nn/modules/__init__.py:108-115` and `:166-172`

- [ ] **Step 1: Edit the imports**

In `ultralytics/nn/modules/__init__.py`, find the block (around lines 108–115):

```python
from .anomaly_v2 import (
    BboxMaskRenderer,
    HeatmapEncoder,
    HeatmapGuidedFusion,
    SegBranch,
    binary_seg_loss,
)
```

Replace with:

```python
from .anomaly_v2 import (
    BboxMaskRenderer,
    HeatmapBiasFusion,
    SegBranch,
    binary_seg_loss,
)
```

And in the `__all__` block (around lines 166–172):

```python
    "HeatmapEncoder",
    "HeatmapGuidedFusion",
```

Replace those two lines with:

```python
    "HeatmapBiasFusion",
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from ultralytics.nn.modules import HeatmapBiasFusion, BboxMaskRenderer, SegBranch; print('imports OK')"
```

Expected: `imports OK`

```bash
python -c "from ultralytics.nn.modules import HeatmapEncoder" 2>&1 | tail -1
```

Expected: an `ImportError` mentioning `HeatmapEncoder`.

- [ ] **Step 3: Commit**

```bash
git add ultralytics/nn/modules/__init__.py
git commit -m "yoloa_v2_softhint: update nn/modules exports for HeatmapBiasFusion"
```

---

## Task 4: Rewire `YOLOAnomalyV2Model` to use `HeatmapBiasFusion`

**Files:**
- Modify: `ultralytics/nn/tasks.py` (the `YOLOAnomalyV2Model` class, currently lines ~522–880)

- [ ] **Step 1: Edit the imports block (top of file)**

Find the import block that includes `HeatmapEncoder` and `HeatmapGuidedFusion` (currently around lines 55–58). Replace those two names with `HeatmapBiasFusion`. The resulting import should look like:

```python
from ultralytics.nn.modules import (
    ...
    BboxMaskRenderer,
    HeatmapBiasFusion,
    SegBranch,
    binary_seg_loss,
    ...
)
```

(Use Edit, not Write — keep all other imports intact.)

- [ ] **Step 2: Edit `YOLOAnomalyV2Model.__init__`**

Inside the constructor (around lines 608–611), replace:

```python
        # Anomaly-side modules (live outside self.model so they are not in the Sequential)
        self.mask_renderer = BboxMaskRenderer(mask_size=mask_size, mode=mask_mode, sigma_factor=sigma_factor)
        self.heatmap_encoders = torch.nn.ModuleList([HeatmapEncoder(c_out=c) for c in pan_channels])
        self.heatmap_fusions = torch.nn.ModuleList([HeatmapGuidedFusion() for _ in pan_channels])
```

With:

```python
        # Anomaly-side modules (live outside self.model so they are not in the Sequential)
        self.mask_renderer = BboxMaskRenderer(mask_size=mask_size, mode=mask_mode, sigma_factor=sigma_factor)
        # Soft-hint fusion: bounded per-pixel bias added (broadcast) to PAN features.
        # beta init 0 -> training starts as vanilla YOLO; the model learns to lean on
        # the heatmap only if it helps the detection loss.
        self.heatmap_bias_fusion = HeatmapBiasFusion(num_scales=detect.nl)
```

Also update the class docstring (around lines 522–545) to match the new mechanism. Replace the docstring with:

```python
    """YOLO Anomaly v2 — detection + soft-hint heatmap fusion (yoloa_v2_softhint branch).

    Extends DetectionModel with two anomaly-side modules attached OUTSIDE the parsed
    Sequential:
      - ``mask_renderer``: rasterizes GT bboxes into a 1-channel mask.
      - ``heatmap_bias_fusion``: HeatmapBiasFusion. Produces a bounded per-pixel bias
        added (broadcast over channels) to each PAN P3/P4/P5 feature before the Detect
        head. Both reg and cls branches see the bias; this is acceptable because the
        bias is bounded and additive, unlike the previous multiplicative amplifier.

    Mask dropout (anti-shortcut, see design.md §3.4):
      During training, with probability ``p_drop`` per sample, the bias for that
      sample is zeroed -> exact passthrough -> model is forced to also perform
      without a mask.

    Mask source:
      - Training: rendered from ``batch["bboxes"]`` (set by ``loss()``).
      - Validation B-on: caller sets bboxes via ``set_mask_input()``.
      - Validation B-off / pure inference: no bboxes -> bias is None -> PAN features
        flow through unchanged (vanilla YOLO).
      - External (e.g. SegBranch, user prompt): ``set_external_mask_once``.

    Spec: docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
    """
```

- [ ] **Step 3: Rewrite the fusion section of `_predict_once`**

Inside `_predict_once`, find the block that starts with `# Apply fusion to the PAN inputs before Detect.` (currently around lines 804–845). Replace the per-scale `for i, p in enumerate(pan_inputs):` loop with the simpler additive form:

```python
                # Apply soft-hint fusion to the PAN inputs before Detect.
                # m.f is a list of indices into y (the PAN P3/P4/P5 outputs).
                pan_inputs = [y[j] for j in m.f]
                # SegBranch (optional) predicts the heatmap from P3/P4. Run unconditionally when present
                # so the seg loss can be computed by loss().
                seg_branch = getattr(self, "seg_branch", None)
                seg_logits_buf = getattr(self, "_seg_logits_buf", None)
                if seg_branch is not None:
                    seg_logits_buf = seg_branch([pan_inputs[0], pan_inputs[1]])
                    self._seg_logits_buf = seg_logits_buf
                mask = self._resolve_fusion_mask(
                    bboxes, batch_idx, external_mask, mask_disabled, seg_logits_buf, batch_size, device
                )
                if (
                    self.training
                    and mask is not None
                    and bboxes is not None
                    and external_mask is None
                    and (self.mask_shuffle_p > 0.0 or self.mask_noise_std > 0.0)
                ):
                    mask = self._augment_mask(mask)
                fused = []
                for i, p in enumerate(pan_inputs):
                    if mask is None:
                        # Pure passthrough: skip fusion entirely.
                        fused.append(p)
                        continue
                    target_h, target_w = p.shape[2], p.shape[3]
                    if mask.shape[2] != target_h or mask.shape[3] != target_w:
                        m_scale = torch.nn.functional.interpolate(
                            mask, size=(target_h, target_w), mode="bilinear", align_corners=False
                        )
                    else:
                        m_scale = mask
                    bias = self.heatmap_bias_fusion(m_scale, i)  # (B, 1, H, W) in [-beta_i, +beta_i]
                    # Per-sample keep mask (mask dropout): dropped samples get zero bias.
                    bias = bias * keep.to(bias.dtype).view(-1, 1, 1, 1)
                    # Broadcast 1-channel bias over the C channels of the PAN feature.
                    fused.append(p + bias)
                x = m(fused)
```

(All non-replaced code in `_predict_once` — the bookkeeping for `y`, `dt`, `embeddings`, profile, visualize, embed — stays unchanged.)

- [ ] **Step 4: Smoke-test the model end-to-end**

```bash
python -c "
import torch
from ultralytics.nn.tasks import YOLOAnomalyV2Model

m = YOLOAnomalyV2Model('ultralytics/cfg/models/v2/yolo26-anomaly-v2.yaml', ch=3, nc=1, verbose=False)
m.eval()
x = torch.randn(2, 3, 640, 640)

# 1) mask-off: no mask input -> bias=None -> vanilla path.
out_off = m(x)
def to_t(o):
    return o[0] if isinstance(o, tuple) else o

# 2) mask-on with beta=0: must be numerically identical to mask-off.
bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
batch_idx = torch.tensor([0], dtype=torch.long)
m.set_mask_input(bboxes, batch_idx)
out_on_beta0 = m(x)
diff0 = (to_t(out_off) - to_t(out_on_beta0)).abs().max().item()
print(f'mask-off vs mask-on (beta=0) max abs diff = {diff0:.2e}')
assert diff0 < 1e-5, 'beta=0 forward must equal mask-off forward'

# 3) Set beta != 0, mask-on should now differ.
with torch.no_grad():
    m.heatmap_bias_fusion.beta.fill_(1.0)
m.set_mask_input(bboxes, batch_idx)
out_on_beta1 = m(x)
diff1 = (to_t(out_off) - to_t(out_on_beta1)).abs().max().item()
print(f'mask-on (beta=1.0) vs mask-off diff = {diff1:.4f}')
assert diff1 > 1e-3, 'beta=1.0 should change output'

print('Softhint model smoke OK.')
"
```

Expected output:
```
mask-off vs mask-on (beta=0) max abs diff = 0.00e+00  (or extremely small)
mask-on (beta=1.0) vs mask-off diff = <some positive number>
Softhint model smoke OK.
```

If the diff for beta=0 is not essentially zero, STOP and investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/nn/tasks.py
git commit -m "yoloa_v2_softhint: rewire YOLOAnomalyV2Model around HeatmapBiasFusion

_predict_once now adds a per-scale, 1-channel, bounded bias (broadcast over PAN
channels) to each P3/P4/P5 feature before Detect. Detect itself is unchanged.

beta init 0 => training starts as vanilla YOLO. Mask dropout (p_drop) still
works by zeroing the bias for dropped samples. Both reg and cls see the bias;
the bias is bounded and additive (vs the previous multiplicative amplifier),
which is the soft-hint property we are validating.

SegBranch / shuffle-noise mask augment / external mask / explicit disable
paths are all preserved."
```

---

## Task 5: Add the softhint YAML configs

**Files:**
- Create: `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml`
- Create: `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml`

- [ ] **Step 1: Create the main softhint YAML**

Write `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml`:

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLO Anomaly v2 SOFT-HINT — yolo26 detection backbone + bounded additive bias on PAN.
# Replaces the multiplicative per-channel gate (Phase 0/2) with HeatmapBiasFusion:
# a 1-channel bounded bias added (broadcast) to PAN P3/P4/P5 before Detect.
# Branch: yoloa_v2_softhint. Spec: docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.

# Parameters
nc: 80
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# Anomaly v2 config (read by YOLOAnomalyV2Model.__init__).
anomaly_v2:
  mask_size: 80
  mask_mode: rect
  sigma_factor: 0.25
  p_drop: 0.5
  # Softhint defaults: no seg branch, no aug knobs (single-variable first run).
  seg_branch: false
  mask_shuffle_p: 0.0
  mask_noise_std: 0.0

# Backbone (identical to yolo26.yaml)
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# Head (identical to yolo26.yaml)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]] # 16

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]] # 19

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22

  - [[16, 19, 22], 1, Detect, [nc]]
```

- [ ] **Step 2: Create the seg-a1 variant**

Write `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml`:

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# YOLO Anomaly v2 SOFT-HINT + SegBranch (pinned alpha=1, GT mask only).
# Confirms that the softhint fusion is non-destructive in the only Phase 2
# configuration that was positive (a1). Branch: yoloa_v2_softhint.

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
  mask_size: 80
  mask_mode: rect
  sigma_factor: 0.25
  p_drop: 0.5
  seg_branch: true
  seg_alpha_mode: pinned_one
  seg_gain: 1.0
  seg_detach: true
  mask_shuffle_p: 0.0
  mask_noise_std: 0.0

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

  - [[16, 19, 22], 1, Detect, [nc]]
```

- [ ] **Step 3: Verify both YAMLs load**

```bash
python -c "
from ultralytics.nn.tasks import YOLOAnomalyV2Model
for cfg in [
    'ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml',
    'ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml',
]:
    m = YOLOAnomalyV2Model(cfg, ch=3, nc=1, verbose=False)
    n_params = sum(p.numel() for p in m.heatmap_bias_fusion.parameters())
    has_seg = m.seg_branch is not None
    print(f'{cfg.split(\"/\")[-1]:50s} bias_fusion_params={n_params} seg_branch={has_seg}')
"
```

Expected: both YAMLs load without exceptions; bias_fusion_params is small (a few dozen — 1->8 conv + 8->1 conv + 3 betas ≈ 89 params).

- [ ] **Step 4: Commit**

```bash
git add ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml \
        ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml
git commit -m "yoloa_v2_softhint: add softhint and softhint-seg-a1 model YAMLs"
```

---

## Task 6: Equivalence sanity script (runtime gate before training)

**Files:**
- Create: `scripts/softhint_sanity.py`

Stand-alone verification: load the actual YAML, assert β=0 equivalence on a tiny batch, assert β!=0 changes output, print bias parameter count and β values.

- [ ] **Step 1: Create the script**

Write `scripts/softhint_sanity.py`:

```python
"""Softhint fusion sanity: verify beta=0 forward equals vanilla, beta!=0 differs.

Run: python scripts/softhint_sanity.py
Exits non-zero on any failure; prints a one-line PASS otherwise.
"""

from __future__ import annotations

import sys
import torch

from ultralytics.nn.tasks import YOLOAnomalyV2Model


def to_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    cfg = "ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml"
    m = YOLOAnomalyV2Model(cfg, ch=3, nc=1, verbose=False)
    m.eval()

    n_bias_params = sum(p.numel() for p in m.heatmap_bias_fusion.parameters())
    print(f"HeatmapBiasFusion params: {n_bias_params}")
    print(f"beta init: {m.heatmap_bias_fusion.beta.detach().tolist()}")

    torch.manual_seed(0)
    x = torch.randn(2, 3, 640, 640)

    # 1) Mask-off forward.
    out_off = to_tensor(m(x))

    # 2) Mask-on with beta=0 -> must equal mask-off.
    bboxes = torch.tensor([[0.30, 0.40, 0.20, 0.15], [0.65, 0.55, 0.30, 0.20]])
    batch_idx = torch.tensor([0, 1], dtype=torch.long)
    m.set_mask_input(bboxes, batch_idx)
    out_on_beta0 = to_tensor(m(x))
    diff0 = (out_off - out_on_beta0).abs().max().item()
    print(f"beta=0 max-abs-diff vs mask-off: {diff0:.3e}")
    if diff0 > 1e-5:
        print("FAIL: beta=0 forward should equal vanilla mask-off forward.")
        return 1

    # 3) Mask-on with beta=1.0 -> output must change.
    with torch.no_grad():
        m.heatmap_bias_fusion.beta.fill_(1.0)
    m.set_mask_input(bboxes, batch_idx)
    out_on_beta1 = to_tensor(m(x))
    diff1 = (out_off - out_on_beta1).abs().max().item()
    print(f"beta=1 max-abs-diff vs mask-off: {diff1:.3e}")
    if diff1 < 1e-3:
        print("FAIL: beta=1 forward should differ measurably from vanilla.")
        return 1

    print("PASS softhint sanity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
cd /Users/louis/workspace/ultra_louis_work/ultralytics
python scripts/softhint_sanity.py
```

Expected last line: `PASS softhint sanity.`
If any FAIL appears, stop and fix the root cause before continuing — do NOT proceed to training.

- [ ] **Step 3: Commit**

```bash
git add scripts/softhint_sanity.py
git commit -m "yoloa_v2_softhint: add softhint_sanity.py (beta=0 equivalence gate)"
```

---

## Task 7: False-prompt eval script

**Files:**
- Create: `scripts/false_prompt_eval.py`

Primary eval signal (§6 of spec). Inputs: a trained checkpoint + a YOLO-format dataset YAML with a val split. Outputs: AUROC of (anomaly + GT mask) max-conf vs (good + random mask) max-conf, plus percentile dump and a histogram.

- [ ] **Step 1: Create the script**

Write `scripts/false_prompt_eval.py`:

```python
"""False-prompt evaluation for YOLOA v2 softhint.

For each anomalous image: predict with its GT-derived mask, record max conf (positive).
For each good image:      predict with a random rect mask,    record max conf (negative).
Report AUROC and percentiles. Optional matplotlib histogram.

Usage:
    python scripts/false_prompt_eval.py \
        --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt \
        --data    /path/to/data.yaml \
        --out     runs/temp/false_prompt_softhint.json \
        [--no-plot]

Notes:
- The data.yaml is expected to have a 'val' (or 'test') split with YOLO labels.
- Images whose label file is empty or missing are treated as 'good'; otherwise 'anomalous'.
- Random mask: uniform center in [0.15, 0.85]^2, square side in [0.10, 0.40] of image.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.anomaly_v2 import AnomalyV2Predictor  # noqa: F401  (ensures task registration)


def list_split_images(data_yaml: str, split: str) -> list[Path]:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", "."))
    rel = cfg.get(split)
    if rel is None:
        raise SystemExit(f"split {split!r} not in {data_yaml}")
    p = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if p.is_file():  # txt with image paths
        return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in exts])


def yolo_label_path(img: Path) -> Path:
    # Standard YOLO layout: .../images/<...>/x.jpg <-> .../labels/<...>/x.txt
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts).with_suffix(".txt")
    return lbl


def read_yolo_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # cls cx cy w h  (normalized)
        cx, cy, w, h = (float(x) for x in parts[1:5])
        rows.append((cx, cy, w, h))
    return rows


def random_rect_mask(h: int = 80, w: int = 80, rng: random.Random | None = None) -> torch.Tensor:
    rng = rng or random
    cx = rng.uniform(0.15, 0.85)
    cy = rng.uniform(0.15, 0.85)
    side = rng.uniform(0.10, 0.40)
    x1 = max(0, int((cx - side / 2) * w))
    x2 = min(w, int((cx + side / 2) * w))
    y1 = max(0, int((cy - side / 2) * h))
    y2 = min(h, int((cy + side / 2) * h))
    m = torch.zeros(1, 1, h, w)
    m[0, 0, y1:y2, x1:x2] = 1.0
    return m


def gt_rect_mask(boxes: list[tuple], h: int = 80, w: int = 80) -> torch.Tensor:
    m = torch.zeros(1, 1, h, w)
    for cx, cy, bw, bh in boxes:
        x1 = max(0, int((cx - bw / 2) * w))
        x2 = min(w, int((cx + bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        y2 = min(h, int((cy + bh / 2) * h))
        m[0, 0, y1:y2, x1:x2] = 1.0
    return m


def max_conf(result) -> float:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
        return 0.0
    return float(boxes.conf.max().item())


def auroc(pos: list[float], neg: list[float]) -> float:
    # Mann-Whitney U based AUROC; robust to ties.
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - counts / 2 + 0.5  # mean rank within tie group
    tie_ranks = avg[inv]
    pos_rank_sum = tie_ranks[labels == 1].sum()
    n_pos = labels.sum()
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    images = list_split_images(args.data, args.split)
    print(f"Loaded {len(images)} images from {args.data}:{args.split}")

    model = YOLO(args.weights, task="anomaly_v2")

    pos_conf: list[float] = []  # anomaly + GT mask
    neg_conf: list[float] = []  # good + random mask

    for img in images:
        boxes = read_yolo_label(yolo_label_path(img))
        is_good = len(boxes) == 0
        mask = random_rect_mask(rng=rng) if is_good else gt_rect_mask(boxes)
        # Warm-up so predictor exists, then inject the mask and predict.
        model.predict(str(img), verbose=False, save=False)
        model.predictor.model.set_external_mask_once(mask.to(next(model.predictor.model.parameters()).device))
        res = model.predict(str(img), verbose=False, save=False)[0]
        c = max_conf(res)
        (neg_conf if is_good else pos_conf).append(c)

    out = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "n_pos": len(pos_conf),
        "n_neg": len(neg_conf),
        "auroc": round(auroc(pos_conf, neg_conf), 4),
        "pos_p50": round(float(np.percentile(pos_conf, 50)), 4) if pos_conf else None,
        "pos_p05": round(float(np.percentile(pos_conf, 5)), 4) if pos_conf else None,
        "neg_p50": round(float(np.percentile(neg_conf, 50)), 4) if neg_conf else None,
        "neg_p95": round(float(np.percentile(neg_conf, 95)), 4) if neg_conf else None,
        "neg_p99": round(float(np.percentile(neg_conf, 99)), 4) if neg_conf else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(neg_conf, bins=30, alpha=0.6, label=f"good + random mask (n={len(neg_conf)})")
            ax.hist(pos_conf, bins=30, alpha=0.6, label=f"anomaly + GT mask (n={len(pos_conf)})")
            ax.set_xlabel("max detection confidence")
            ax.set_ylabel("image count")
            ax.set_title(f"False-prompt eval | AUROC = {out['auroc']:.4f}")
            ax.legend()
            fig.tight_layout()
            fig_path = out_path.with_suffix(".png")
            fig.savefig(fig_path, dpi=120)
            print(f"Saved histogram to {fig_path}")
        except Exception as e:
            print(f"Plot skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/false_prompt_eval.py
git commit -m "yoloa_v2_softhint: add scripts/false_prompt_eval.py (max-conf AUROC)"
```

(The script is run on real checkpoints later — Task 8 runbook covers that.)

---

## Task 8: Runbook for the two training runs

**Files:**
- Create: `docs_yoloa_v2/phase0_softhint_commands.md`

- [ ] **Step 1: Write the runbook**

Write `docs_yoloa_v2/phase0_softhint_commands.md`:

````markdown
# Softhint Fusion — ultra6 Training Commands

Run on ultra6 (`ssh ultra6`), repo at `~/ultra_louis_work/ultralytics/`.
Preconditions:
1. `conda activate ultra`
2. `set_wandb_true`
3. Branch `yoloa_v2_softhint` is checked out and clean
4. The Phase 0 mask-augment runs from `yoloa_v2` may still be running on other GPUs — coordinate device IDs to avoid conflict (`gpuu6` to check).

Each run:
- **20 epochs** (fast iteration), batch 96, 3 GPUs DDP
- Identical hparams to baseline `26m_yoloav2_v5_binary_cm20_rect_pd50_v1` except fusion mechanism and epochs.

Baseline reference (already trained on `yoloa_v2`):
```
26m_yoloav2_v5_binary_cm20_rect_pd50_v1     mask-on 0.6941 / off 0.6229 (e50)
```

---

## 1. Softhint main — `softhint_rect_pd50_v1`

```
nohuprun python -m ultralytics.cfg \
  train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-softhint.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=0,1,2 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_softhint_rect_pd50_v1
```

If `python -m ultralytics.cfg` is not your preferred entry point on ultra6, use whatever wrapper your shell aliases (`nohupyolo`, etc.) — preserve exactly these arg=values.

## 2. Softhint + SegBranch a1 — `softhint_rect_pd50_seg_a1_v1`

Use a different device set if 0–2 are taken.

```
nohuprun python -m ultralytics.cfg \
  train task=anomaly_v2 \
  model=yolo26m-anomaly-v2-softhint-seg-a1.yaml \
  pretrained=yolo26m.pt \
  data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=3,4,5 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2 name=26m_yoloav2_softhint_rect_pd50_seg_a1_v1
```

## 3. Monitor

```
tail -f runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1.log
lsta
lsddp
```

Inspect `beta` after epoch 20:

```
python -c "
import torch
ckpt = torch.load('runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt', map_location='cpu', weights_only=False)
m = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
print('beta:', m.heatmap_bias_fusion.beta.detach().tolist())
"
```

## 4. Evaluation

After both runs finish:

```
# False-prompt benchmark on each softhint run
python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_softhint.json

python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_softhint_rect_pd50_seg_a1_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_softhint_seg_a1.json

# Baseline for comparison: checkout yoloa_v2 in a worktree so the script picks up
# that branch's HeatmapEncoder/HeatmapGuidedFusion code, then re-run.
git -C ~/ultra_louis_work/ultralytics worktree add /tmp/yoloa_v2_wt yoloa_v2
cd /tmp/yoloa_v2_wt && python scripts/false_prompt_eval.py \
    --weights runs/yoloa_v2/26m_yoloav2_v5_binary_cm20_rect_pd50_v1/weights/best.pt \
    --data    /home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v5_binary/data.yaml \
    --out     runs/temp/false_prompt_baseline.json
```

(The baseline branch doesn't have `scripts/false_prompt_eval.py`. Copy it across with `cp ~/ultra_louis_work/ultralytics/scripts/false_prompt_eval.py /tmp/yoloa_v2_wt/scripts/` before running, or run the eval from a worktree of `yoloa_v2_softhint` against the baseline's `best.pt` — the eval script only depends on the `anomaly_v2` predictor's `set_external_mask_once`, which exists on both branches.)

## 5. Pass criteria

- Softhint AUROC > baseline AUROC by ≥ 0.05
- Softhint mAP50-95 (mask-on, e20) within 0.02 of baseline (mAP50-95 at baseline e20 — read from `runs/yoloa_v2/26m_yoloav2_v5_binary_cm20_rect_pd50_v1/results.csv` row 20)
- `beta` final values are finite (printed at step 3)
````

- [ ] **Step 2: Commit**

```bash
git add docs_yoloa_v2/phase0_softhint_commands.md
git commit -m "yoloa_v2_softhint: add softhint phase0 runbook (20-epoch training, false-prompt eval)"
```

---

## Task 9: Push branch, update project memory

**Files:** none (git + memory)

- [ ] **Step 1: Push branch**

```bash
git push
```

- [ ] **Step 2: Update project memory file** `/Users/louis/.claude/projects/-Users-louis-workspace-ultra-louis-work-ultralytics/memory/project_yoloa_v2.md`

Append (or update if relevant section exists) a paragraph noting:
- Branch `yoloa_v2_softhint` exists off `yoloa_v2`
- Mechanism: `HeatmapBiasFusion` (1ch bounded additive bias broadcast onto PAN P3/P4/P5 features; Detect head unchanged)
- β = 3 per-scale learnable scalars, init 0 → vanilla on launch
- Pending: launch `softhint_rect_pd50_v1` (20 epochs) on ultra6, eval via `scripts/false_prompt_eval.py`
- Convert "Active 2026-06-02" line in `MEMORY.md` description if needed

Do NOT include the spec or plan content — just the high-level state pointer.

- [ ] **Step 3: Done**

Branch is ready. Next manual step (out of this plan's scope): launch the two training runs on ultra6 per the runbook.

---

## Self-review checklist (run before handing off)

Plan was self-reviewed against the spec:

- **Spec §1 (problem)** — addressed by Task 4 (rewire) + Task 5 (YAML).
- **Spec §2 (goal: bounded + low-bandwidth; reg-can-be-affected explicit)** — Task 2 module enforces bounded; Task 4 broadcasts 1-ch bias to C channels (low bandwidth); reg path explicitly acknowledged as affected in the docstring + commit message.
- **Spec §3.1 (HeatmapBiasFusion: 1→8→1 conv, 3 beta, init 0)** — Task 2.
- **Spec §3.2 (injection: P + bias on PAN, Detect unchanged)** — Task 4 step 3.
- **Spec §3.3 (mask resolution unchanged)** — Task 4 keeps `_resolve_fusion_mask` and `_augment_mask` intact.
- **Spec §3.4 (delete HeatmapEncoder / HeatmapGuidedFusion only on softhint branch)** — Task 2 + Task 3.
- **Spec §4 (training: 20 epochs, hparams match rect_pd50_v1)** — Task 8 runbook.
- **Spec §4 sanity gate** — Task 6.
- **Spec §5 (two comparison runs)** — Task 5 YAMLs + Task 8 runbook.
- **Spec §6 (false-prompt benchmark)** — Task 7.
- **Spec §7 (out of scope)** — respected; no Plan B (backbone-mid) / Plan C (cross-attn) / SegBranch changes / augment stacking touched.
- **Spec §8 file touch list** — all eight items covered by Tasks 2–8. `head.py` is explicitly NOT in the list.

No placeholders. Types consistent: `bias` is a `Tensor` of shape `(B, 1, H, W)` everywhere; `HeatmapBiasFusion.forward(mask, scale_idx)` signature consistent across all references.
