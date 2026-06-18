# Soft-Hint Fusion Redesign (YOLOA v2)

**Date:** 2026-06-02
**Branch (impl):** `yoloa_v2_softhint` (off `yoloa_v2`)
**Status:** Spec — awaiting implementation plan

---

## 1. Problem

Current `HeatmapGuidedFusion` (PAN-late, per-channel multiplicative gate `P * 2·sigmoid(AF)`) is too aggressive: when an external/rendered heatmap is provided, the model produces a detection there **regardless of the underlying visual evidence**. Observed in demo usage with a user-drawn box on a wrong location: a false detection still appears.

Two structural causes:

1. **Amplifier semantics.** Multiplier upper bound is 2.0 applied to the PAN feature itself; any feature `× 2` looks positive to the classifier.
2. **High bandwidth.** The 1-channel mask is projected to a per-channel (C = 256/512/512) AF tensor that rewrites the feature in *all* channels. The cls head learns to treat the gate signal itself as evidence (shortcut).

Both also propagate into the regression and objectness branches, giving the heatmap power to alter box position/shape — beyond what a "soft hint" should do.

## 2. Goal

Redesign fusion so the heatmap behaves as a **soft hint**:

- Heatmap influences ranking / confidence, not feature identity.
- Heatmap on a clean image with no real anomaly should produce **low-confidence (sub-threshold) detections at most**.
- Prior-free path (SegBranch, future memory-bank) remains compatible — the fusion must support a graceful passthrough when no mask is provided.
- Architecture stays CNN.

**Not required:** isolating the heatmap's effect to the classification branch only. An earlier draft of this spec made that an explicit goal ("cannot move or resize boxes"), but it was the *implementation* of the soft-hint property, not the property itself. The core hypothesis is *bounded + low-bandwidth* fusion; whether the bias reaches the regression branch is a second-order question we treat empirically — start by allowing it (simpler), and only isolate cls if the regression branch turns out to be polluted.

## 3. Design

### 3.1 Mechanism — `HeatmapBiasFusion`

Replace the per-channel multiplicative gate with a low-bandwidth, bounded, additive bias applied to the PAN features at each scale, just before they enter the Detect head.

```python
class HeatmapBiasFusion(nn.Module):
    """1-ch mask -> bounded per-pixel bias added to PAN features before Detect."""
    def __init__(self, num_scales: int = 3):
        super().__init__()
        # Shared 1->8->1 conv across scales (mask is resized per scale before forward).
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        # Per-scale learnable magnitude. Init 0 -> training starts as passthrough.
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        # mask: (B, 1, H, W) — caller resizes to the PAN scale before calling.
        raw = self.conv(mask)              # (B, 1, H, W) unbounded
        bounded = torch.tanh(raw)          # per-pixel in [-1, +1]
        return self.beta[scale_idx] * bounded  # (B, 1, H, W) in [-beta_i, +beta_i]
```

Properties:

- **Bandwidth = 1.** Output is a single channel, broadcast to all C channels of the PAN feature when added. Adds at most one degree of per-pixel additive freedom per scale — far below the current per-channel C-dim modulation.
- **Bounded.** `tanh` clamps per-pixel magnitude to ±1; `beta` scales it. `beta` is a learnable scalar per PAN scale (3 scalars total) with no hard cap — if `beta` blows up during training the lift on real anomalies blows up too, which is easy to monitor.
- **Init = passthrough.** `beta = 0` at init → training starts as vanilla YOLO. The model learns to use the heatmap only if it helps the detection loss.
- **Additive, not multiplicative.** Unlike `P * 2·sigmoid(AF)` (an unbounded amplifier on the feature itself), `P + bias` shifts the activation by a bounded amount. The detection head still has to decide that the *content* of `P` looks like an anomaly — bias alone cannot generate strong features out of nothing.

### 3.2 Network injection

Injection point: PAN-late, immediately before the Detect head. PAN features at each scale are replaced by `P_fused = P + bias`, where `bias` is broadcast over C channels. The Detect head itself is **unchanged** — vanilla `Detect`, with its existing `cv2` (reg) and `cv3` (cls) branches, runs on `P_fused`.

| Stage | Current | New |
| --- | --- | --- |
| PAN P3/P4/P5 outputs | rewritten by `P * 2·sigmoid(AF)` | replaced by `P + bias` (1-ch bias, broadcast to C) |
| Detect `cv2` (reg) | sees fused feature | sees `P_fused` (same path) |
| Detect `cv3` (cls) | sees fused feature | sees `P_fused` (same path) |
| `head.py` changes | n/a | **none** |

The bias reaches both regression and classification, which is the intentional simplification over an earlier draft. We treat "is reg polluted?" as an empirical question gated on the first run — see §2.

### 3.3 Mask resolution

Unchanged from current `_resolve_fusion_mask`:

- Training: rendered from GT bboxes (`BboxMaskRenderer`, rect mode), per-sample `mask_dropout` with `p_drop = 0.5`.
- Inference: external mask (e.g. user prompt) or `SegBranch` prediction, both fed into the same code path.
- Mask augmentation knobs (`mask_shuffle_p`, `mask_noise_std`) are preserved but default to 0.0 in the first softhint runs (single-variable).

### 3.4 Removed components

On `yoloa_v2_softhint` only:

- `HeatmapEncoder` — deleted from `anomaly_v2.py`.
- `HeatmapGuidedFusion` — deleted.
- `YOLOAnomalyV2Model.heatmap_encoders` / `heatmap_fusions` `ModuleList`s — replaced by a single `self.heatmap_bias_fusion = HeatmapBiasFusion(num_scales=3)`.

`yoloa_v2` (parent branch) keeps the old code untouched so cross-branch comparison remains apples-to-apples against `rect_pd50_v1`.

`SegBranch`, `BboxMaskRenderer`, `binary_seg_loss`, `p_drop`, `mask_shuffle/noise` augment are **not** touched.

## 4. Training

Single-variable change from `rect_pd50_v1` — same dataset, optimizer, augmentations, seed; only fusion swapped and epochs shortened for fast iteration.

| Setting | Value |
| --- | --- |
| backbone init | `yolo26m.pt` |
| task | `anomaly_v2` |
| dataset | `merge_data_v5_binary` |
| epochs | **20** (was 50; faster iteration per user) |
| batch / device | 96 / 3 GPUs |
| optimizer | MuSGD, lr0=0.00125, lrf=0.5, momentum=0.9, wd=0.0005 |
| close_mosaic | 20 |
| mask_mode | rect |
| p_drop | 0.5 |
| mask_shuffle_p / mask_noise_std | 0.0 / 0.0 |
| save_json | True |

**Sanity gate** (before launching the long run): a forward-pass equivalence test — with `beta = 0` and mask provided, output must equal the same model run with `mask = None`. Lives as a unit test next to the new module.

## 5. First-batch comparison runs

| Run name | Branch | Notes |
| --- | --- | --- |
| `26m_yoloav2_softhint_rect_pd50_v1` | `yoloa_v2_softhint` | Main softhint result. Direct apples-to-apples vs `26m_yoloav2_v5_binary_cm20_rect_pd50_v1` (epochs differ — see below). |
| `26m_yoloav2_softhint_rect_pd50_seg_a1_v1` | `yoloa_v2_softhint` | Pinned α=1 with SegBranch — confirms softhint fusion doesn't break the architecture used in Phase 2 a1 (which was the one positive Phase 2 result). |

**Epoch caveat.** Softhint runs are 20 epochs; the baseline `rect_pd50_v1` was 50 epochs. To compare fairly, either (a) read baseline metrics at its e20 checkpoint, or (b) rerun a `26m_yoloav2_rect_pd50_e20_v1` baseline on `yoloa_v2` at 20 epochs. The plan should pick one — recommendation: (a), using `results.csv` row at epoch 20.

## 6. False-prompt benchmark

New eval script `scripts/false_prompt_eval.py`:

1. Inputs: a trained checkpoint + a dataset split.
2. Build two cohorts:
   - **Positive:** anomalous images + their GT-derived mask.
   - **Negative (false-prompt):** *good* images + a synthetic random rect mask (uniform position, area 5%–30% of image).
3. For each image, run prediction and record the **max detection confidence**.
4. Output metrics:
   - AUROC over the two max-conf distributions (higher = better separation).
   - False-prompt max-conf percentiles (P50, P95, P99).
   - Side-by-side histogram plot.
5. Same script runs on both branches' checkpoints for direct comparison.

**Pass condition:** softhint AUROC > yoloa_v2 baseline AUROC by a clear margin (target ≥ 0.05), with mAP50-95 drop ≤ 0.02 vs baseline e20.

## 7. Out of scope

- Backbone-mid heatmap injection (Plan B) — revisit only if softhint mAP regresses meaningfully.
- CNN cross-attention fusion (Plan C) — shelved; doesn't address the amplifier root cause.
- SegBranch architecture changes — separate problem.
- Stacking mask_shuffle/noise augments with softhint — gated on first softhint run finishing.
- Memory-bank as prior source (v2.1) — still deferred.

## 8. File touch list (for plan)

| File | Change |
| --- | --- |
| `ultralytics/nn/modules/anomaly_v2.py` | Delete `HeatmapEncoder`, `HeatmapGuidedFusion`; add `HeatmapBiasFusion`. |
| `ultralytics/nn/modules/__init__.py` | Update imports/exports (remove old names, add `HeatmapBiasFusion`). |
| `ultralytics/nn/tasks.py` `YOLOAnomalyV2Model` | Remove old `ModuleList`s; add `self.heatmap_bias_fusion`; in `_predict_once`, replace per-scale `P * 2·sigmoid(AF)` with `P + bias`. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml` | New YAML; same architecture, `anomaly_v2` config block unchanged. |
| `ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint-seg-a1.yaml` | New YAML; same as above + `seg_branch: true`, `seg_alpha_mode: pinned_one`. |
| `scripts/false_prompt_eval.py` | New — described in §6. |
| `scripts/softhint_sanity.py` | Sanity equivalence test (§4 sanity gate). |
| `docs_yoloa_v2/phase0_softhint_commands.md` | New — train launch + eval commands for both runs. |

**Note:** `ultralytics/nn/modules/head.py` is **not** touched. The Detect head is reused as-is.

## 9. Decision summary

- Soft hint (option A) over strong prompt (B) and adaptive (C). User picked A.
- On clean image + false prompt: option B — bounded influence is acceptable; A and C deferred.
- SegBranch path: option B — kept as optional, design must remain compatible.
- β: per-scale learnable scalars (3), no hard cap.
- False-prompt benchmark: adopted as a primary eval signal alongside mAP.
- Comparison: cross-branch (`yoloa_v2` vs `yoloa_v2_softhint`), no in-branch toggle.
- **Bias injection:** PAN feature-level (`P + bias`) rather than Detect cls-only. Trades the "reg untouched" guarantee for a much smaller diff (Detect unchanged) and tests the *core* hypothesis (bounded + low-bandwidth) without the cls-isolation confound. If the first run shows reg pollution, upgrade to cls-only as a follow-up — at that point we will have a concrete reason.
