# h9 — Two Calibration Experiments (Both Null)

Following h8's finding that champion fails on 261 *recoverable* near-miss queries
(97% with cosine margin in [-0.10, 0]), this folder ran two cheap calibration
interventions on seetacloud GPU 3. Both failed.

## Experiment B — Top-50 re-ranking MLP head (`exp_b_rerank.py`)

**Hypothesis:** Train a small MLP head `score(q_emb, g_emb) → logit` on Market train,
re-rank champion's top-50 retrieval on test. Since true PID is in top-50 in 94% of
failures, this should recover most of the gap.

**Result:**

| | R1 | mAP |
|---|---|---|
| baseline (champion raw) | 0.8996 | 0.7404 |
| + rerank MLP | 0.8999 | 0.7340 |
| **delta** | **+0.0003** | **−0.0064** |

Wall-clock: 5.7 min on 1 GPU.

**Diagnosis:** Train accuracy hit 1.000 by epoch 15 (perfect memorization), but
test improvement was zero. Market train and test PIDs do not overlap. The
`[q-g, q*g, cos]` input features encode identity-specific patterns at training
time that don't transfer to unseen test identities. A re-ranker on top of
champion's frozen embeddings cannot add information that champion's
embeddings don't already encode.

## Experiment A — Hard-margin triplet fine-tune (`exp_a_hardmargin.py`)

**Hypothesis:** Resume training from `champion/best.pt` with `triplet_margin=0.05`
(down from champion's 0.5). Smaller margin focuses loss pressure on the
borderline cosine band where 97% of recoverable failures live.

**Result:**

| | R1 (no TTA) | mAP (no TTA) | R1 (TTA+rerank) |
|---|---|---|---|
| baseline (champion) | 0.8996 | 0.7546 | (0.9267 published) |
| new (margin=0.05, +30 ep) | 0.8657 | 0.6770 | 0.9023 |
| **delta R1 (no-TTA)** | **−0.034** | **−0.078** | |

Wall-clock: 9.6 min on 1 GPU.

**Diagnosis (mass-shifting catastrophe):** Triplet loss `max(0, d_pos − d_neg + margin)`
fires only when positive is barely closer than negative (within `margin`). With
`margin=0.05`, easy triplets (d_pos ≪ d_neg) contribute exactly 0 gradient — no
signal at all. The model is forced to overcorrect on the tiny near-miss subset
each batch while forgetting the broader well-calibrated manifold. Going from
0.5 → 0.05 mid-training is too abrupt; the loss surface destabilizes.

## What this tells us

The h8 analysis was correct about the geometry (97% of failures are near-misses
in a tight band), but **neither cheap calibration lever works**:
- A learned head on top of frozen embeddings cannot cross the train/test PID
  gap with the architecture I used.
- A retrained backbone with a tighter margin loses the bulk-manifold structure
  without gaining selective hard-pair discrimination.

The champion's R1≈0.927 ceiling is real and tightly coupled to its embedding
geometry. Moving past it likely requires:
- A *curriculum*: schedule `triplet_margin` from 0.5 → 0.1 over training (not
  swap mid-stream), or weight easy-vs-hard pairs adaptively.
- A *different loss family*: SupCon or InfoNCE with adaptive hard negatives.
- A *data-side lever*: LUPerson-NL pretrain (still blocked by disk).
- A *teacher signal* with proper architectural alignment (a CNN teacher, not
  the Swin SOLIDER — cross-arch distill is documented to fail here).

## Reproducibility

- Champion ckpt: `/root/autodl-tmp/ultralytics_reid/runs/reid/runs/reid/runs/arch31_imgsz384/weights/best.pt`
- Market: `/root/.cache/autoresearch/Market-1501-v15.09.15`
- Scripts in this dir; outputs persisted to `/root/expa/` and `/root/expb/` on
  the seetacloud box (`connect.westb.seetacloud.com:25792`).
- Both used `CUDA_VISIBLE_DEVICES=3` to coexist with the ty1/ty2/ty3 training
  runs occupying GPUs 0–2.

## Result jsons (verbatim)

### exp_b
```
{"baseline":{"r1":0.8996,"mAP":0.7404},"rerank":{"r1":0.8999,"mAP":0.7340},
 "delta_r1":+0.0003,"delta_mAP":-0.0064,"wall_clock_min":5.7}
```

### exp_a
```
{"baseline_no_tta":{"r1":0.8996,"mAP":0.7546},
 "new_no_tta":{"r1":0.8657,"mAP":0.6770},
 "new_tta_rerank":{"r1":0.9023,"mAP":0.8474},
 "delta_r1_no_tta":-0.034,"wall_clock_min":9.6}
```

---

## Experiment C — multi-scale fine-tune (added 2026-05-30)

After h8 follow-up surfaced a +2.47pp R1 from champion+TX2 ensemble (different
imgsz models concat'd), this experiment trained a single model on random imgsz
∈ {384, 416, 448} per batch for 40 epochs from champion's best.pt, lr0=5e-4.

**Result: also null — worse at every eval scale.**

| | R1 no-TTA | R1 TTA+rerank |
|---|---|---|
| champion @384 (baseline) | 0.8996 | (0.9267 published) |
| TX2 @448 (reference, separately trained) | 0.9044 | (0.9311 published) |
| new multi-scale @384 | 0.8741 (−0.0255) | 0.9136 |
| new multi-scale @448 | 0.8700 (−0.0140) | 0.9050 |

12.5 min training, GPU 3 on seetacloud.

**Diagnosis (lr-disrupts-converged-state pattern):** Combined with Experiment A's
margin=0.05 fine-tune (delta −0.034) and B's rerank head (delta +0.0003), three
fine-tunes from champion's converged best.pt all degraded R1. The common cause:
lr0=5e-4 is 143× higher than champion's final LR of 3.5e-6. Restarting Adam
optimizer state and bumping LR to "training" levels disturbs an already-
calibrated embedding manifold. The model has to climb back through the loss
landscape to find a different (worse) local minimum.

**Implication:** any future fine-tune from champion needs either (a) much lower
lr (try 5e-5), (b) very short schedule (5 epochs), or (c) a fundamentally
different state to start from (e.g., from-scratch training with the new objective).
