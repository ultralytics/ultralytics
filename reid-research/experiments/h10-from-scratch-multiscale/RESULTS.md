# h10 — From-scratch multi-scale (the fifth null)

After h9 showed three fine-tunes from champion's converged best.pt all degraded
R1 (lr-disturbs-converged-state), this experiment trained a fresh model from
MSMT-pretrained backbone with per-batch random imgsz ∈ {384, 416, 448}, 100
epochs, 4-GPU DDP, lr0=3.5e-3 — replicating TX2's recipe but with multi-scale
augmentation enabled.

## Result: also null, also worse than champion

| | R1 raw | R1 TTA+rerank | mAP raw |
|---|---|---|---|
| champion @384 | 0.8996 | 0.9267 | 0.7404 |
| TX2 @448 | 0.9044 | 0.9311 | 0.7467 |
| champion+TX2 concat (2-model) | 0.9243 | — | 0.7761 |
| **NEW multi-scale @384** | **0.8599** | 0.8973 | 0.6851 |
| **NEW multi-scale @448** | **0.8548** | 0.8884 | 0.6805 |

Wall-clock: 10.5 min on 4-GPU DDP, 100 epochs.

## Training trajectory (R1 @384 per saved epoch)

| Epoch | R1 | Δ |
|---|---|---|
| 10 | 0.640 | — |
| 20 | 0.735 | +0.095 |
| 30 | 0.788 | +0.053 |
| 40 | 0.828 | +0.040 |
| 50 | 0.840 | +0.012 |
| 60 | 0.854 | +0.014 |
| **70** | **0.860 (peak)** | +0.006 |
| 80 | 0.857 | −0.003 |
| 90 | 0.860 | +0.003 |
| 100 | 0.860 | 0 |

**R1 plateaus at 0.860 from epoch 70 onward — never approaches champion's
0.8996 at single-scale.** The model has converged to a worse equilibrium.

## Diagnosis

The +2.47pp ensemble gain (champion+TX2 concat → R1=0.9243) requires **two
separate parameterizations with two separate optimization trajectories**.
When a single model is trained on the union of scales, it learns the average
of both — landing at the noisier midpoint (R1=0.86) instead of capturing
either scale's strengths.

The concat ensemble works because each model independently specializes its
weights to its training scale; concatenation preserves both specializations
intact. A single model trained with random imgsz per batch cannot maintain
two parallel internal representations within one weight matrix — the SGD
gradient averages across scales, producing a single "average-scale"
representation that's strictly worse than either dedicated single-scale
model.

**This is the architectural limit.** Multi-scale information at champion's
parameter count is forced through a single weight matrix that can encode
ONE scale-specific geometry well. Two scales require two networks.

## Implication for future work

The path to R1 > 0.927 single-model is now confirmed to NOT pass through:
- Cheap head-only post-processing (h9-B: rerank MLP, null)
- Loss-recipe fine-tunes from champion (h9-A: tighter margin, −0.034)
- Multi-scale fine-tunes from champion (h9-C, −0.025)
- Multi-scale from-scratch training (h10, −0.04)

What remains untested:
- **LUPerson-NL pretrain** — storage-blocked. The strongest single known
  lever in the broader literature; would replace MSMT pretrain with a far
  larger unsupervised person dataset.
- **Same-architecture distillation from champion+TX2 ensemble** — use the
  concat ensemble's R1=0.9243 embeddings as a teacher target for a fresh
  yolo26l-2psa student. Same-arch teacher avoids the cross-arch failure
  mode that sank the T5 distillation series.
- **Channel-expanded backbone** — double the channel count to give the
  model the parameter budget to encode two scale-specific subspaces in
  one network. Architecturally non-trivial.

After 5 consecutive nulls (h9-A/B/C, h10, plus the previous research
program's 9 baselines), the working assumption is that **the champion's
single-model R1=0.927 is the ceiling for this architecture at this
parameter count on Market-1501 with no new data.** Pushing past it requires
either more data (LUPerson) or more parameters (channel expansion).
