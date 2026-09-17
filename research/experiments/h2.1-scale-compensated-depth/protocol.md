# Protocol — H2.1, scale-compensated depth labels

_Locked 2026-09-17, before the run. One arm, one change from `B_warmstart`._

## The hypothesis

H2 established that both of its readings are true at once: scale augmentation **corrupts the metric depth
target** (every relative-depth measure improved when it was removed, on both datasets) and scale augmentation
**is what buys cross-domain scale robustness** (absolute root depth got 39% worse on 3DPW when it was removed).
Deleting the augmentation trades one for the other. H2.1 proposes taking both.

**H2.1: compensating the depth label for the resize recovers C's relative-depth gain and B's out-of-domain
root accuracy at the same time, so it beats both on 3DPW.**

## The mechanism, stated precisely

Scaling an image by `s` about its centre maps a point at `(X, Y, Z)` onto exactly the pixels `(X, Y, Z/s)`
would produce under the same focal length. So the depth consistent with the resized image is `Z/s` — for the
root, and for the root-relative channels too, since joint differences scale with it. `rescale_encoded_z`
applies that to the encoded channels inside `RandomPerspective`, using `s = sqrt(|det(M[:2,:2])|)`, which
recovers the uniform scale from any similarity transform.

Verified before launch, exactly and in both directions: a person labelled 8.0 m comes back at 4.0 m when the
image is scaled 2x and 16.0 m when scaled 0.5x, relative depths scaling with them, and a no-op at s = 1.

**The letterbox deliberately gets no compensation.** The assumed focal is defined as a ratio of the letterboxed
image size, so it scales with the letterbox and the two cancel. That is the same invariance that made
`fx/max(w, h)` constant across 3DPW's portrait and landscape sequences, and it is why the per-image letterbox
ratio recorded as a confound in the R0/H2 protocol turns out not to be one.

## Arm

`D_scalecomp` — identical to `B_warmstart` in every respect (warm start from `yolo26n-pose`, `mosaic=1.0`,
`scale=0.5`, 100 epochs, batch 64, 640 px, seed 0, same 56,599-image dataset) plus the compensation.

`B_warmstart` is the control and is **not** re-run: `git diff a6c3f54fd..HEAD -- ultralytics/` touches only
`augment.py` (the treatment, 6 lines), plus `val.py`, `pose3d.py` and a dataset path, all evaluation-side. The
loss, the model, the dataset builder and the trainer are byte-identical to the code B trained under.

## Predictions

- **3DPW AbsRel(Z) beats B's 0.1266** — this is the claim. If it lands between C's 0.1754 and B's 0.1266,
  the compensation helps but something else is also broken.
- **3DPW delta1(Z) at least matches C's 0.7487**, since the relative-depth supervision is now correct rather
  than merely absent.
- **2D mAP stays at B's level** (0.774 pose), because the augmentation itself is unchanged — if mAP drops, the
  compensation is corrupting something beyond depth.
- **MPJPE below B's 113.2 mm.**

A result where AbsRel improves but delta1 does not would say the root and relative channels need different
treatment, which would be the next split.
