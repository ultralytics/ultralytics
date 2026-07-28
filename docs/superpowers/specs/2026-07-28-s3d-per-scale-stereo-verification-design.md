# Design: verify that the s3d stereo cue is wired to only one FPN scale

**Goal:** prove or refute, before changing any model code, that the `StereoCostVolume` output reaches the
depth branches at **P3 only**, leaving the P4 and P5 depth branches with no right-image tensor in their
inference computation graph at all — and quantify how many objects that strands on a monocular path.

**Why now:** every stereo lever of the July campaign (finer bins, sub-pixel soft-argmin, cascade,
unimodal, photometric) moved the pooled matching share within a 8–42% band and then plateaued. If ~60%
of objects are decoded from branches that structurally cannot see the right image, that band is a
ceiling imposed by wiring, not a tuning limit, and every future lever inherits it.

## The hypothesis

`Stereo3DDetHead.forward_head` (`ultralytics/models/yolo/s3d/head.py:174`) concatenates the cost volume
into the `lr_distance` / `depth` branches only at scale index 0:

```python
if cost_vol is not None and i == 0 and name in ("lr_distance", "depth"):
    feat = torch.cat([feat, cost_vol], dim=1)
```

`__init__:130` mirrors it (`in_ch = x + self.cv_ch if i == 0 else x`). If this reading is right, the P4
and P5 depth outputs are monocular **by construction**, not by learned preference.

### Supporting evidence already in hand (this session, local CPU probes)

| Observation                                                     | Measurement                                                                                             |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| cube_s3d objects assigned to P3                                 | **0 of 90** (100% land on P4, box long side 81–181 px) — explains cube's exact 0.00 sensitivity         |
| KITTI val objects by size band (estimate)                       | P3 39.4% (median z 40.6 m), P4 52.1% (18.2 m), P5 8.4% (7.3 m)                                          |
| Correlation volume peak quality (cube, trained tap features)    | peak z-score 1.44; argmax within ±1 feature px for 16% of objects (≈ chance); raw pooled RGB scores 13% |
| Cost-volume branch weights                                      | alive, ~50% of the P3 first-conv energy — so this is _not_ a dead-weight/pruning story                  |
| Stereo precision available per scale (cv48 bin = 4.09 input px) | P3 2.3 bins / 43% depth error per bin; P4 5.2 bins / 19%; P5 13.1 bins / 8%                             |

The last row is the sharpest form of the problem: the cost volume is wired exclusively to the scale
where disparity is smallest and stereo is worth least, and is absent from the near-range scales where it
would be most precise.

### Consistency with prior campaign results

This reconciles two results previously recorded as contradictory
(`research/research-log.md`, `s3d-kitti-zoom-experiment` memory):

- KITTI's right-image ablation halved AP → the ~39% of objects on P3 genuinely use stereo.
- cube's ablation was a no-op → 0% of its objects are on P3.

It also predicts the campaign's plateau: the pooled matching share cannot exceed the fraction of
detections whose anchors sit on P3, because `_shift_probe` averages over decoded detections and P4/P5
detections contribute exactly zero. v4b's 42% (lr supervision fully off) sits right at that bound.

### Never previously tested

"Multi-scale cost volume" appears twice as a deferred follow-up and was never implemented. Phase 1
(commit `c63f929f`, later reverted) went the other way and entrenched the limitation — its own message
reads "P4/P5 anchors fall back to per-anchor lr_distance (no regression at coarse scales)". The revert
removed the soft-argmax primary-depth change, not the P3-only wiring, which has been the substrate for
every experiment since.

## Decision rule (pre-registered)

**The structural proof is the gate.** Reachability of the right image from each scale's depth output is
a yes/no property of the graph. If P4/P5 receive no gradient from the right-image channels, that is a
defect and the wiring gets fixed, independent of how large the present-day payoff looks.

- **Confirmed** → non-zero right-image gradient at P3, exactly zero at P4 and P5. Proceed to the code change.
- **Refuted** → any material right-image gradient at P4/P5 in eval mode. The diagnosis is wrong; stop and re-open it.

Response magnitudes (Component B) are **context, not a gate**. They size the prize and set the scope of
the subsequent fix: if P3's own per-scale response is also ≈0, the matching descriptor is a co-requisite
lever and the follow-up change must address it too.

## Architecture: one gate test, one probe extension

Two components, deliberately split by cost and by lifetime — a permanent guard that needs no data, and a
measurement that needs a checkpoint.

### Component A — the gate: per-scale stereo reachability test

New test in `tests/` on the feature branch (not the internal diagnostics branch — this guards shipped
behaviour, so it belongs with the code it protects).

```python
model = Stereo3DDetModel("yolo26n-s3d.yaml").eval()
x = torch.rand(1, 6, 384, 1248, requires_grad=True)
preds = model(x)[1]
for scale, idx in per_scale_anchor_indices(model):      # one anchor index in each of P3/P4/P5
    for key in ("depth", "lr_distance"):
        model.zero_grad()
        x.grad = None
        preds[key][0, 0, idx].backward(retain_graph=True)
        assert x.grad[:, 3:].abs().sum() > 0, f"{key} at {scale} cannot see the right image"
```

`per_scale_anchor_indices` derives the flat-index offsets from `model.stride` and the input size, matching
the `torch.cat(feats, -1)` scale ordering in `forward_head` (P3, then P4, then P5).

Two specified details that decide whether this test means anything:

- **`eval()` mode is mandatory.** In training mode backbone layers 0–1 run on the concatenated
  `[2B, 3, H, W]` batch, so BatchNorm computes statistics across left _and_ right images; the left path
  then acquires a spurious dependence on the right image through batch statistics and the test would
  pass at every scale for the wrong reason. At inference BN uses running statistics and that coupling
  disappears. (This train/eval asymmetry in the siamese batch trick is worth recording independently —
  it means training sees a weak right-image influence at all scales that vanishes at deployment.)
- **Random weights, no checkpoint, no dataset, no GPU.** Reachability is a property of the graph, not of
  training. The test runs in seconds and is cheap enough to keep in CI.

Expected result today: passes at P3, fails at P4 and P5 with `x.grad[:, 3:].abs().sum()` exactly `0.0`.
That failure is the verification. After the wiring fix the same test passes at all three scales, so it
converts directly into the regression guard — no second test to write.

### Component B — context: per-scale response decomposition

Extend the existing `_shift_probe` in `ultralytics/data/scripts/diagnose_s3d.py` (internal `s3d-exp2`
branch) rather than adding a script: it already owns matched-detection pairing and `stereo_sensitivity`.
Two deficiencies get fixed in place:

1. **It is scale-blind.** Tag each matched detection with the FPN scale of its anchor and report response
   per scale alongside the pooled number. The pooled figure is precisely the quantity that plateaued
   across the whole campaign, and it cannot distinguish "stereo is weak" from "stereo is absent for most
   objects".
2. **Its default 2 px shift is sub-bin.** One bin is 8.35 input px at 24 bins, 4.09 at cv48 — the
   documented cause of the earlier "KITTI ignores stereo" over-claim. Sweep `{4, 8, 16, 24}` px and add
   a left-copy ablation arm, which is immune to disparity quantisation entirely.

Reported per scale: detection count, median observed response against the geometric expectation, and
depth MAE under the left-copy ablation. The per-scale detection counts also replace this design's
size-band _estimate_ of the P3/P4/P5 split with a measured, detection-weighted census, for free.

### Component C — runs

| Run   | Where                                                        | Config                                              | Purpose                                                                                                                             |
| ----- | ------------------------------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Stock | local CPU, `~/weights/yolo26n-s3d.pt`                        | KITTI 1000 ep, imgsz [384,1248], 24-bin             | tests the arithmetic decomposition: 8–12% pooled should resolve to ~0.20–0.30 per-object on P3 × ~39% of objects, and 0.00 on P4/P5 |
| v6e   | rented AutoDL box, weights from the `/autodl-fs/data` backup | cv48 + photometric (campaign winner, 26–29% pooled) | per-scale strength in the recipe the fix would build on                                                                             |

The local run subsamples ~300 val images with a fixed seed: six forwards per image (base + four shifts +
ablation) at 384×1248 on CPU is roughly 10 s/image, so full val would be ~10 h. The box run does full
val. Close the box when done.

cube_s3d needs no new run — 100% of its objects are on P4 and its right-image ablation is already a
measured no-op, which is the cube-side confirmation.

## Data flow

```
Component A:  yolo26n-s3d.yaml -> Stereo3DDetModel.eval() -> random 6ch input (requires_grad)
              -> preds[key][0,0,idx].backward() -> x.grad[:, 3:] per scale -> pass/fail  [THE GATE]

Component B:  checkpoint + KITTI val -> diagnose_s3d
              -> _shift_probe(base, shifted{4,8,16,24}, left-copy) matched per detection
              -> anchor flat index -> FPN scale tag
              -> per-scale {count, median response / expected, ablation depth MAE}  [CONTEXT]
```

## Error handling and known confounds

- **BatchNorm leakage** — addressed by requiring `eval()`; recorded above as a finding in its own right.
- **Anchor-index → scale mapping** — derived from `model.stride` and input size rather than hardcoded, so
  it stays correct at any `imgsz`; asserted against the total anchor count.
- **Sub-bin shift** — addressed by the `{4, 8, 16, 24}` px sweep plus the quantisation-immune ablation arm.
- **Detection-weighted vs GT-weighted census** — the size-band split in this document is an estimate from
  GT boxes; Component B replaces it with the measured detection-weighted split. Conclusions rest on the
  measured version.
- **AP3D is spiky on small val sets** — Component B reports depth MAE and response, not AP. No AP claim is
  made anywhere in this experiment.
- **v6e checkpoint provenance** — the v6e weights unpickle only with the code that trained them; verify the
  cost-volume `num_bins` is 48 on load before trusting the per-scale numbers.

## Testing

Component A _is_ a test, and its post-fix form is the permanent regression guard. Component B is
instrumentation on an internal branch; it gets a unit test asserting that the anchor-index → scale
mapping is correct for both a square and a rectangular `imgsz` (the s3d subsystem has a documented
history of imgsz-dependent decode bugs, so this mapping is not assumed safe).

## Out of scope

No wiring change, no descriptor change, no training runs, no AP measurement. This experiment only
establishes whether the right image reaches each scale, and what it is worth where it does.
