# Protocol — R0 (warm start) and H2 (augmentation vs metric depth)

_Locked 2026-09-16, before any run. Three arms, one change each, so both hypotheses read off a single pair._

## Arms

| arm | change vs the arm above | answers |
| --- | --- | --- |
| `A_scratch` | — (random init, default augmentation) | R0 control |
| `B_warmstart` | `.load(yolo26n-pose.pt)` | **R0** = B − A |
| `C_depthsafe` | `mosaic=0.0, scale=0.0` | **H2** = C − B |

Everything else is identical and fixed: `yolo26n-pose3d.yaml`, `coco-pose3d.yaml`, 640 px, 100 epochs,
batch 64, one RTX 6000D per arm, `seed=0`, `workers=8`. Arms run concurrently on ultra15 GPUs 0, 1, 3.

Warm start verified before launch: `.load()` transfers **720/792 tensors** — the whole trunk plus the box and
class branches. The 120 keypoint-head tensors stay at init because the channel count changes (17x3 = 51 -> 18x4
= 72), which is the intended behaviour: R0 tests whether a 2D-pose trunk is a better starting point, not
whether the 3D head can be copied.

## Predictions, written down before the numbers exist

- **R0**: B beats A on every metric, and by a lot. Precedent is s3d, where a pretrained backbone plus a long
  schedule moved Car AP3D@0.5 from 34.3 to 52.0 — more than any architectural lever tested there. If B does
  **not** clearly beat A, the 2D-pose trunk is not transferring and the warm-start assumption behind the whole
  distillation plan needs revisiting.
- **H2**: C beats B on the depth metrics (`AbsRel(Z)` down, `delta1(Z)` up) and loses a little 2D mAP, because
  mosaic and scale are real regularizers. The mechanism is specific: apparent size is the monocular depth cue,
  and resizing a person changes that cue without changing the metric label, so B is trained on a systematically
  corrupted depth target while its 2D supervision stays correct. If C shows **no** depth gain, then either the
  corruption averages out, or root depth is being driven by something other than apparent size — both would be
  more interesting than the expected result.

## Metrics

Primary, from `Pose3DMetrics`: `MPJPE`, `PA-MPJPE`, `AbsRel(Z)`, `delta1(Z)`, alongside box and pose mAP.
These are **teacher agreement** on the held-out val2017 pseudo-labels, not accuracy — 3DPW is on disk
(`ultra11:/data/rick/datasets/3dpw`) and the real-GT evaluation comes after, on the winning arm.

## Known confound, recorded now

Letterboxing to 640 is itself a resize, and its ratio varies per image, so the apparent-size-to-depth relation
is not constant even in arm C. It applies equally to all three arms, so the A/B comparisons hold, but it caps
how well any of them can do on absolute depth, and it is the obvious next thing to fix if H2 is supported.
