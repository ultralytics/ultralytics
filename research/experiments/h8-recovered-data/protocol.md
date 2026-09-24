# Protocol — H8, the recovered half

_Locked 2026-09-24, before the run. One arm, one change from `E_refocal`: the labels on disk._

## Hypothesis

H7 measured the cost of the loader dropping 48% of the training set: restricting COCO GT to the 1,168 images
whose labels survived validation cut `yolo26n-pose`'s lead over `E_refocal` from 0.1316 to 0.0759. That assigned
**~0.056 of the 2D gap to images the model never saw**. The clip landed on 2026-09-22 (log 23) and
`coco-pose3d-refocal` now loads 56,599 train and 2,346 val with zero corrupt.

**H8: training on the recovered images closes most of that 0.056 on the COCO GT ruler, and the gain concentrates
in the frame-edge images that were missing rather than in the ones already present.**

## Arm

`F_clip` — byte-identical config to `E_refocal` (`yolo26n-pose` warm start, `coco-pose3d-refocal.yaml`, 100
epochs, batch 64, imgsz 640, seed 0). The only difference is that the dataset root now holds clipped labels, so
the loader accepts 56,599 images instead of 29,344 and 166k persons instead of 66k.

## Predictions

- **COCO GT ruler, full 2,346 val images: pose mAP50-95 near 0.49**, up from E's 0.4360 — recovering roughly the
  0.056 H7 attributed to the missing images. Below 0.46 refutes the attribution; above 0.52 means the missing
  data was costing more than H7's decomposition said.
- **COCO GT ruler, the 1,168 surviving subset: little movement**, within 0.02 of E's 0.5410. This is the sharp
  one. The recovered images are the frame-edge ones; if the subset number jumps as much as the full-set number,
  the gain is generic "more data", not the specific images H7 pointed at.
- **3DPW test (`3dpw-pose3d-r12.yaml`, still out of domain for this arm): MPJPE at or below E's 106.5 mm**, and
  **AbsRel(Z) at or below 0.0398**. The recovered people are at frame edges, where perspective is strongest, so
  depth should not get worse; a large depth gain would be a surprise worth chasing.
- **Teacher-agreement numbers are not comparable to any earlier arm** — that val split grew from 1,168 to 2,346
  images. Report them, do not rank on them.

## The confound, and the control that removes it

100 epochs over 2x the images is 2x the optimization steps, so `F_clip` alone measures "more data *and* more
steps". `G_stepmatch` removes the second half: the same arm at **52 epochs**, which is 46,020 steps against the
45,900 `E_refocal` took over 29,344 images — a 0.3% match, where a round 50 would have landed 3.6% short.
Batch 64 throughout, so steps and image-presentations are the same quantity here.

That makes a three-point line with one variable moving at a time:

| arm | images | epochs | steps | isolates |
| --- | ---: | ---: | ---: | --- |
| `E_refocal` | 29,344 | 100 | 45,900 | the baseline |
| `G_stepmatch` | 56,599 | 52 | 46,020 | **the recovered images**, at fixed compute |
| `F_clip` | 56,599 | 100 | 88,500 | **the extra steps**, at fixed data |

Added prediction: **G lands close to F on the COCO GT ruler** — near 0.48, most of the way from E's 0.4360 — if
H7 was right that the missing images, not the compute, were the binding constraint. G near E instead would say
the recovered data bought nothing and F's gain was the schedule.

One thing the control cannot equalize: G decays its learning rate over 52 epochs and E over 100, so the
schedules have the same length in steps but a different shape per epoch, and warmup (defined in iterations from
`warmup_epochs=3`) is longer for G. That residual is inherent to matching steps across datasets of different
size; it is small next to the effect being measured.

The prediction that discriminates without any of this is still the subset one: more steps would lift the 1,168
subset too, the missing images should not.

## Not in this arm

`coco-3dpw-pose3d.yaml` (log 25) folds in 3DPW train+validation and would confound the recovered COCO half with
a real-GT component, while making the benchmark in-domain. That is a separate experiment with a separate
protocol; this arm keeps the benchmark out of domain so its numbers stay on the same ruler as R0 through H6.
