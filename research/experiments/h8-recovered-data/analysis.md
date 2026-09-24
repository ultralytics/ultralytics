# Analysis — H8, the recovered half

_2026-09-24. `G_stepmatch` complete (52 epochs, 3.95 h, ultra15 GPU 2). `F_clip` still training on GPU 0; its
rows land here when it finishes. Every model below was re-scored in the same invocation as its controls._

## Results so far

**COCO GT ruler, all 2,346 val images.**

| arm | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d (`yolo26n-pose`) | 0.6645 | **0.8250** | **0.5676** |
| E_refocal (29,344 imgs, 100 ep) | 0.6298 | 0.7336 | 0.4360 |
| G_stepmatch (56,599 imgs, 52 ep) | **0.6804** | 0.7778 | 0.4624 |

**COCO GT ruler, the 1,168 images E_refocal was also trained to expect.**

| arm | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d | 0.6852 | **0.8628** | **0.6169** |
| E_refocal | 0.6963 | 0.8395 | 0.5410 |
| G_stepmatch | **0.6984** | 0.8364 | 0.5275 |

**3DPW test, each arm in its own convention (`3dpw-pose3d-r12.yaml`).**

| arm | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E_refocal | **0.7803** | **109.4** | **82.05** | **0.0398** | **0.6496** |
| G_stepmatch | 0.7535 | 111.7 | 83.84 | 0.0445 | 0.6470 |

`pose2d` and `E_refocal` reproduce their H7 and H6 rows to four significant figures (0.5676 / 0.4360 / 0.6169 /
0.5410; 0.7803 / 109.4 / 82.05 / 0.0398 / 0.6496), so nothing in the evaluator drifted across the H7 and H8 code
changes, commit `93144aae3` included.

## Against the locked predictions

| prediction | outcome |
| --- | --- |
| COCO GT full: near 0.49, refuted below 0.46 | **survives, barely** — 0.4624, recovering 0.026 of the predicted 0.056 |
| COCO GT subset: within 0.02 of 0.5410 | **confirmed** — 0.5275, and it moved *down* |
| 3DPW MPJPE at or below E | **refuted** — 111.7 vs 109.4 |
| 3DPW AbsRel(Z) at or below 0.0398 | **refuted** — 0.0445 |

## What the two rulers say together

The sharp prediction was the subset one, and it landed: **the gain is localized to the images that were
missing.** At matched compute, adding the frame-edge half lifts the full-set score by +0.026 pose and **+0.051
box**, while the 1,168 images E already trained on move by −0.014 pose and +0.002 box. The recovered data buys
accuracy exactly where H7 said the loss was, and nowhere else.

That the subset moves *down* slightly is the other half of the finding: at a fixed 3.4M parameters and fixed
steps, covering the frame-edge distribution costs a little on the distribution already covered. Nothing here
separates capacity from optimization; both would look like this.

**Detection gained more than pose, and overtook the baseline.** Box mAP on the full ruler goes 0.6298 ->
0.6804, the largest single move in the table, putting `G_stepmatch` **ahead of `yolo26n-pose`'s 0.6645**. The
pose3d arm is now the better person detector on the full val set while still trailing by 0.105 on keypoints.
Boxes were never pseudo-labelled — they are COCO's own — so the recovered images restored real detection
supervision, and that is the part of the loss that came back in full.

**3DPW got worse on everything.** MPJPE +2.3 mm, AbsRel +0.005, PA-MPJPE +1.8 mm, and 3DPW pose mAP −0.027. The
recovered people are truncated at the frame edge, where the root is often outside the image and the depth
target is least constrained; at matched steps that appears to cost 3D generalization rather than buy it. This is
a real trade the protocol did not anticipate: **the images that help COCO 2D hurt the out-of-domain 3D
benchmark.**

## Where the 2D gap now stands

`yolo26n-pose` led `E_refocal` by 0.1316 on the full ruler. It leads `G_stepmatch` by **0.1052** — 20% of the
gap closed at equal compute, against the ~42% H7's decomposition made available. The rest of that 0.056 is
presumably reachable only with more than matched steps, which is what `F_clip` tests.
