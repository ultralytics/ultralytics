# Analysis — H8, the recovered half

_2026-09-24. `G_stepmatch` 52 epochs in 3.95 h (GPU 2), `F_clip` 100 epochs in 7.56 h (GPU 0). Every model was
re-scored in the same invocation as its controls._

## Results

**COCO GT ruler, all 2,346 val images.**

| arm | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d (`yolo26n-pose`) | 0.6645 | **0.8250** | **0.5676** |
| E_refocal (29,344 imgs, 100 ep) | 0.6298 | 0.7336 | 0.4360 |
| G_stepmatch (56,599 imgs, 52 ep) | 0.6804 | 0.7778 | 0.4624 |
| F_clip (56,599 imgs, 100 ep) | **0.6891** | 0.7833 | 0.4732 |

**COCO GT ruler, the 1,168 images E_refocal was also trained to expect.**

| arm | box mAP50-95 | pose mAP50 | pose mAP50-95 |
| --- | ---: | ---: | ---: |
| pose2d | 0.6852 | **0.8628** | **0.6169** |
| E_refocal | 0.6963 | 0.8395 | 0.5410 |
| G_stepmatch | 0.6984 | 0.8364 | 0.5275 |
| F_clip | **0.7085** | 0.8389 | 0.5383 |

**3DPW test, each arm in its own convention (`3dpw-pose3d-r12.yaml`).**

| arm | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E_refocal | **0.7803** | **109.4** | **82.05** | **0.0398** | **0.6496** |
| G_stepmatch | 0.7535 | 111.7 | 83.84 | 0.0445 | 0.6470 |
| F_clip | 0.7584 | 109.8 | 82.96 | 0.0456 | **0.6568** |

`pose2d` and `E_refocal` reproduce their H7 and H6 rows to four significant figures (0.5676 / 0.4360 / 0.6169 /
0.5410; 0.7803 / 109.4 / 82.05 / 0.0398 / 0.6496), so nothing in the evaluator drifted across the H7 and H8 code
changes, commit `93144aae3` included.

## Against the locked predictions

| prediction | G | F |
| --- | --- | --- |
| COCO GT full near 0.49, refuted below 0.46 | survives — 0.4624 | survives — 0.4732, short of the point estimate |
| COCO GT subset within 0.02 of 0.5410 | **confirmed** — 0.5275, and it moved *down* | **confirmed** — 0.5383 |
| G close to F on the COCO GT ruler | **confirmed** — 0.4624 vs 0.4732 | — |
| 3DPW MPJPE at or below E | **refuted** — 111.7 | refuted by 0.4 mm — 109.8 |
| 3DPW AbsRel(Z) at or below 0.0398 | **refuted** — 0.0445 | **refuted** — 0.0456 |

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

## The decomposition, with all three points

Pose mAP50-95 on the full COCO GT ruler: **E 0.4360 -> G 0.4624 -> F 0.4732**, a total of +0.0372.

| component | measured by | value | share |
| --- | --- | ---: | ---: |
| the recovered images, at fixed compute | E -> G | +0.0264 | 71% |
| the extra steps, at fixed data | G -> F | +0.0108 | 29% |

**The data is the dominant term, which is what H7 predicted and what the control was built to check.** Had the
gain been the schedule, G would have sat near E; it sits two thirds of the way to F.

The subset column closes the argument. G paid −0.0135 on the 1,168 images it shared with E; F, given the full
100 epochs, gives that back and lands at −0.0027 — level with E to within noise. So **the frame-edge images cost
nothing on the distribution already covered once the schedule is long enough**; at matched steps the model has
to choose, at full steps it does not.

## What did not come back

Root-depth AbsRel degrades with the recovered data and ignores the schedule entirely: **0.0398 -> 0.0445 ->
0.0456**. More steps do not repair it, so this is the data, not the optimization. The mechanism is visible in
what the recovered images are: people truncated at the frame edge, whose mid-hip root is frequently outside the
image, so the root-depth target is being regressed from a point the network cannot see. Relative depth moves the
other way — delta1(Z) is best of the three at 0.6568 — which is the same split this project keeps finding:
**root depth and relative pose respond to different things, and here they respond in opposite directions.**

MPJPE recovers with steps (111.7 -> 109.8) to within 0.4 mm of E, so the pose half of the 3D task is not
harmed by the new images once trained long enough. Only the absolute-depth half is.

## Where the 2D gap now stands

`yolo26n-pose` led `E_refocal` by 0.1316 on the full ruler. It leads `F_clip` by **0.0944** — **28% of the gap
closed**, against the ~42% H7's decomposition made available. The student is now at 83% of the 2D baseline,
from 77%.

And on detection it is no longer behind at all: **F_clip 0.6891 box mAP50-95 against `yolo26n-pose`'s 0.6645**
on the full ruler, and 0.7085 against 0.6852 on the subset. Boxes were never distilled — they are COCO's own —
so once the images came back, so did that supervision, in full. The residual 0.0944 is entirely a keypoint gap,
and the H7 ruler-swing measurement says most of what remains is the teacher's annotation convention rather than
anything the student is failing to learn.
