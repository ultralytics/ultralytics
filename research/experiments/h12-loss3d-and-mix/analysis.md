# Analysis — H12, MPJPE-space loss (#3) and COCO + BEDLAM mixing (#1)

_2026-10-24, task ft-syjg. Protocol locked first (`protocol.md`, commit c5f265fb5). 5 arms on ultra11 GPUs 2-6,
~6.5 GPU-h; every arm and F_clip re-scored on the same host on six rulers. Single seed per arm._

## Answer

- **#3 works.** Adding MPJPE in metres on the lifted root-relative skeleton (`POSE3D_MPJPE_GAIN=4`, 20-epoch
  fine-tune of F_clip) cuts **3DPW MPJPE 111.1 -> 107.2 mm (−3.9)** against the same-compute control, COCO val
  −7.6 mm, BEDLAM-val −6.9 mm, root AbsRel −7%, with **no 2D cost** (COCO-GT pose 0.460 -> 0.461). It is the first
  change in the project to beat F_clip on 3DPW (107.2 vs 109.9). Gain 1 is in the same direction at a third of the
  size; the effect is monotone in the gain, so the optimum may be above 4.
- **#1 is a domain trade, not a win.** Mixing BEDLAM 1:1 makes the model much better on BEDLAM (−19 mm) and only
  slightly better on 3DPW (−1.4 mm, under the 3 mm bar), while COCO gets worse (+6 mm MPJPE, −0.018 GT pose mAP,
  truncated-person root AbsRel +18%). Not adopted at this ratio.

## #3 — loss on the lifted skeleton (all: F_clip + 20 ep COCO)

| arm | MPJPE gain | 3DPW MPJPE | 3DPW PA | 3DPW AbsRel | 3DPW δ1 | COCO MPJPE | framed | trunc MPJPE | trunc AbsRel | BEDLAM-val | COCO-GT pose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_clip (reference) | — | 109.85 | 82.96 | 0.0456 | 0.657 | 110.01 | 107.78 | 128.62 | **0.109** | 162.52 | **0.473** |
| X_ftctrl | 0 | 111.08 | 83.43 | 0.0434 | 0.652 | 111.82 | 109.46 | 131.08 | 0.115 | 161.93 | 0.460 |
| X_ftmpjpe1 | 1 | 109.78 | 82.69 | 0.0412 | 0.659 | 108.79 | 106.73 | 125.45 | 0.111 | 158.34 | 0.459 |
| **X_ftmpjpe4** | 4 | **107.23** | **81.58** | **0.0402** | **0.670** | **104.27** | **102.32** | **120.04** | 0.117 | **155.00** | 0.461 |

vs control at gain 4: 3DPW −3.85 mm / PA −1.85 / AbsRel −7.4% / δ1 +0.018; COCO −7.6; framed −7.1; truncated −11.0
MPJPE with root AbsRel +1.5% (inside the 10% guard); BEDLAM −6.9; COCO-GT pose +0.002, box −0.0003. **Meets the
decision rule.**

Why it works where H10's bone prior did not: the bone prior constrained something the student already roughly
satisfied (lengths), while this term supervises the error MPJPE actually measures — relative depth, 2D and root
depth combined through `x = (u − cx) z / f` — in metres, weighting a joint by how far its 3D position is off rather
than by its encoded-depth residual. Relative-depth δ1 rising 0.652 -> 0.670 is the channel H10 identified as 55% of
the error, moving.

## #1 — COCO + BEDLAM 1:1, step-matched

| arm | init | 3DPW MPJPE | 3DPW PA | 3DPW AbsRel | COCO MPJPE | trunc AbsRel | BEDLAM-val | COCO-GT pose | vs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F_clip | yolo26n-pose, 100 ep COCO | 109.85 | 82.96 | 0.0456 | **110.01** | **0.109** | 162.52 | **0.473** | — |
| M_mix50 | yolo26n-pose, 50 ep mix | **108.42** | **81.06** | **0.0433** | 115.96 | 0.128 | **143.22** | 0.455 | F_clip |
| X_ftctrl | F_clip + 20 ep COCO | 111.08 | 83.43 | 0.0434 | **111.82** | **0.115** | 161.93 | **0.460** | — |
| M_mixft10 | F_clip + 10 ep mix | 111.16 | 82.30 | 0.0445 | 118.29 | 0.117 | **146.55** | 0.443 | X_ftctrl |

The mix buys what BEDLAM labels are good for — BEDLAM-val −19.3 / −15.4 mm — and transfers only weakly to real
3DPW (−1.4 / +0.1 mm). Halving the COCO steps costs the COCO domain everywhere: teacher-label MPJPE +6.0 / +6.5,
GT 2D pose −0.018 / −0.016, truncated root depth +18% (BEDLAM's people are almost never cut by the frame). Box mAP on
COCO-GT drops 0.022 — BEDLAM boxes are joint hulls, a different box convention from COCO's.

## Against the locked predictions

| # | prediction | outcome |
| --- | --- | --- |
| A1 | gain 1: −2 to −6 mm 3DPW, PA ≥ 3 mm | **refuted (too small)** — −1.3 mm, PA −0.7 |
| A2 | gain 4 helps more but costs > 0.01 pose mAP | **half** — helps more (−3.9 mm), **no 2D cost** (+0.002) |
| A3 | root AbsRel within ±5% | **refuted (better)** — −7.4% at gain 4 |
| B1 | mix50 beats F_clip ≥ 3 mm 3DPW and ≥ 10 mm BEDLAM-val | **half** — BEDLAM −19.3 ✓, 3DPW −1.4 ✗ |
| B2 | COCO-GT pose within 0.01 | **refuted** — −0.018 |
| B3 | mixft10 beats X_ftctrl on 3DPW | **refuted** — +0.1 mm |

## Next

1. **Adopt the MPJPE term** in the pose3d recipe; sweep the gain above 4 (8, 16) and train it from the start (100 ep
   from yolo26n-pose) rather than as a fine-tune — the fine-tune is the cheapest test, not the best use.
2. Seeds: the 3.9 mm gain is ~2-4× the seed spread H10 measured; confirm with a second seed before shipping.
3. Mixing: retry at a lower BEDLAM share (1:4 or 1:8) with the MPJPE term on, and/or keep COCO's step budget whole
   (mix as extra steps, not a replacement), so the COCO domain is not traded away.
