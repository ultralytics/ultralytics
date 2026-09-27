# Analysis — H9, masking the root when it leaves the frame

_2026-09-27. `H_rootmask` 100 epochs in 3.45 h (ultra15 GPU 4). `F_clip` re-scored in every invocation._

## Results

**3DPW test — the headline benchmark.**

| arm | pose mAP50-95 | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | ---: | ---: | ---: | ---: |
| F_clip | 0.7584 | 109.8 | 82.96 | 0.04556 | 0.6568 |
| H_rootmask | **0.7607** | **109.2** | **81.97** | **0.04501** | **0.6573** |

**COCO val2017 (teacher labels, every root labelled), split by whether the root is in the picture.**

| ruler | persons | arm | MPJPE (mm) | PA-MPJPE (mm) | AbsRel(Z) | delta1(Z) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| all | 6,352 | F_clip | **110.0** | 63.89 | **0.0619** | 0.7540 |
| all | 6,352 | H_rootmask | 111.2 | **63.74** | 0.0762 | **0.7588** |
| framed root | 5,676 | F_clip | 107.8 | 66.13 | 0.0562 | 0.7603 |
| framed root | 5,676 | H_rootmask | **106.9** | **65.75** | **0.0560** | **0.7677** |
| truncated root | 676 | F_clip | **128.6** | **45.07** | **0.109** | **0.7025** |
| truncated root | 676 | H_rootmask | 148.4 | 46.64 | 0.246 | 0.6880 |

**COCO GT 2D rulers.**

| ruler | arm | box mAP50-95 | pose mAP50-95 |
| --- | --- | ---: | ---: |
| full, 2,346 imgs | F_clip | **0.6891** | 0.4732 |
| full, 2,346 imgs | H_rootmask | 0.6880 | **0.4747** |
| surviving, 1,168 imgs | F_clip | **0.7085** | 0.5383 |
| surviving, 1,168 imgs | H_rootmask | 0.7076 | **0.5411** |

## Against the locked predictions

| prediction | outcome |
| --- | --- |
| 3DPW AbsRel(Z) below 0.0430, null at 0.0450 | **refuted** — 0.04501, exactly the null line |
| delta1(Z) within 0.01 of 0.6568 | confirmed — 0.6573 |
| MPJPE within 2 mm of 109.8 | confirmed — 109.2 |
| COCO GT 2D within 0.005 pose, box unchanged | confirmed — +0.0015 pose, −0.0011 box |

## The answer: it is a trade, and a bad one

The split is unambiguous. On people whose root is in the picture — 89% of COCO, and effectively 100% of 3DPW —
the mask is a rounding error in the right direction: AbsRel 0.0562 -> 0.0560, MPJPE 107.8 -> 106.9, delta1
0.7603 -> 0.7677. On the 676 people whose root is outside it, **root-depth error rises from 0.109 to 0.246**, a
factor of 2.3, and MPJPE with it, 128.6 -> 148.4 mm. PA-MPJPE barely moves (45.07 -> 46.64), so the body shape
is intact and what broke is exactly what stopped being supervised: where the person is.

On the mixed population that is real COCO, the trade is **23% worse overall**: AbsRel 0.0619 -> 0.0762, all of
it from the 10.6% of persons carrying the masked root.

So the premise behind the change was wrong. The teacher's depth for a truncated person is not unlearnable just
because the mid-hip is out of shot — the visible body constrains it, and apparent size is the cue the network
uses anyway. Removing the target does not remove the prediction; it removes the only thing keeping it honest.

## Why 3DPW said the opposite, and why that matters more than this result

3DPW test contains **3 truncated-root persons out of 7,097**. It is structurally incapable of seeing the cost,
and it duly reported the mask as a small win on every metric. Shipping on the project's own headline benchmark
would have shipped a change that doubles root-depth error on a tenth of real images.

The protocol named this blind spot before the run and built the two splits for it, which is the only reason the
trade is visible. **That is the transferable result, not the mask:** 3DPW's population is not COCO's, and any
conclusion about truncated, crowded or frame-edge people has to be measured somewhere else.

It also amends H8. That experiment read root AbsRel rising 0.0398 -> 0.0456 as the recovered frame-edge images
"costing" root depth. H9 says the cost was not noise from unlearnable targets: masking those same targets makes
things worse, not better. What H8 measured was a distribution shift — a model trained on more frame-edge people
is a slightly worse fit to a benchmark that has none.

## Recommendation

**Revert the mask.** The evidence says an out-of-frame root should keep supervising. The one thing worth
keeping from the change is its cheapest half: the root's *2D location* target for those persons is a clipped
point on the image border, which is a genuinely wrong coordinate, while its depth target is fine. Splitting the
root's visibility from its depth-supervision flag would allow dropping the former and keeping the latter — but
that needs a channel the label format does not have, and on this evidence the location term is not what moved
the numbers. Not worth it yet.

The measurement to keep is the framed/truncated split. Every future depth claim should report both.
