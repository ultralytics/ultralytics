# SC-ELAN v3 Draft: P1d-Inspired Adaptive CAI

## 1. Design Goal

Keep the strongest structural base from v2 (`LSKA23 + TSCG`) and improve the head-side reweighting that made `v2-P1d` effective:

- Keep `alpha` conservative (`0.10`) for stability.
- Keep `beta` strong (`0.40`) for tail enhancement.
- Replace fixed `beta` with bounded dynamic routing, only during training.

Inference behavior stays equivalent to `Detect` (no extra deployment path).

## 2. Core Hypothesis

`v2-P1d` suggests that strict localization gain comes from **moderate base gain + stronger tail interaction gain**.
The next improvement should come from making this interaction **conditional**, not globally fixed:

- stronger for tail-rich states,
- stronger for detailed/high-frequency maps (small object cues),
- stronger when class confidence is uncertain,
- but always bounded to avoid unstable amplification.

## 3. Module Definition (`DetectCAIv3`)

`DetectCAIv3` extends `DetectCAI` with three additions:

1. Dynamic beta router
2. Warmup on CAI residual
3. Hard clipping of gate range

### 3.1 Dynamic Beta Router

For each pyramid level `i`:

- Tail prior score:
  `tail = sum_c prior(c) * tail_mask(c)`
- Detail score (per image):
  `detail = mean(|x - avgpool5x5(x)|) / (mean(|x|) + eps)`
- Uncertainty score (per image):
  `uncert = 1 - max softmax(logits_prior_i)`

Then:

`beta_ratio = clip(1 + g_tail*tail + g_detail*detail + g_uncert*uncert, r_min, r_max)`

`beta_dyn_i = beta_base * level_gain_i * beta_ratio`

### 3.2 Gate Construction

`gate_res = alpha * base_gate + beta_dyn_i * cond_gate * tail`

`gate = clip(1 + warmup_t * gate_res, gate_min, gate_max)`

`x_out = x * gate`

Where `warmup_t` linearly ramps from 0 to 1 in `cai_warmup_steps`.

## 4. Why This Is Logically Tight

- `alpha` fixed low: protects global calibration.
- `beta` adaptive: only pushes hard where needed, avoiding over-correction.
- uncertainty term: increases adaptation when prior prediction is weak.
- detail term: allocates more gain to likely small-object-sensitive regions.
- clipping + warmup: explicit stability constraints against oscillation/regression.

## 5. Current Draft Config

Model YAML:
- `models/sc_elan/yolo11-scelan-v3-p1d-adacai.yaml`

Head params in draft:
- `alpha=0.10`, `beta=0.40`, `momentum=0.90`
- `gamma_tail=0.35`, `gamma_detail=0.25`, `gamma_uncert=0.20`
- `beta_ratio in [0.80, 1.50]`
- `gate in [0.70, 1.60]`
- `warmup_steps=1000`
- `level_gain=[1.15, 1.00, 0.90]`

## 6. No-Seeds Validation Plan (as requested)

Single-seed, low-cost first pass:

1. Baseline: `v2-P1d`
2. New: `v3-p1d-adacai`
3. Keep all other settings identical.

Report:

- overall `P/R/mAP50/mAP50-95`
- per-class `people/bicycle/tricycle`
- latency and GFLOPs
- train stability indicators (loss spikes, NaN check, gate saturation rate)

If no global gain but tail classes improve, tighten gate bounds first before structural changes.
