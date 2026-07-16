# s3d Tier-1 A/B (projected-center + depth-uncertainty) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two flag-gated localization levers to the s3d head/decode — a projected-3D-center offset and heteroscedastic depth uncertainty (with inverse-variance fusion and score-weighting) — plus the A/B harness to run 4 training runs → 8 eval arms on KITTI.

**Architecture:** Both levers default OFF (baseline behavior unchanged). Train-time flags (`use_proj_center`, `use_depth_uncertainty`) in the model YAML `training:` block are read in `model.py` and used to (a) enable head branches via post-build head methods (mirroring the existing `set_depth_mode` pattern) and (b) configure `Stereo3DDetLoss`. Decode-time knobs (`ivw_fusion`, `score_weight`, `use_proj_center`) reach `decode_stereo3d_outputs` through `self.args` via `getattr` (mirroring the existing `use_geometric` arg), so fusion-vs-score is toggled at val time on a shared checkpoint.

**Tech Stack:** PyTorch, Ultralytics YOLO engine, pytest. KITTI stereo. Runs execute on the weste seetacloud box.

## Global Constraints

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` (Actions bot adds it; don't add manually).
- Google-style docstrings; Ruff line length 120; `ruff format . && ruff check --fix .` before commit.
- Delete > Replace > Add. Flags are experiment scaffolding; the winning path becomes unconditional and losing branches/flags are deleted after the A/B. Every commit body carries a `Deleted:` line.
- Baseline to preserve: with both flags OFF and decode knobs OFF, output must be byte-identical to current behavior (Car Moderate AP3D@0.5=34.3, @0.7=4.2). Task 1 asserts this.
- Tests run on the weste GPU box (torch present); CPU-only unit tests must not require CUDA. Run pattern: `python -m pytest tests/test_s3d.py::<name> -v -p no:cacheprovider`.
- Base branch: `001-stereo-centernet-gaps` @ ec3af8bdc (imgsz depth fix present). Work in the `worktree-s3d-tier1-ab` worktree.

---

### Task 1: Config-flag plumbing + head feature-enable methods

**Files:**

- Modify: `ultralytics/models/yolo/s3d/head.py` (add `enable_proj_center`, `enable_depth_uncertainty` to `Stereo3DDetHead`)
- Modify: `ultralytics/models/yolo/s3d/model.py:46-53` (read new flags, call head methods)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Produces: `Stereo3DDetHead.enable_proj_center()` — adds `"proj_offset": 2` to `self.aux_specs` and builds `self.aux["proj_offset"]` as a per-scale `nn.ModuleList` of `_branch(x, 2, hidden)`. `Stereo3DDetHead.enable_depth_uncertainty()` — sets `self.use_uncertainty = True` and rebuilds the `lr_distance` branch list to output 2 channels (value + log-variance) via `_deep_branch(in_ch, 2, depth_hidden)`; `forward_head` splits the 2nd channel into `preds["lr_logvar"]`.
- Consumes: nothing (first task).

- [ ] **Step 1: Write the failing test**

```python
def test_feature_flags_gate_head_branches():
    """use_proj_center / use_depth_uncertainty in YAML training block add head branches; default off is unchanged."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS

    base = YOLO("yolo26n-s3d.yaml").model
    head = base.model[-1]
    assert "proj_offset" not in head.aux_specs
    assert head.aux["lr_distance"][0][-1].out_channels == 1  # scalar disparity
    assert getattr(head, "use_uncertainty", False) is False

    # Enable both features directly (model.py wires these from the YAML training block).
    head.enable_proj_center()
    head.enable_depth_uncertainty()
    assert head.aux_specs["proj_offset"] == 2
    assert head.aux["proj_offset"][0][-1].out_channels == 2
    assert head.aux["lr_distance"][0][-1].out_channels == 2  # value + log-variance
    assert head.use_uncertainty is True
    # orientation/depth untouched
    assert head.aux_specs["orientation"] == ORIENT_CHANNELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3d.py::test_feature_flags_gate_head_branches -v -p no:cacheprovider`
Expected: FAIL — `AttributeError: 'Stereo3DDetHead' object has no attribute 'enable_proj_center'`.

- [ ] **Step 3: Write minimal implementation**

In `head.py`, add to `Stereo3DDetHead` (reuse existing `_branch`, `_deep_branch`, and the `hidden`/`depth_hidden` sizing from `__init__` — store them as `self._hidden`, `self._depth_hidden`, `self._ch` in `__init__` so the enable methods can rebuild branches):

```python
    def enable_proj_center(self) -> None:
        """Add the projected-3D-center offset branch (2ch: Δu, Δv). Idempotent."""
        if "proj_offset" in self.aux_specs:
            return
        self.aux_specs["proj_offset"] = 2
        self.aux["proj_offset"] = nn.ModuleList(_branch(x, 2, self._hidden) for x in self._ch)

    def enable_depth_uncertainty(self) -> None:
        """Widen the lr_distance branch to emit a log-variance channel and flag NLL/decode use. Idempotent."""
        if getattr(self, "use_uncertainty", False):
            return
        self.use_uncertainty = True
        self.aux_specs["lr_distance"] = 2  # value + log-variance
        branches = [_deep_branch(x + (self.cv_ch if i == 0 else 0), 2, self._depth_hidden)
                    for i, x in enumerate(self._ch)]
        self.aux["lr_distance"] = nn.ModuleList(branches)
```

Add `self.use_uncertainty = False`, `self._hidden = hidden`, `self._depth_hidden = depth_hidden`, `self._ch = ch` in `__init__` (near the existing `hidden`/`depth_hidden` computation). In `forward_head`, after the aux loop, split the uncertainty channel:

```python
        if getattr(self, "use_uncertainty", False) and "lr_distance" in preds:
            lr = preds["lr_distance"]
            preds["lr_distance"] = lr[:, :1]        # value
            preds["lr_logvar"] = lr[:, 1:2]         # log-variance
```

In `model.py`, alongside the existing `depth_mode` handling (~line 47-53):

```python
        training_cfg = (self.yaml or {}).get("training", {})
        head = self.model[-1]
        if training_cfg.get("use_proj_center", False):
            head.enable_proj_center()
        if training_cfg.get("use_depth_uncertainty", False):
            head.enable_depth_uncertainty()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_s3d.py::test_feature_flags_gate_head_branches -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Baseline-unchanged regression + commit**

Run: `python -m pytest tests/test_s3d.py -v -p no:cacheprovider -k "not test_train and not test_predict and not test_export"`
Expected: all PASS (including `test_val`, `test_depth_decode_imgsz_invariant`).

```bash
ruff format ultralytics/models/yolo/s3d/head.py ultralytics/models/yolo/s3d/model.py && ruff check --fix ultralytics/models/yolo/s3d/head.py ultralytics/models/yolo/s3d/model.py
git add ultralytics/models/yolo/s3d/head.py ultralytics/models/yolo/s3d/model.py tests/test_s3d.py
git commit -m "Add flag-gated head branches for proj-center and depth-uncertainty

Deleted: nothing (additive scaffolding; branches default off, baseline byte-identical)."
```

---

### Task 2: Projected-3D-center target encoding

**Files:**

- Modify: `ultralytics/models/yolo/s3d/dataset.py` (aux-target assembly loop, ~lines 813-855; per-image stack ~856+)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Consumes: per-object `location_3d` (bottom-center x,y,z, camera frame), `dimensions_3d` (height), the sample calib (fx, fy, cx, cy), the 2D box center `cx,cy` (letterbox-normalized), and the sample letterbox `ratio_pad` (scale, pad_left, pad_top) already applied to labels.
- Produces: `aux_targets["proj_offset"]` shape `[B, max_n, 2]` = (Δu, Δv) in letterbox-normalized units; a module helper `encode_proj_offset(location_3d, height, calib, box_center_norm, ratio_pad, input_wh) -> (du, dv)`.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_proj_offset_roundtrip():
    """Projected-centroid offset must invert: encode (centroid->projected px->offset) then decode (box_center+offset ->
    back-project at true z) recovers the centroid X/Y.
    """
    from ultralytics.models.yolo.s3d.dataset import encode_proj_offset

    fx = fy = 721.5377
    cx = 609.5593
    cy = 172.8540
    calib = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    input_w, input_h = 1248, 384
    scale, pad_left, pad_top = 1.0048, 0, 3  # aspect-preserving letterbox of 375x1242
    # A car: bottom-center location (X,Y,Z), height h. Centroid Y = Y - h/2 (camera y-down).
    X, Y, Z, h = 3.0, 1.65, 20.0, 1.5
    # 2D box center (letterbox-normalized) — deliberately NOT the projected centroid.
    box_u_px = (2.4 * scale) + pad_left + cx * scale  # arbitrary; see impl for framing
    box_center_norm = (0.42, 0.55)  # (u,v) normalized in letterbox space

    du, dv = encode_proj_offset((X, Y, Z), h, calib, box_center_norm, (scale, pad_left, pad_top), (input_w, input_h))

    # Decode: recovered projected center in original px, then back-project at Z.
    u_norm = box_center_norm[0] + du
    v_norm = box_center_norm[1] + dv
    u_lb = u_norm * input_w
    v_lb = v_norm * input_h
    u_orig = (u_lb - pad_left) / scale
    v_orig = (v_lb - pad_top) / scale
    x_rec = (u_orig - cx) * Z / fx
    y_rec = (v_orig - cy) * Z / fy
    assert abs(x_rec - X) < 1e-3, f"x {x_rec} != {X}"
    assert abs(y_rec - (Y - h / 2)) < 1e-3, f"y {y_rec} != centroid {Y - h / 2}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3d.py::test_proj_offset_roundtrip -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'encode_proj_offset'`.

- [ ] **Step 3: Write minimal implementation**

Add to `dataset.py` (module level):

```python
def encode_proj_offset(location_3d, height, calib, box_center_norm, ratio_pad, input_wh):
    """Encode the offset from the 2D box center to the projected 3D centroid (letterbox-normalized).

    Args:
        location_3d (tuple): (X, Y, Z) bottom-center in camera frame (meters).
        height (float): 3D box height (meters); centroid Y = Y - height/2 (camera y-down).
        calib (dict): fx, fy, cx, cy (original-image pixels).
        box_center_norm (tuple): (u, v) 2D box center, letterbox-normalized [0,1].
        ratio_pad (tuple): (scale, pad_left, pad_top) of the letterbox applied to this sample.
        input_wh (tuple): (input_w, input_h) letterbox canvas size.

    Returns:
        (du, dv): projected-centroid minus box-center, in letterbox-normalized units.
    """
    X, Y, Z = location_3d
    Yc = Y - height / 2.0
    scale, pad_left, pad_top = ratio_pad
    input_w, input_h = input_wh
    u_orig = calib["fx"] * X / Z + calib["cx"]
    v_orig = calib["fy"] * Yc / Z + calib["cy"]
    u_lb = u_orig * scale + pad_left
    v_lb = v_orig * scale + pad_top
    u_norm = u_lb / input_w
    v_norm = v_lb / input_h
    return u_norm - box_center_norm[0], v_norm - box_center_norm[1]
```

Wire it into the aux-target loop (near the orientation encode, ~line 842-847): compute `(du, dv)` per object from `location_3d[j]`, `dimensions_3d[j][2]`, the sample calib, the object's `(cx, cy)` (already normalized), and the sample `ratio_pad`/`input_wh` (from the label transform). Append to a `proj_list`, then `per_image_aux["proj_offset"].append(torch.stack(proj_list, 0))`, guarded by `if self.use_proj_center` (a dataset attribute set from `data_cfg`/model config — default False so baseline is unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_s3d.py::test_proj_offset_roundtrip -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format ultralytics/models/yolo/s3d/dataset.py && ruff check --fix ultralytics/models/yolo/s3d/dataset.py
git add ultralytics/models/yolo/s3d/dataset.py tests/test_s3d.py
git commit -m "Encode projected-3D-centroid offset target for s3d

Deleted: nothing (new aux target, gated by use_proj_center; baseline unchanged)."
```

---

### Task 3: Projected-center loss term

**Files:**

- Modify: `ultralytics/models/yolo/s3d/loss.py` (`__init__` flags, `_compute_aux_losses` loop ~line 145, `loss` tally ~line 272-289)
- Modify: `ultralytics/models/yolo/s3d/model.py:143-158` (`init_criterion` passes `use_proj_center`)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Consumes: `aux_targets["proj_offset"]` (Task 2), `preds["proj_offset"]` (Task 1 head).
- Produces: a `proj_center` entry in the loss dict, weighted by `loss_weights.proj_center`.

- [ ] **Step 1: Write the failing test**

```python
def test_proj_center_loss_present():
    """When use_proj_center is set, the aux-loss dict gains a smooth-L1 proj_center term."""
    import torch

    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    model.model[-1].enable_proj_center()
    crit = Stereo3DDetLoss(model, loss_weights={"proj_center": 1.0}, use_proj_center=True)
    B, HW = 2, 5
    preds = {"proj_offset": torch.zeros(B, 2, HW)}
    gt = torch.ones(B, 3, 2)  # [B, max_n, 2]
    idx = torch.zeros(B, HW, dtype=torch.long)
    fg = torch.ones(B, HW, dtype=torch.bool)
    losses = crit._compute_aux_losses(preds, {"aux_targets": {"proj_offset": gt}}, idx, fg)
    assert "proj_center" in losses and losses["proj_center"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3d.py::test_proj_center_loss_present -v -p no:cacheprovider`
Expected: FAIL — `Stereo3DDetLoss.__init__() got an unexpected keyword argument 'use_proj_center'`.

- [ ] **Step 3: Write minimal implementation**

In `loss.py` `__init__`, add `use_proj_center: bool = False` and `use_uncertainty: bool = False` params, store on `self`. In `_compute_aux_losses`, extend the key loop to include `"proj_offset"` mapped to a `proj_center` loss via the existing `_aux_loss` (smooth-L1) path:

```python
        if self.use_proj_center and "proj_offset" in aux_targets and "proj_offset" in aux_preds:
            aux_losses["proj_center"] = self._aux_loss(
                aux_preds["proj_offset"], aux_targets["proj_offset"].to(self.device),
                target_gt_idx, fg_mask, aux_weights,
            )
```

Add `proj_offset` to the `aux_keys` set in `loss()` (line 269) and extend the `loss` tensor to 7 slots, tallying `proj_center` with weight `self.aux_w.get("proj_center", 1.0)`. In `model.py::init_criterion`, read `use_proj_center`/`use_depth_uncertainty` from the training block and pass them to `Stereo3DDetLoss(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_s3d.py::test_proj_center_loss_present -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format ultralytics/models/yolo/s3d/loss.py ultralytics/models/yolo/s3d/model.py && ruff check --fix ultralytics/models/yolo/s3d/loss.py ultralytics/models/yolo/s3d/model.py
git add ultralytics/models/yolo/s3d/loss.py ultralytics/models/yolo/s3d/model.py tests/test_s3d.py
git commit -m "Add proj_center smooth-L1 loss term (gated by use_proj_center)

Deleted: nothing (new gated loss term; off by default)."
```

---

### Task 4: Projected-center decode

**Files:**

- Modify: `ultralytics/models/yolo/s3d/preprocess.py` (`decode_stereo3d_outputs` signature + the u,v computation ~lines 249-256; `decode_and_refine_predictions` pass-through)
- Modify: `ultralytics/models/yolo/s3d/val.py:249-258` (pass `use_proj_center` from `self.args`)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Consumes: `preds["proj_offset"]` sampled at the anchor; the `use_proj_center` decode flag.
- Produces: corrected `(x_3d, y_3d)` and `ray_angle`.

- [ ] **Step 1: Write the failing test**

```python
def test_decode_uses_proj_offset():
    """With use_proj_center, a nonzero proj_offset shifts the recovered x_3d by du*input_w/scale*z/fx."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params, decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3
    input_h, input_w = imgsz
    scale, _pad_left, _ = compute_letterbox_params(*ori_hw, imgsz)
    z = 20.0
    det = torch.zeros(1, 4 + nc, 1)
    det[0, :4, 0] = torch.tensor([input_w / 2, input_h / 2, 20.0, 20.0])
    det[0, 4, 0] = 0.99
    du = 0.01
    outputs = {
        "det": det,
        "dimensions": torch.zeros(1, 3, 1),
        "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
        "depth": torch.tensor([[[math.log(z)]]]),
        "proj_offset": torch.tensor([[[du], [0.0]]]).float(),
    }
    x_off = decode_stereo3d_outputs(outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], use_proj_center=True)[0][
        0
    ].center_3d[0]
    x_no = decode_stereo3d_outputs(outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], use_proj_center=False)[0][
        0
    ].center_3d[0]
    expected_shift = (du * input_w / scale) * z / calib["fx"]
    assert abs((x_off - x_no) - expected_shift) < 1e-2, f"{x_off - x_no} != {expected_shift}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3d.py::test_decode_uses_proj_offset -v -p no:cacheprovider`
Expected: FAIL — `decode_stereo3d_outputs() got an unexpected keyword argument 'use_proj_center'`.

- [ ] **Step 3: Write minimal implementation**

Add `use_proj_center: bool = False` to `decode_stereo3d_outputs`. Sample `proj_offset` at `flat_idx` like the other aux keys, and adjust the letterbox center before un-letterboxing (~line 249-252):

```python
            u_letterbox = float(((x1_l + x2_l) / 2.0).item())
            v_letterbox = float(((y1_l + y2_l) / 2.0).item())
            if use_proj_center and "proj_offset" in outputs:
                off = outputs["proj_offset"][b, :, flat_idx].float()
                u_letterbox += float(off[0]) * input_w
                v_letterbox += float(off[1]) * input_h
            u_orig = (u_letterbox - pad_left) / letterbox_scale
            v_orig = (v_letterbox - pad_top) / letterbox_scale
```

Thread `use_proj_center` through `decode_and_refine_predictions` → `decode_stereo3d_outputs`. In `val.py`, read `getattr(self.args, "use_proj_center", None)` and pass it (mirroring `use_geometric`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_s3d.py::test_decode_uses_proj_offset -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py && ruff check --fix ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py
git add ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py tests/test_s3d.py
git commit -m "Decode 3D center from predicted projected-center offset (gated)

Deleted: nothing (gated decode path; off by default)."
```

---

### Task 5: Depth-uncertainty NLL loss

**Files:**

- Modify: `ultralytics/models/yolo/s3d/loss.py` (add `_lr_nll_loss`; switch lr_distance term when `use_uncertainty`)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Consumes: `preds["lr_distance"]` (value) + `preds["lr_logvar"]` (Task 1), `aux_targets["lr_distance"]`.
- Produces: a Laplacian-NLL lr_distance loss.

- [ ] **Step 1: Write the failing test**

```python
def test_lr_nll_attenuates_with_uncertainty():
    """Laplacian NLL: for a fixed residual, a larger predicted log-variance lowers the loss (attenuation), but the
    log-variance penalty prevents collapse — loss is convex in logvar.
    """
    import torch

    from ultralytics.models.yolo.s3d.loss import laplacian_nll

    pred = torch.tensor([1.0])
    tgt = torch.tensor([3.0])  # residual 2.0 (min at logvar=ln2≈0.69, between probes 0 and 1)
    low = laplacian_nll(pred, tgt, logvar=torch.tensor([0.0]))
    mid = laplacian_nll(pred, tgt, logvar=torch.tensor([1.0]))
    high = laplacian_nll(pred, tgt, logvar=torch.tensor([5.0]))
    assert mid < low, "some uncertainty should reduce loss vs zero-variance for a nonzero residual"
    assert high > mid, "excessive uncertainty is penalized by the logvar term"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_s3d.py::test_lr_nll_attenuates_with_uncertainty -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'laplacian_nll'`.

- [ ] **Step 3: Write minimal implementation**

In `loss.py` (module level):

```python
def laplacian_nll(pred, target, logvar):
    """Laplacian negative log-likelihood: |pred-target|*exp(-logvar) + logvar. Mean-reduced."""
    return (torch.abs(pred - target) * torch.exp(-logvar) + logvar).mean()
```

Add `_lr_nll_loss(self, pred_val, pred_logvar, aux_gt, gt_idx, fg_mask, aux_weights)` mirroring `_aux_loss`'s gather/fg-select, using `laplacian_nll` on the foreground residual. In `_compute_aux_losses`, when `self.use_uncertainty and "lr_logvar" in aux_preds`, route `lr_distance` through `_lr_nll_loss(aux_preds["lr_distance"], aux_preds["lr_logvar"], ...)` instead of `_aux_loss`. Add `"lr_logvar"` to the `aux_keys` set in `loss()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_s3d.py::test_lr_nll_attenuates_with_uncertainty -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format ultralytics/models/yolo/s3d/loss.py && ruff check --fix ultralytics/models/yolo/s3d/loss.py
git add ultralytics/models/yolo/s3d/loss.py tests/test_s3d.py
git commit -m "Add Laplacian-NLL depth-uncertainty loss for lr_distance (gated)

Deleted: nothing (gated loss path; plain smooth-L1 retained when off)."
```

---

### Task 6: Decode uncertainty knobs — inverse-variance fusion + score-weighting

**Files:**

- Modify: `ultralytics/models/yolo/s3d/preprocess.py` (`decode_stereo3d_outputs`: `ivw_fusion`, `score_weight`, `score_k` params; fusion block ~lines 226-247; confidence ~line 199/280)
- Modify: `ultralytics/models/yolo/s3d/val.py` (pass `ivw_fusion`, `score_weight` from `self.args`)
- Test: `tests/test_s3d.py`

**Interfaces:**

- Consumes: `preds["lr_logvar"]` and the DFL depth distribution spread (recompute `σ_direct²` from the raw `depth_bins` if present, else treat as high variance).
- Produces: inverse-variance-fused `z_3d`; optionally score scaled by `exp(-score_k * σ_total)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ivw_fusion_equal_sigma_matches_geomean():
    """With equal per-cue variance, inverse-variance fusion in log-space == geometric mean (A0 continuity)."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3
    det = torch.zeros(1, 4 + nc, 1)
    det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
    det[0, 4, 0] = 0.99
    # disparity cue and direct cue encode different depths so the mean is nontrivial.
    outputs = {
        "det": det,
        "dimensions": torch.zeros(1, 3, 1),
        "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
        "lr_distance": torch.tensor([[[math.log(0.03)]]]),
        "depth": torch.tensor([[[math.log(25.0)]]]),
        "lr_logvar": torch.tensor([[[0.0]]]),
    }
    z_geo = decode_stereo3d_outputs(outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], ivw_fusion=False)[0][
        0
    ].center_3d[2]
    z_ivw = decode_stereo3d_outputs(outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], ivw_fusion=True)[0][
        0
    ].center_3d[2]
    # equal-variance IVW reduces to the geometric mean
    assert abs(z_ivw - z_geo) < 1e-2, f"ivw {z_ivw} != geomean {z_geo}"


def test_score_weight_demotes_uncertain():
    """score_weight multiplies confidence by exp(-k*sigma): higher lr_logvar => lower final score."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3

    def conf(logvar):
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
        det[0, 4, 0] = 0.9
        outputs = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(0.03)]]]),
            "depth": torch.tensor([[[math.log(25.0)]]]),
            "lr_logvar": torch.tensor([[[logvar]]]),
        }
        return decode_stereo3d_outputs(
            outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], score_weight=True, score_k=0.5
        )[0][0].confidence

    assert conf(4.0) < conf(0.0), "higher uncertainty must lower the score"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_s3d.py::test_ivw_fusion_equal_sigma_matches_geomean tests/test_s3d.py::test_score_weight_demotes_uncertain -v -p no:cacheprovider`
Expected: FAIL — unexpected keyword arguments `ivw_fusion` / `score_weight`.

- [ ] **Step 3: Write minimal implementation**

Add `ivw_fusion: bool = False, score_weight: bool = False, score_k: float = 0.5` to `decode_stereo3d_outputs`. Sample `lr_logvar` at the anchor when present. Replace the fusion block:

```python
            if z_from_disp is not None and z_from_direct is not None:
                if ivw_fusion and lr_logvar is not None:
                    var_disp = math.exp(lr_logvar)
                    var_direct = _dfl_variance(outputs, b, flat_idx)  # spread of depth bins, else 1.0
                    w_disp, w_direct = 1.0 / max(var_disp, eps), 1.0 / max(var_direct, eps)
                    log_z = (w_disp * math.log(z_from_disp) + w_direct * math.log(z_from_direct)) / (w_disp + w_direct)
                    z_3d = math.exp(log_z)
                else:
                    z_3d = math.sqrt(z_from_disp * z_from_direct)  # geometric mean (equal-weight IVW)
```

Add `_dfl_variance(outputs, b, idx)` computing `Σ pᵢ(bᵢ-μ)²` from `outputs["depth_bins"]` when present (raw logits → softmax over `DepthDFL.bin_values`), else return `1.0`. After computing `confidence` (~line 199), if `score_weight and lr_logvar is not None`: `confidence *= math.exp(-score_k * math.sqrt(math.exp(lr_logvar)))`. Thread the three params through `decode_and_refine_predictions` and pass `ivw_fusion`/`score_weight` from `self.args` in `val.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_s3d.py::test_ivw_fusion_equal_sigma_matches_geomean tests/test_s3d.py::test_score_weight_demotes_uncertain -v -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest tests/test_s3d.py -v -p no:cacheprovider -k "not test_train and not test_predict and not test_export"`
Expected: all PASS.

```bash
ruff format ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py && ruff check --fix ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py
git add ultralytics/models/yolo/s3d/preprocess.py ultralytics/models/yolo/s3d/val.py tests/test_s3d.py
git commit -m "Add inverse-variance depth fusion and uncertainty score-weighting to s3d decode

Deleted: nothing (decode knobs default off; geometric-mean path is the equal-variance case)."
```

---

### Task 7: A/B harness — 4 training configs + launch/eval matrix

**Files:**

- Create: `ultralytics/cfg/models/26/yolo26-s3d-T0.yaml`, `-Tc.yaml`, `-Tsigma.yaml`, `-Tcsigma.yaml` (each = base yaml + the relevant `training:` flags)
- Create: `ultralytics/data/scripts/run_tier1_ab.sh` (train 4 runs, then val 8 arms)
- Test: manual smoke on kitti-stereo8 (2-epoch), then full run on weste

**Interfaces:**

- Consumes: the flags/knobs from Tasks 1-6.
- Produces: an 8-row results table (Car Easy/Mod/Hard AP3D@{0.5,0.7}, AP_BEV, AOS).

- [ ] **Step 1: Create the four configs**

Each config is the current `yolo26-s3d.yaml` with an added `training:` block. Example `yolo26-s3d-Tcsigma.yaml` overlay:

```yaml
training:
    use_proj_center: true
    use_depth_uncertainty: true
    loss_weights: { lr_distance: 2.0, depth: 3.0, dimensions: 1.0, orientation: 1.0, proj_center: 1.0 }
```

T0 = no new flags; Tc = `use_proj_center: true` (+ proj_center weight); Tsigma = `use_depth_uncertainty: true`.

- [ ] **Step 2: Write the launch/eval script**

`run_tier1_ab.sh` (parameterized by `DATA`, `EPOCHS=200`, `BATCH=32`, `DEVICE`):

```bash
#!/usr/bin/env bash
set -euo pipefail
DATA=${DATA:?}
EP=${EPOCHS:-200}
B=${BATCH:-32}
for run in T0 Tc Tsigma Tcsigma; do
  yolo train task=s3d model=yolo26-s3d-$run.yaml data=$DATA epochs=$EP batch=$B val=False amp=False \
    project=runs/tier1_ab name=$run
done
# Eval matrix: model, and decode flags. Columns: use_proj_center ivw_fusion score_weight
eval_arm() { # $1=name $2=run $3=proj $4=ivw $5=score
  yolo val task=s3d model=runs/tier1_ab/$2/weights/best.pt data=$DATA \
    use_proj_center=$3 ivw_fusion=$4 score_weight=$5 project=runs/tier1_ab/eval name=$1
}
eval_arm A0_baseline T0 False False False
eval_arm A1_center Tc True False False
eval_arm A2_fusion Tsigma False True False
eval_arm A3_score Tsigma False False True
eval_arm A4_fusion_score Tsigma False True True
eval_arm A5_full Tcsigma True True True
eval_arm A6_center_fusion Tcsigma True True False
eval_arm A7_center_score Tcsigma True False True
```

- [ ] **Step 3: Smoke test on kitti-stereo8 (2 epochs, CPU/1-GPU)**

Run (locally or on the box): `DATA=kitti-stereo8.yaml EPOCHS=2 BATCH=2 bash ultralytics/data/scripts/run_tier1_ab.sh`
Expected: all 4 trainings finish; all 8 val arms produce an AP3D/BEV/AOS dict without error (metrics ~0 at 2ep/8img is fine). This exercises every flag path end-to-end.

- [ ] **Step 4: Verify arg pass-through**

Confirm `use_proj_center`/`ivw_fusion`/`score_weight` reach `self.args` in val (the smoke run's A2 must differ from A0 on the same imagery). If `get_cfg` strips unknown keys, register them in `ultralytics/cfg/default.yaml` (with `False` defaults) — check first, add only if needed.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/cfg/models/26/yolo26-s3d-T0.yaml ultralytics/cfg/models/26/yolo26-s3d-Tc.yaml ultralytics/cfg/models/26/yolo26-s3d-Tsigma.yaml ultralytics/cfg/models/26/yolo26-s3d-Tcsigma.yaml ultralytics/data/scripts/run_tier1_ab.sh
git commit -m "Add Tier-1 A/B configs and launch/eval harness

Deleted: nothing (experiment scaffolding; removed after a winner is chosen)."
```

- [ ] **Step 6: Full run on weste** (out-of-band, after setup: SSH + autodl-proxy, ship branch via `git archive`, editable install, `yolo settings datasets_dir/runs_dir`, caches on data disk, full kitti-stereo). Collect the 8-row table vs the 34.3/4.2 baseline.

---

## Self-Review

- **Spec coverage:** Lever 1 (head T1, target T2, loss T3, decode T4); Lever 2 (head σ T1, NLL T5, decode knobs T6); flag gating (T1 train flags, T4/T6 decode flags); 4-run→8-arm harness (T7); tests (round-trips T2/T6, baseline-unchanged T1). KITTI centroid subtlety → T2 round-trip. All covered.
- **Placeholders:** none — every code step shows code; the only deferred item is the weste full run (T7 step 6), which is execution, not code.
- **Type consistency:** `enable_proj_center`/`enable_depth_uncertainty` (T1) used verbatim in T3 test; `encode_proj_offset` signature (T2) matches its test; `laplacian_nll` (T5) matches; decode params `use_proj_center`/`ivw_fusion`/`score_weight`/`score_k` consistent across T4/T6/T7.
- **Open flag to watch:** T7 step 4 (custom val args reaching `self.args`) — the one integration risk; verified in the smoke test before the full run.
