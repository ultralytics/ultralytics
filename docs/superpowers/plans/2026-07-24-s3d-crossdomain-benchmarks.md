# S3D Cross-Domain Benchmark Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four standalone cross-domain stereo-3D benchmarks (KITTI all-classes, SHIFT, StereOBJ-1M, Argoverse 1 Stereo), each converted to the existing 18-value KITTI-stereo format and evaluated independently.

**Architecture:** Every converter subclasses the existing `KITTIToYOLO3D` (`ultralytics/data/scripts/convert_kitti_3d.py`), overriding only source-specific calib parsing, label parsing, and source-frame→left-camera transform. A new shared reprojection self-check utility gates every converter's correctness before any GPU time. Phases land cheapest-first with a review gate between each.

**Tech Stack:** Python, NumPy, Ultralytics YOLO s3d task (`ultralytics/models/yolo/s3d/`), pytest. Dataset dev kits: SHIFT `shift-dev`, StereOBJ `stereobj-1m`, Argoverse `argoverse-api` (v1).

## Global Constraints

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` (Actions bot adds it; do not hand-edit headers).
- Ruff format, line length 120, Google-style docstrings. Run `ruff format . && ruff check --fix .` before every commit.
- **NEVER push to main. NEVER force push.** This work runs in a git worktree on a feature branch off `001-stereo-centernet-gaps`; open a PR.
- Target label format is FIXED — 18 values: `class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y truncated occluded`. 2D coords normalized [0,1]; dims meters (L,W,H); `loc` = camera-frame bottom-center; `rot_y` radians.
- Simplified calib file lines: `fx fy cx cy right_cx right_cy baseline image_width image_height`.
- Every `<name>-s3d.yaml` MUST set `depth_min`/`depth_max` from the dataset's real depth range (the KITTI 2–80 m default drives AP3D=0 off-domain — the cube_s3d trap). The head validates `0 < depth_min < depth_max` (`ultralytics/models/yolo/s3d/head.py:37`); `train.py:160-161` reads them from the data cfg.
- Every `std_dims` value ≥ 0.01 m (the constant-dimension divide-by-zero trap).
- Reprojection self-check floor: mean left-box IoU ≥ 0.80 (VKITTI2 achieved 0.856). A converter below the floor has a frame/calib bug and must not train.
- Eval recipe: pretrained COCO backbone (`model.load("yolo26n.pt")`), **rectangular imgsz** (square imgsz is the confirmed AP3D=0 trap), scene/sequence-holdout val split.
- Deps like `argoverse-api`, `shift-dev` are converter-only — import them lazily inside the converter, never at package import time. CI must not gain a new hard dependency.

---

## File Structure

- Create: `ultralytics/data/scripts/s3d_reproject_check.py` — shared correctness gate (reproject converted 3D box → left 2D, IoU vs stored left box).
- Create: `tests/test_s3d_convert.py` — unit tests for the reproject utility and per-converter fixtures.
- Create: `ultralytics/cfg/datasets/kitti-stereo-allclasses.yaml` — only if the existing 8-class `kitti-stereo.yaml` needs a distinct all-class eval config (see Task 2).
- Create: `ultralytics/data/scripts/convert_shift_3d.py` + `ultralytics/cfg/datasets/shift-s3d.yaml`.
- Create: `ultralytics/data/scripts/convert_stereobj_3d.py` + `ultralytics/cfg/datasets/stereobj-s3d.yaml`.
- Create: `ultralytics/data/scripts/convert_argoverse_stereo_3d.py` + `ultralytics/cfg/datasets/argoverse-stereo-s3d.yaml`.
- Create: `docs/en/datasets/depth/{shift,stereobj,argoverse-stereo}.md` (mirror existing `docs/en/datasets/depth/vkitti2.md`).
- Reuse (do not modify): `ultralytics/data/scripts/convert_kitti_3d.py` (`KITTIToYOLO3D`).

---

## Phase 0 — Shared reprojection self-check utility

### Task 1: Reprojection self-check utility + test

**Files:**
- Create: `ultralytics/data/scripts/s3d_reproject_check.py`
- Test: `tests/test_s3d_convert.py`

**Interfaces:**
- Produces: `reproject_box_2d(loc_xyz, dims_lwh, rot_y, calib) -> (x1, y1, x2, y2)` (left-image pixels); `box_iou_xyxy(a, b) -> float`; `check_split(labels_dir, calib_dir, img_w, img_h, sample=50) -> float` (mean IoU of converted left box vs reprojected 3D box over up to `sample` labels).

- [ ] **Step 1: Write the failing test** (synthetic KITTI-like object — no external data needed)

```python
# tests/test_s3d_convert.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import numpy as np
from ultralytics.data.scripts.s3d_reproject_check import reproject_box_2d, box_iou_xyxy


def test_reproject_box_matches_manual_projection():
    # A 3.9x1.6x1.5 m (L,W,H) box, 15 m ahead, no rotation, KITTI-ish calib.
    calib = {"fx": 721.5, "fy": 721.5, "cx": 610.0, "cy": 172.0}
    x1, y1, x2, y2 = reproject_box_2d(
        loc_xyz=(0.0, 1.65, 15.0), dims_lwh=(3.9, 1.6, 1.5), rot_y=0.0, calib=calib
    )
    # Box straddles the principal ray, so it is centered near cx; width > height in px.
    assert x1 < 610.0 < x2, f"box should straddle cx: {(x1, x2)}"
    assert (x2 - x1) > (y2 - y1), "car is wider than tall in image space at this range"
    # Self-consistency: IoU of a box with itself is 1.0
    assert abs(box_iou_xyxy((x1, y1, x2, y2), (x1, y1, x2, y2)) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_s3d_convert.py::test_reproject_box_matches_manual_projection -v`
Expected: FAIL — `ModuleNotFoundError: ultralytics.data.scripts.s3d_reproject_check`

- [ ] **Step 3: Write minimal implementation**

```python
# ultralytics/data/scripts/s3d_reproject_check.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared correctness gate for s3d dataset converters.

Projects a converted 3D box (camera-frame bottom-center location, L/W/H dims, rot_y)
back into the left image and compares against the stored left 2D box. A converter with
a correct frame transform and calibration reprojects to high IoU; a low mean IoU means
the transform or calib is wrong. Floor: mean IoU >= 0.80 (VKITTI2 baseline 0.856).
"""

from pathlib import Path

import numpy as np


def reproject_box_2d(loc_xyz, dims_lwh, rot_y, calib):
    """Project a 3D box into the left image, returning its axis-aligned 2D bounds (pixels).

    Args:
        loc_xyz (tuple): (X, Y, Z) camera-frame bottom-center location, meters.
        dims_lwh (tuple): (length, width, height), meters.
        rot_y (float): rotation about the camera Y axis, radians.
        calib (dict): must contain fx, fy, cx, cy.

    Returns:
        (tuple): (x1, y1, x2, y2) in left-image pixels.
    """
    x, y, z = loc_xyz
    l, w, h = dims_lwh
    corners = np.array(
        [
            [-l / 2, 0, -w / 2], [l / 2, 0, -w / 2], [l / 2, 0, w / 2], [-l / 2, 0, w / 2],
            [-l / 2, -h, -w / 2], [l / 2, -h, -w / 2], [l / 2, -h, w / 2], [-l / 2, -h, w / 2],
        ]
    )
    c, s = np.cos(rot_y), np.sin(rot_y)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    cam = corners @ r.T + np.array([x, y, z])
    cam = cam[cam[:, 2] > 1e-6]
    if len(cam) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    u = calib["fx"] * cam[:, 0] / cam[:, 2] + calib["cx"]
    v = calib["fy"] * cam[:, 1] / cam[:, 2] + calib["cy"]
    return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


def box_iou_xyxy(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_s3d_convert.py::test_reproject_box_matches_manual_projection -v`
Expected: PASS

- [ ] **Step 5: Add the split-level checker + its test**

```python
# append to tests/test_s3d_convert.py
def test_check_split_perfect_on_selfconsistent_fixture(tmp_path):
    from ultralytics.data.scripts.s3d_reproject_check import check_split, reproject_box_2d
    calib = {"fx": 721.5, "fy": 721.5, "cx": 610.0, "cy": 172.0}
    img_w, img_h = 1242, 375
    (tmp_path / "labels").mkdir(); (tmp_path / "calib").mkdir()
    x1, y1, x2, y2 = reproject_box_2d((0.0, 1.65, 15.0), (3.9, 1.6, 1.5), 0.0, calib)
    lb = [
        "0",
        f"{(x1 + x2) / 2 / img_w:.6f}", f"{(y1 + y2) / 2 / img_h:.6f}",
        f"{(x2 - x1) / img_w:.6f}", f"{(y2 - y1) / img_h:.6f}",
        "0 0 0 0",                       # right box unused by the left-reprojection check
        "3.9 1.6 1.5", "0.0 1.65 15.0", "0.0", "0 0",
    ]
    (tmp_path / "labels" / "000000.txt").write_text(" ".join(lb) + "\n")
    (tmp_path / "calib" / "000000.txt").write_text(
        "fx: 721.5\nfy: 721.5\ncx: 610.0\ncy: 172.0\n"
        "right_cx: 610.0\nright_cy: 172.0\nbaseline: 0.54\n"
        f"image_width: {img_w}\nimage_height: {img_h}\n"
    )
    assert check_split(tmp_path / "labels", tmp_path / "calib", img_w, img_h) > 0.999
```

```python
# append to ultralytics/data/scripts/s3d_reproject_check.py
def _read_calib(path):
    """Parse the simplified s3d calib file into a dict of floats."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = float(v)
    return out


def check_split(labels_dir, calib_dir, img_w, img_h, sample=50):
    """Mean IoU between each converted left 2D box and its reprojected 3D box.

    Args:
        labels_dir (Path): converted labels dir (18-value files).
        calib_dir (Path): matching simplified calib dir.
        img_w (int): image width (pixels) for de-normalizing the stored left box.
        img_h (int): image height (pixels).
        sample (int): cap on labels scanned.

    Returns:
        (float): mean IoU over all objects scanned (0.0 if none).
    """
    labels_dir, calib_dir = Path(labels_dir), Path(calib_dir)
    ious = []
    for lf in sorted(labels_dir.glob("*.txt"))[:sample]:
        cf = calib_dir / f"{lf.stem}.txt"
        if not cf.exists():
            continue
        calib = _read_calib(cf)
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 16:
                continue
            xc, yc, bw, bh = (float(p[i]) for i in (1, 2, 3, 4))
            stored = ((xc - bw / 2) * img_w, (yc - bh / 2) * img_h, (xc + bw / 2) * img_w, (yc + bh / 2) * img_h)
            loc = (float(p[12]), float(p[13]), float(p[14]))
            dims = (float(p[9]), float(p[10]), float(p[11]))
            proj = reproject_box_2d(loc, dims, float(p[15]), calib)
            ious.append(box_iou_xyxy(stored, proj))
    return float(np.mean(ious)) if ious else 0.0
```

- [ ] **Step 6: Run both tests + lint**

Run: `pytest tests/test_s3d_convert.py -v && ruff format ultralytics/data/scripts/s3d_reproject_check.py tests/test_s3d_convert.py && ruff check --fix ultralytics/data/scripts/s3d_reproject_check.py`
Expected: 2 passed; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add ultralytics/data/scripts/s3d_reproject_check.py tests/test_s3d_convert.py
git commit -m "feat(s3d): shared reprojection self-check for dataset converters"
```

---

## Phase 1 — KITTI all-classes benchmark (near-zero effort)

### Task 2: Produce and validate the all-class KITTI-stereo benchmark

**Files:**
- Modify/verify: `ultralytics/cfg/datasets/kitti-stereo.yaml` (already 8-class — confirm it needs no change)
- Create (only if a distinct eval config is wanted): `ultralytics/cfg/datasets/kitti-stereo-allclasses.yaml`

**Interfaces:**
- Consumes: `convert_kitti_3d.py` `KITTIToYOLO3D` (run with no `--filter-classes`); `check_split` from Task 1.
- Produces: a converted KITTI dir carrying all 8 classes' 3D labels; a validated dataset YAML usable by `yolo val task=s3d`.

- [ ] **Step 1: Convert KITTI with all classes** (on the GPU box that already stages KITTI — see memory `autodl-fs-datasets`)

Run: `python ultralytics/data/scripts/convert_kitti_3d.py --kitti-root <KITTI_ROOT>`
Expected: log reports non-zero mean_dims for Pedestrian and Cyclist (not just Car); `labels/train` files contain class ids 3 and 5.

- [ ] **Step 2: Verify the reprojection floor on the converted split**

```bash
python -c "
from ultralytics.data.scripts.s3d_reproject_check import check_split
m = check_split('<KITTI_ROOT>/labels/train', '<KITTI_ROOT>/calib/train', 1242, 375)
print('mean IoU', m); assert m >= 0.80, m"
```
Expected: prints mean IoU ≥ 0.80.

- [ ] **Step 3: Confirm the YAML carries all classes + correct KITTI depth range**

Read `ultralytics/cfg/datasets/kitti-stereo.yaml`; confirm `names` has all 8 classes (it does), `depth_min: 2.0`, `depth_max: 80.0`, and per-class `mean_dims`/`std_dims` for Pedestrian/Cyclist exist (they do). No new file is needed unless you want an eval-only alias — in that case copy it to `kitti-stereo-allclasses.yaml` unchanged and note in a comment it exists only to name the all-class eval.

- [ ] **Step 4: Run multi-class eval with a KITTI-trained checkpoint**

Run: `yolo val task=s3d model=<kitti_n_A5full_pre1000/weights/best.pt> data=kitti-stereo.yaml imgsz=[384,1248]`
Expected: AP3D reported per class; Pedestrian and Cyclist rows present and non-zero (Cyclist/Pedestrian evaluated @0.5 per KITTI convention). Record Car vs Pedestrian vs Cyclist AP3D.

- [ ] **Step 5: Commit** (only if a new alias YAML or doc note was added)

```bash
git add ultralytics/cfg/datasets/kitti-stereo-allclasses.yaml
git commit -m "docs(s3d): all-class KITTI-stereo eval alias"
```

> **REVIEW GATE 1** — stop here for review before starting Phase 2.

---

## Phase 2 — SHIFT (synthetic, new categories)

### Task 3: Pin the SHIFT schema from a real sample

**Files:**
- Create (scaffold + recorded schema in docstring): `ultralytics/data/scripts/convert_shift_3d.py`

- [ ] **Step 1: Download a discrete-set SHIFT sample with the stereo view + 3D boxes**

Use the official downloader (front + `left_stereo` image views and the `det_3d` label). On the GPU box:
```bash
python -m pip install shift-dev  # converter-only dep; do NOT add to package deps
# follow shift-dev README to pull: views=[front,left_stereo], group=[img,det_3d], split=minival
```
Expected: a small `discrete/images/minival/.../{front,left_stereo}/*.jpg` tree + `det_3d.json`.

- [ ] **Step 2: Inspect and RECORD the exact schema** into the converter docstring — do not guess. Confirm:
  - Which view holds the `det_3d` labels (front vs left_stereo) and whether both left+right stereo frames exist per sample.
  - Field names/order of each 3D box: center (x,y,z) frame + units, dims order, rotation (euler vs quaternion; which axis is yaw).
  - Intrinsics location and the stereo baseline (SHIFT config says 0.5 m — confirm) and resolution (1280×800 — confirm).
  - The camera frame convention (SHIFT uses a right-handed frame; determine the rotation to KITTI camera frame: +x right, +y down, +z forward).

Write the confirmed answers as a `SCHEMA:` block in the module docstring. This block is the contract Task 4 codes against.

- [ ] **Step 3: Commit the scaffold + recorded schema**

```bash
git add ultralytics/data/scripts/convert_shift_3d.py
git commit -m "chore(s3d): record verified SHIFT det_3d schema for converter"
```

### Task 4: SHIFT converter subclass + self-check

**Files:**
- Modify: `ultralytics/data/scripts/convert_shift_3d.py`
- Create: `ultralytics/cfg/datasets/shift-s3d.yaml`
- Test: `tests/test_s3d_convert.py`

**Interfaces:**
- Consumes: `KITTIToYOLO3D` (base); `check_split` (Task 1); the `SCHEMA:` block (Task 3).
- Produces: `ShiftToYOLO3D(KITTIToYOLO3D)` overriding `parse_calibration`, `parse_source_label`, and a `_to_kitti_cam(center, rot)` transform; a `shift-s3d.yaml`.

- [ ] **Step 1: Write the failing transform test** (uses the recorded frame convention; values below are the KITTI target frame — replace the rotation in Step 3 with the one Task 3 recorded)

```python
# append to tests/test_s3d_convert.py
def test_shift_frame_transform_to_kitti_cam():
    from ultralytics.data.scripts.convert_shift_3d import ShiftToYOLO3D
    conv = ShiftToYOLO3D.__new__(ShiftToYOLO3D)  # no I/O
    # A point 10 m in front, 2 m right, 1 m below the SHIFT camera → KITTI cam frame.
    loc, ry = conv._to_kitti_cam(center_shift=(2.0, -1.0, 10.0), yaw_shift=0.0)
    assert loc[2] > 0, "Z (depth) must be positive/forward in KITTI cam frame"
    assert abs(loc[0] - 2.0) < 1e-6 and abs(loc[2] - 10.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_s3d_convert.py::test_shift_frame_transform_to_kitti_cam -v`
Expected: FAIL — module/class not defined.

- [ ] **Step 3: Implement the subclass** (fill `parse_source_label`/`parse_calibration` bodies from the Task 3 `SCHEMA:` block; the structure below is fixed)

```python
# ultralytics/data/scripts/convert_shift_3d.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert SHIFT (SysCV) stereo + det_3d to the YOLO 18-value stereo-3D format.

SCHEMA: <paste the verified schema recorded in Task 3 here>
"""

import numpy as np

from ultralytics.data.scripts.convert_kitti_3d import KITTIToYOLO3D

# SHIFT category name -> contiguous class id (order defines the YAML `names`).
SHIFT_CLASSES = {"pedestrian": 0, "car": 1, "truck": 2, "bus": 3, "motorcycle": 4, "bicycle": 5}


class ShiftToYOLO3D(KITTIToYOLO3D):
    """SHIFT → YOLO stereo-3D converter (see module SCHEMA for field layout)."""

    def __init__(self, shift_root, output_root, **kw):
        super().__init__(shift_root, output_root, **kw)
        self.class_map = dict(SHIFT_CLASSES)
        self.img_width, self.img_height = 1280, 800  # confirm from Task 3

    def _to_kitti_cam(self, center_shift, yaw_shift):
        """Map a SHIFT-camera-frame center+yaw to the KITTI camera frame (+x right,+y down,+z fwd).

        Returns:
            (tuple, float): ((X, Y, Z) meters, rot_y radians). REPLACE the identity below
            with the exact axis permutation/sign flips recorded in Task 3.
        """
        x, y, z = center_shift  # TODO(Task3): apply recorded permutation/signs
        return (float(x), float(y), float(z)), float(yaw_shift)

    def parse_calibration(self, calib_source):
        """Return the simplified-calib dict from SHIFT intrinsics (see SCHEMA)."""
        raise NotImplementedError("fill from Task 3 SCHEMA: fx,fy,cx,cy,baseline=0.5,right_cx,right_cy")

    def parse_source_label(self, det_3d_record):
        """Yield (class_id, box2d_left, dims_lwh, loc_xyz_cam, rot_y, trunc, occ) per object."""
        raise NotImplementedError("fill from Task 3 SCHEMA")
```

Replace the `_to_kitti_cam` body so the test's assertions hold (X→right, Z→forward). If SHIFT's frame already matches, the identity passes; if not, apply the recorded permutation.

- [ ] **Step 4: Run the transform test**

Run: `pytest tests/test_s3d_convert.py::test_shift_frame_transform_to_kitti_cam -v`
Expected: PASS

- [ ] **Step 5: Wire the base-class emission path**

Implement `parse_calibration` + `parse_source_label` per SCHEMA, then add a `convert_split()` that (like `KITTIToYOLO3D.convert_split`) iterates frames, calls the base label/calib writers, copies left+right images, and computes mean/std dims. Reuse base helpers — do not reimplement the 18-value writer.

- [ ] **Step 6: Run the converter on the minival sample + verify reprojection floor**

```bash
python ultralytics/data/scripts/convert_shift_3d.py --shift-root <SHIFT_SAMPLE> --output-root <OUT>
python -c "
from ultralytics.data.scripts.s3d_reproject_check import check_split
m = check_split('<OUT>/labels/val', '<OUT>/calib/val', 1280, 800); print('mean IoU', m); assert m >= 0.80, m"
```
Expected: mean IoU ≥ 0.80. If below, the `_to_kitti_cam` transform or intrinsics are wrong — fix before proceeding (systematic-debugging).

- [ ] **Step 7: Write `shift-s3d.yaml`** with confirmed depth range + per-class priors

```yaml
# ultralytics/cfg/datasets/shift-s3d.yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
path: shift-s3d
train: images/train/left
val: images/val/left
train_right: images/train/right
val_right: images/val/right
names: {0: pedestrian, 1: car, 2: truck, 3: bus, 4: motorcycle, 5: bicycle}
baseline: 0.50
depth_min: 1.0    # REPLACE with 1st-percentile loc_z from the converter log
depth_max: 80.0   # REPLACE with 99th-percentile loc_z from the converter log
channels: 6
# mean_dims / std_dims: paste the converter-computed values (std floored at 0.01)
```

- [ ] **Step 8: Lint + commit**

```bash
ruff format ultralytics/data/scripts/convert_shift_3d.py && ruff check --fix ultralytics/data/scripts/convert_shift_3d.py
git add ultralytics/data/scripts/convert_shift_3d.py ultralytics/cfg/datasets/shift-s3d.yaml tests/test_s3d_convert.py
git commit -m "feat(s3d): SHIFT stereo-3D converter + dataset yaml"
```

- [ ] **Step 9: Full SHIFT convert + from-scratch benchmark** (GPU box)

Convert the full discrete set, then: `python train_pretrained.py` equivalent (`model.load("yolo26n.pt")`, rect imgsz matching 1280×800 aspect, scene-holdout val). Record per-class AP3D@0.5/0.7, BEV, AOS. Also run zero-shot: KITTI checkpoint → `yolo val data=shift-s3d.yaml` for the car/pedestrian overlap.

- [ ] **Step 10: Docs page** — create `docs/en/datasets/depth/shift.md` mirroring `vkitti2.md`; then `python docs/build_reference.py` if any public API changed (the converter script is not a public API, so likely not needed). Format: `npx prettier@3.6.2 --tab-width 4 --print-width 120 --write docs/en/datasets/depth/shift.md`. Commit.

> **REVIEW GATE 2** — stop for review before Phase 3.

---

## Phase 3 — StereOBJ-1M (indoor / non-automotive)

### Task 5: Pin the StereOBJ-1M schema from a real sample

**Files:**
- Create (scaffold + recorded schema): `ultralytics/data/scripts/convert_stereobj_3d.py`

- [ ] **Step 1: Fetch a small StereOBJ-1M scene** from the open Dropbox (`github.com/xingyul/stereobj-1m` README; no login). Grab one scene's left+right frames, its `camera.json`, and the pose/label json.

- [ ] **Step 2: Inspect and RECORD** into the docstring `SCHEMA:` block: exact `camera.json` keys (fx,fy,cx,cy per camera; baseline in meters + resolution — both unconfirmed, must read here); per-object 6-DoF pose representation (rotation matrix vs quaternion; translation units); the CAD dimension source (per-object L/W/H, meters) and object→class-id mapping for the 18 objects; the camera-frame convention. Amodal 3D box = pose translation is the object center; convert center→bottom-center by subtracting H/2 along the camera down-axis; yaw = the pose rotation projected to rot_y about the camera Y axis.

- [ ] **Step 3: Commit scaffold + schema**

```bash
git add ultralytics/data/scripts/convert_stereobj_3d.py
git commit -m "chore(s3d): record verified StereOBJ-1M schema for converter"
```

### Task 6: StereOBJ-1M converter subclass + self-check

**Files:**
- Modify: `ultralytics/data/scripts/convert_stereobj_3d.py`
- Create: `ultralytics/cfg/datasets/stereobj-s3d.yaml`
- Test: `tests/test_s3d_convert.py`

**Interfaces:**
- Consumes: `KITTIToYOLO3D`; `check_split`; the recorded SCHEMA.
- Produces: `StereObjToYOLO3D(KITTIToYOLO3D)` with `parse_calibration`, `parse_source_label`, and `_pose_to_box(pose, cad_dims)` → `(loc_xyz_bottom_center, dims_lwh, rot_y)`.

- [ ] **Step 1: Write the failing pose→box test** (identity pose → center at translation, bottom-center below by H/2)

```python
# append to tests/test_s3d_convert.py
def test_stereobj_pose_to_bottom_center():
    import numpy as np
    from ultralytics.data.scripts.convert_stereobj_3d import StereObjToYOLO3D
    conv = StereObjToYOLO3D.__new__(StereObjToYOLO3D)
    R = np.eye(3); t = np.array([0.1, 0.0, 0.6])  # 0.6 m in front
    loc, dims, ry = conv._pose_to_box((R, t), cad_dims=(0.05, 0.05, 0.10))
    assert abs(loc[2] - 0.6) < 1e-6
    assert abs(loc[1] - (0.0 + 0.10 / 2)) < 1e-6, "bottom-center is +H/2 along cam down-axis"
    assert abs(ry) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_s3d_convert.py::test_stereobj_pose_to_bottom_center -v`
Expected: FAIL — module/class not defined.

- [ ] **Step 3: Implement `_pose_to_box`** (and class scaffold) so the test passes

```python
# ultralytics/data/scripts/convert_stereobj_3d.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert StereOBJ-1M (indoor stereo, 6-DoF pose) to the YOLO 18-value stereo-3D format.

SCHEMA: <paste verified schema from Task 5>
"""

import numpy as np

from ultralytics.data.scripts.convert_kitti_3d import KITTIToYOLO3D


class StereObjToYOLO3D(KITTIToYOLO3D):
    """StereOBJ-1M → YOLO stereo-3D (amodal box from 6-DoF pose + CAD dims)."""

    def _pose_to_box(self, pose, cad_dims):
        """Convert (R, t) object pose + CAD L/W/H into (loc_bottom_center, dims_lwh, rot_y).

        Bottom-center = center + H/2 along the camera down-axis (+y in KITTI cam frame).
        rot_y = yaw of R about the camera Y axis.
        """
        r, t = pose
        l, w, h = cad_dims
        loc = (float(t[0]), float(t[1] + h / 2.0), float(t[2]))
        rot_y = float(np.arctan2(r[0, 2], r[2, 2]))  # yaw about cam-Y; confirm axis vs SCHEMA
        return loc, (float(l), float(w), float(h)), rot_y

    def parse_calibration(self, camera_json):
        """Return simplified-calib dict from camera.json (see SCHEMA for keys + baseline)."""
        raise NotImplementedError("fill from Task 5 SCHEMA")

    def parse_source_label(self, scene_record):
        """Yield (class_id, box2d_left, dims_lwh, loc_xyz_cam, rot_y, trunc, occ) per object."""
        raise NotImplementedError("fill from Task 5 SCHEMA")
```

- [ ] **Step 4: Run the pose test**

Run: `pytest tests/test_s3d_convert.py::test_stereobj_pose_to_bottom_center -v`
Expected: PASS

- [ ] **Step 5: Implement calib + label parsing + `convert_split`** per SCHEMA (2D left box = projected 3D-box bounds via `reproject_box_2d`, since StereOBJ ships pose not 2D boxes — import it from `s3d_reproject_check`). Reuse base writers.

- [ ] **Step 6: Convert the sample scene + verify reprojection floor**

```bash
python ultralytics/data/scripts/convert_stereobj_3d.py --stereobj-root <SCENE> --output-root <OUT>
python -c "
from ultralytics.data.scripts.s3d_reproject_check import check_split
m = check_split('<OUT>/labels/val', '<OUT>/calib/val', <W>, <H>); print('mean IoU', m); assert m >= 0.80, m"
```
Expected: mean IoU ≥ 0.80 (note: since the 2D box is itself derived by reprojection here, this mainly validates dims/pose self-consistency and calib — a low value means the pose→box or intrinsics are wrong).

- [ ] **Step 7: Write `stereobj-s3d.yaml`** — indoor depth range is small (objects ~0.3–1.5 m): set `depth_min`/`depth_max` from the converter's percentile log (NOT the KITTI default), `std_dims` floored at 0.01, `baseline` from `camera.json`. Classes = the 18 objects (or a grouped subset — decide from SCHEMA and note the choice in a comment).

- [ ] **Step 8: Lint + commit**

```bash
ruff format ultralytics/data/scripts/convert_stereobj_3d.py && ruff check --fix ultralytics/data/scripts/convert_stereobj_3d.py
git add ultralytics/data/scripts/convert_stereobj_3d.py ultralytics/cfg/datasets/stereobj-s3d.yaml tests/test_s3d_convert.py
git commit -m "feat(s3d): StereOBJ-1M stereo-3D converter + dataset yaml"
```

- [ ] **Step 9: Full convert + from-scratch benchmark** (from-scratch only — classes disjoint from KITTI so no zero-shot). Scene-holdout val. Record AP3D. Expect this to stress the depth-range + dims-prior invariants hardest.

- [ ] **Step 10: Docs page** `docs/en/datasets/depth/stereobj.md`; prettier; commit.

> **REVIEW GATE 3** — stop for review before Phase 4.

---

## Phase 4 — Argoverse 1 Stereo (real, new rig, highest effort)

### Task 7: Pin the Argoverse stereo↔tracking join + frame transform

**Files:**
- Create (scaffold + recorded schema): `ultralytics/data/scripts/convert_argoverse_stereo_3d.py`

- [ ] **Step 1: Download the stereo tar + one tracking log** (verified-live open S3, no login)

```bash
aria2c -x8 https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/rectified_stereo_images_v1.1.tar.gz
aria2c -x8 https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/tracking_val_v1.1.tar.gz
python -m pip install argoverse-api  # converter-only dep
```
Expected: `rectified_stereo` tree with `stereo_front_left`/`stereo_front_right` per log; tracking logs with `per_sweep_annotations`/`poses`/`calibration`.

- [ ] **Step 2: Inspect and RECORD** into the docstring SCHEMA: how a stereo log id maps to its tracking log (same log guid — confirm the 74-of-113 subset); the rectified-stereo calibration (fx,fy,cx,cy + the 0.30 m baseline; confirm which json); the cuboid representation (center in city/ego frame + quaternion) and the ego→camera extrinsic chain to get boxes into the `stereo_front_left` frame; the class-name list. Amodal box: transform cuboid center to left-cam frame, convert to bottom-center (+H/2 cam-down), rot_y from the cuboid yaw in cam frame.

- [ ] **Step 3: Commit scaffold + schema**

```bash
git add ultralytics/data/scripts/convert_argoverse_stereo_3d.py
git commit -m "chore(s3d): record verified Argoverse1 stereo↔tracking schema"
```

### Task 8: Argoverse converter subclass + self-check

**Files:**
- Modify: `ultralytics/data/scripts/convert_argoverse_stereo_3d.py`
- Create: `ultralytics/cfg/datasets/argoverse-stereo-s3d.yaml`
- Test: `tests/test_s3d_convert.py`

**Interfaces:**
- Consumes: `KITTIToYOLO3D`; `check_split`; `argoverse-api`; recorded SCHEMA.
- Produces: `ArgoverseStereoToYOLO3D(KITTIToYOLO3D)` with `parse_calibration`, `_cuboid_to_cam(cuboid, ego_to_cam)` → `(loc_bottom_center, dims_lwh, rot_y)`, and a stereo↔tracking join in `convert_split`.

- [ ] **Step 1: Write the failing extrinsic-transform test** (a cuboid already in cam frame passes through; a known ego offset lands correctly)

```python
# append to tests/test_s3d_convert.py
def test_argoverse_cuboid_to_cam_identity():
    import numpy as np
    from ultralytics.data.scripts.convert_argoverse_stereo_3d import ArgoverseStereoToYOLO3D
    conv = ArgoverseStereoToYOLO3D.__new__(ArgoverseStereoToYOLO3D)
    # Cuboid centered 12 m forward in the CAMERA frame, identity ego->cam, yaw 0.
    loc, dims, ry = conv._cuboid_to_cam(
        center_cam=(0.0, 1.4, 12.0), dims_lwh=(4.0, 1.8, 1.5), yaw_cam=0.0
    )
    assert abs(loc[2] - 12.0) < 1e-6
    assert abs(loc[1] - (1.4 + 1.5 / 2)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_s3d_convert.py::test_argoverse_cuboid_to_cam_identity -v`
Expected: FAIL — module/class not defined.

- [ ] **Step 3: Implement `_cuboid_to_cam` + scaffold** so the test passes

```python
# ultralytics/data/scripts/convert_argoverse_stereo_3d.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert Argoverse 1 rectified stereo + 3D-tracking cuboids to YOLO 18-value stereo-3D.

SCHEMA: <paste verified schema from Task 7>
"""

import numpy as np

from ultralytics.data.scripts.convert_kitti_3d import KITTIToYOLO3D


class ArgoverseStereoToYOLO3D(KITTIToYOLO3D):
    """Argoverse 1 stereo → YOLO stereo-3D (join stereo frames to tracking cuboids)."""

    def _cuboid_to_cam(self, center_cam, dims_lwh, yaw_cam):
        """Camera-frame cuboid center+dims+yaw → (bottom-center loc, dims_lwh, rot_y).

        Caller applies the ego→left-cam extrinsic to get center_cam/yaw_cam first
        (see convert_split); this only does the center→bottom-center + rot_y packaging.
        """
        l, w, h = dims_lwh
        loc = (float(center_cam[0]), float(center_cam[1] + h / 2.0), float(center_cam[2]))
        return loc, (float(l), float(w), float(h)), float(yaw_cam)

    def parse_calibration(self, calib_json):
        """Return simplified-calib from the rectified-stereo calibration (baseline≈0.30)."""
        raise NotImplementedError("fill from Task 7 SCHEMA")

    def convert_split(self, split="train"):
        """Join the 74 stereo logs to tracking cuboids, transform to left-cam frame, emit."""
        raise NotImplementedError("fill from Task 7 SCHEMA: use argoverse-api loaders + extrinsics")
```

- [ ] **Step 4: Run the transform test**

Run: `pytest tests/test_s3d_convert.py::test_argoverse_cuboid_to_cam_identity -v`
Expected: PASS

- [ ] **Step 5: Implement `parse_calibration` + `convert_split`** using `argoverse-api` loaders: for each stereo log, for each stereo-timestamped frame, load the nearest cuboid annotations, apply the ego→`stereo_front_left` extrinsic (from SCHEMA) to each cuboid center + yaw, call `_cuboid_to_cam`, derive the left 2D box via `reproject_box_2d` (or the provided per-camera projection), and emit via the base writers. Sequence-holdout: reserve a fixed subset of logs for val.

- [ ] **Step 6: Convert 2–3 logs + verify reprojection floor**

```bash
python ultralytics/data/scripts/convert_argoverse_stereo_3d.py --av1-stereo-root <S> --av1-tracking-root <T> --output-root <OUT> --max-logs 3
python -c "
from ultralytics.data.scripts.s3d_reproject_check import check_split
m = check_split('<OUT>/labels/train', '<OUT>/calib/train', 2464, 2056); print('mean IoU', m); assert m >= 0.80, m"
```
Expected: mean IoU ≥ 0.80. A low value means the ego→cam extrinsic chain or the rectified intrinsics are wrong — debug the transform (this is the phase's main risk).

- [ ] **Step 7: Write `argoverse-stereo-s3d.yaml`** — `baseline: 0.30`, depth range from percentile log (Argoverse objects reach far — likely `depth_max` ~ 100+; confirm), per-class `mean_dims`/`std_dims` (std floored 0.01), full `names` list from SCHEMA.

- [ ] **Step 8: Lint + commit**

```bash
ruff format ultralytics/data/scripts/convert_argoverse_stereo_3d.py && ruff check --fix ultralytics/data/scripts/convert_argoverse_stereo_3d.py
git add ultralytics/data/scripts/convert_argoverse_stereo_3d.py ultralytics/cfg/datasets/argoverse-stereo-s3d.yaml tests/test_s3d_convert.py
git commit -m "feat(s3d): Argoverse 1 stereo-3D converter + dataset yaml"
```

- [ ] **Step 9: Full convert + benchmark** — convert all 74 stereo logs (join train/val tracking tars), sequence-holdout split. Run from-scratch + zero-shot (KITTI ckpt on the Vehicle/Pedestrian/Bicyclist overlap). Record AP3D per class.

- [ ] **Step 10: Docs page** `docs/en/datasets/depth/argoverse-stereo.md`; prettier; commit.

> **REVIEW GATE 4** — final review; assemble the portfolio results table.

---

## Task 9: Portfolio results summary

**Files:**
- Modify: `docs/en/datasets/depth/` index or a new `docs/en/datasets/depth/s3d-benchmarks.md`

- [ ] **Step 1:** Assemble a table: per dataset (KITTI-all, SHIFT, StereOBJ, Argoverse) × {zero-shot AP3D, from-scratch AP3D@0.5/0.7, BEV, AOS}. Note domain-gap deltas vs KITTI.
- [ ] **Step 2:** Format with prettier; commit. Email the summary to Rick's qqmail (memory `qqmail-notify`).

---

## Self-Review

**Spec coverage:** all four datasets (§"four benchmarks") → Tasks 2/4/6/8; shared self-check (§Testing) → Task 1; both invariants (depth range, std floor) → Global Constraints + each YAML task + Task 1 assertions; dual eval modes (§Evaluation) → Steps 9 of Phases 2/4 (from-scratch everywhere; zero-shot where classes overlap, noted disjoint for StereOBJ); docs (§Deliverables) → Steps 10 + Task 9; verification debt (§Open risks) → Tasks 3/5/7 "pin the schema" first-steps. Covered.

**Placeholder note:** The `NotImplementedError` bodies and `SCHEMA:` blocks in the converter scaffolds are **deliberate, not placeholders** — the exact external field layouts are honest verification debt (flagged in the spec), and each is pinned by a concrete inspect-and-record task (3/5/7) *before* the task that codes against it. Every step that CAN be written against known ground truth (the reproject utility, transforms, tests, YAML structure, base-class reuse) contains complete code. Do not "fill in" a converter body from memory — run its pin-the-schema task first.

**Type consistency:** `check_split`, `reproject_box_2d`, `box_iou_xyxy` signatures match across Tasks 1/2/4/6/8. Each converter subclass name (`ShiftToYOLO3D`, `StereObjToYOLO3D`, `ArgoverseStereoToYOLO3D`) is used consistently in its test and CLI.
