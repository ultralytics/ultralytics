# S3D Cross-Domain Benchmark Portfolio — Design

**Date:** 2026-07-24
**Branch context:** `001-stereo-centernet-gaps`
**Status:** Design approved, pending spec review

## Problem

The stereo-3D (s3d) task is researched almost entirely on KITTI, with Virtual KITTI 2 as the
only second benchmark and the client `cube_s3d` set as an out-of-distribution curiosity. To
measure how the stereo-CenterNet model behaves off-distribution, we need a **portfolio of
standalone cross-domain benchmarks** — each converted to the existing KITTI-stereo format and
evaluated independently (VKITTI2-style), not pooled into one training set.

Three domain axes are in scope:

1. **Real automotive, new rig** — different stereo baseline / resolution / fleet than KITTI.
2. **Non-automotive / indoor** — a scene type far from street driving.
3. **New object categories** — classes beyond Car/Van/Truck.

## Non-goals

- Multi-dataset joint training or domain-adaptation methods (explicitly deferred; goal is
  "just more benchmarks").
- Any dataset behind a login/registration/EULA wall. Cityscapes-3D and KITTI-360 are **rejected**
  on this basis (Cityscapes was already a confirmed no-login dead-end; KITTI-360 is registration +
  GDPR-gated and, at baseline 0.60 m from the same lab, only marginally "cross-domain").
- Monocular or RGB-D datasets. nuScenes, Waymo, SUN RGB-D, ScanNet, Objectron, GraspNet are
  ruled out — they lack a genuine calibrated rectified stereo pair. DrivingStereo and DSEC have
  stereo but only depth / 2D-box GT, not 3D boxes.

## Target format (fixed — all converters must emit exactly this)

Established by `ultralytics/data/scripts/convert_kitti_3d.py` and `cfg/datasets/kitti-stereo.yaml`.

- **Layout:** `images/{train,val}/{left,right}/`, `labels/{train,val}/*.txt`,
  `calib/{train,val}/*.txt`, `dataset.yaml`, `train.txt`/`val.txt`.
- **Label (18 values):**
  `class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y truncated occluded`
  — 2D boxes normalized to [0,1]; dims in meters (L, W, H); location = camera-frame bottom-center;
  `rot_y` in radians.
- **Calib (simplified):** `fx fy cx cy right_cx right_cy baseline image_width image_height`.
- **`dataset.yaml` must carry:** `names`, per-class `mean_dims` and `std_dims` ([L,W,H], meters),
  `baseline`, `channels: 6`, and **`depth_min` / `depth_max`**.

### Load-bearing invariant: per-dataset depth range

`depth_min`/`depth_max` **must** be set from each dataset's actual object-depth distribution.
The KITTI default (2–80 m) silently clamps any nearer/farther domain to a constant and drives
AP3D to 0 — this is the exact `cube_s3d` failure (cubes at 0.45–1.66 m clamped to the 2 m floor).
Each converter computes the empirical depth percentiles from its train split and writes them; the
converter asserts `depth_min < min(loc_z) and depth_max > max(loc_z)` on the produced labels.

### Second invariant: `std_dims > 0`

For (near-)constant-dimension classes the `(dim-mean)/std` normalization divides by zero
(the `cube_s3d` constant-cube trap). Converters clamp each computed `std` to a small positive
floor (≥ 0.01 m).

## Architecture

**Every converter subclasses `KITTIToYOLO3D`**, overriding only source-specific parsing. The base
class already owns everything that must stay identical across benchmarks: the 18-value writer,
`compute_right_box` (disparity → right-image center + corner-projected width), the simplified-calib
writer, mean/std-dims computation, `dataset.yaml` emission, and the 3DOP-style split plumbing.
VKITTI2 already validated this subclassing pattern.

Per dataset, exactly two new artifacts:

- `ultralytics/data/scripts/convert_<name>_3d.py` — subclass implementing: source calib parsing,
  source label parsing, and any source-frame → left-camera-frame transform.
- `ultralytics/cfg/datasets/<name>-s3d.yaml` — classes + priors + baseline + depth range.

**Rejected alternative:** a single mega-converter with `--source` flags. It would accrete the same
branch/key-mismatch tangle the repo's Core Principles warn against, and is harder to unit-test than
small focused subclasses. Separate subclasses match the existing VKITTI2 precedent.

Each converter overrides:

| Hook | What it does |
|---|---|
| `parse_calibration(file)` | Return `{fx, fy, cx, cy, right_cx, right_cy, baseline, image_width, image_height}` from the source calib. |
| `parse_source_label(...)` | Yield per-object `(class, box2d_left, dims_lwh, loc_xyz_cam, rot_y, trunc, occ)`. |
| frame transform | Map the source annotation frame to the **left-camera** frame before emitting `loc_xyz`/`rot_y`. |
| split definition | Scene/sequence-holdout where sequences exist (avoid near-duplicate-frame leakage — the VKITTI2 lesson). |

## The four benchmarks (implementation + review order = cheapest first)

### Phase 1 — KITTI all-classes (near-zero effort)

No new converter. Re-run `convert_kitti_3d.py` with **no** `--filter-classes`, producing Van, Truck,
Pedestrian, Person_sitting, Cyclist, Tram, Misc 3D boxes alongside Car. The 8-class
`cfg/datasets/kitti-stereo.yaml` already exists, so the deliverable is primarily **evaluating** the
existing all-class YAML (Pedestrian/Cyclist AP3D@0.5, per KITTI convention). Confirms the pipeline
handles multi-class 3D eval before we introduce new data. Covers the "new categories" axis at ~zero
cost. Rig: baseline 0.54 m, 1242×375.

### Phase 2 — SHIFT (synthetic, open direct download)

`convert_shift_3d.py`. Source: SysCV SHIFT (vis.xyz/shift), CC BY-SA 4.0, open Python-script download.
Parse SHIFT `det_3d` 9-DoF boxes (drop pitch/roll → keep yaw as `rot_y`, matching the KITTI-format
target) and the stereo intrinsics. Rig: baseline 0.50 m, 1280×800. Classes: car, truck, bus, bicycle,
motorcycle, pedestrian → new categories **bus, motorcycle, bicycle, pedestrian** on a second
synthetic rig.
**Verify at implementation:** that 3D boxes are packaged for the `left_stereo` view specifically
(front-camera reference is the likely label frame — confirm against the devkit before GPU time).

### Phase 3 — StereOBJ-1M (indoor / non-automotive, open)

`convert_stereobj_3d.py`. Source: `github.com/xingyul/stereobj-1m`, MIT, open Dropbox (no login).
Real calibrated indoor stereo (tabletop / mechanical / medical objects; 18 known-CAD instances).
Convert 6-DoF object pose + CAD dimensions → amodal 3D box (center, dims, yaw) per object; read
`camera.json` for intrinsics/baseline. This is the **only** viable non-automotive axis — no other
free indoor stereo-3D-box dataset exists (all indoor 3D-box data is RGB-D or monocular; synthetic
generation was the only alternative and is far higher effort).
**Verify at implementation:** actual baseline (m) and image resolution from `camera.json`; the
near-field depth range (drives `depth_min/max`) and small object dims (drive `mean_dims`) differ most
from KITTI here, so both invariants above are load-bearing for this dataset.

### Phase 4 — Argoverse 1 Stereo (real, new rig, open, highest effort)

`convert_argoverse_stereo_3d.py`. Source: open S3 (verified live 2026-07-24 —
`s3://argoverse/datasets/av1.1/tars/rectified_stereo_images_v1.1.tar.gz`, 13.6 GB, and the
`tracking_{train1..4,val}_v1.1.tar.gz` cuboid tars in the same open bucket). CC BY-NC-SA 4.0.
Rig: baseline 0.30 m, 2056×2464, 6,624 rectified stereo pairs across 74 of 113 tracking sequences.
Steps: (1) join the 74 stereo sequences to the AV1 3D-tracking cuboids (labels live in the tracking
release, not the stereo tarball); (2) transform ego/city-frame cuboids → left-camera frame via the
provided per-sequence extrinsics; (3) derive `rot_y` from the camera-frame yaw. Richest taxonomy —
15 classes incl. animals, strollers, mopeds, bus, trailer.
**Verify at implementation:** the stereo pair is epipolar-rectified (docs say "rectified"; confirm
before trusting `compute_right_box`'s disparity geometry).

## Evaluation protocol

Each benchmark is standalone. Recipe follows the validated VKITTI2 path — **pretrained COCO backbone
(`model.load("yolo26n.pt")`), rectangular imgsz** (square imgsz is the confirmed AP3D=0 trap),
scene/sequence-holdout val split. Metrics: per-class AP3D@0.5 and @0.7, AP_BEV@0.5, AOS (Car plus the
dataset's new classes).

Two modes per dataset:

- **(a) Zero-shot** — a KITTI-trained checkpoint evaluated directly on the new benchmark's val split.
  Measures the domain gap. (Only meaningful where classes overlap KITTI, e.g. Car/Pedestrian/Cyclist;
  disjoint classes like StereOBJ objects are from-scratch only.)
- **(b) From-scratch** — train on the benchmark's own train split, eval on its val. Measures the
  learnable ceiling for that domain.

## Testing

Per converter, a **reprojection self-check** is the correctness gate before any GPU time: project the
converted 3D box back into the left 2D image and compute IoU against the source 2D box on a handful of
frames. Floor: mean IoU ≥ 0.80 (VKITTI2 achieved 0.856). A converter that misses the floor has a
frame-transform or calibration bug and must not proceed to training.

Additional per-converter assertions (run inside the converter, cheap):

- `depth_min < min(loc_z)` and `depth_max > max(loc_z)` over produced labels.
- all `std_dims ≥ 0.01`.
- every emitted label line has exactly 18 fields; all normalized 2D coords in [0,1] (except the
  documented truncated-object exception the base class already handles).

## Deliverables

- Phase 1: `kitti-stereo` all-class evaluation (reuse existing YAML; no new code).
- Phases 2–4: one `convert_<name>_3d.py` + one `<name>-s3d.yaml` each, plus a short docs page under
  `docs/en/datasets/` per dataset (mirroring the existing `depth/vkitti2.md`).
- A results table (per dataset: zero-shot vs from-scratch AP3D) accumulated as phases land.

## Open risks / verification debt

Carried into implementation (each blocks its own phase, not the others):

1. SHIFT — 3D-box packaging for the stereo view unconfirmed.
2. StereOBJ-1M — baseline + resolution unconfirmed (read `camera.json`).
3. Argoverse 1 — stereo rectification (epipolar alignment) unconfirmed; tracking↔stereo sequence
   join mapping needs to be established from the dev kit.
