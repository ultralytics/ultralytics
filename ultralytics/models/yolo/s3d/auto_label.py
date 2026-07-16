# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Auto-generate auxiliary class pseudo-labels for stereo 3D detection.

When training with few classes, the depth branch can collapse because the backbone
learns spatial shortcuts. This module uses a pretrained 2D detector to find auxiliary
objects and generates pseudo-labels with two quality tiers:

- **Stereo-matched** (occ=10): L/R detections matched, depth from triangulation.
- **Mono-only** (occ=20): Left detection only, depth estimated from bbox height.

During training, the loss function weights these differently from real labels (occ 0-2),
applying reduced 3D loss weight and phasing them out in the final 10% of epochs.

Supports two modes:
- **nc=1** (default): class_offset=1, no skips. Pseudo class IDs = COCO_ID + 1.
- **nc>1** (auto_label): class_offset=nc, skip COCO classes overlapping real classes.
  Pseudo class IDs are assigned contiguously starting at nc.

Usage:
  python -m ultralytics.models.yolo.s3d.auto_label --data kitti-stereo.yaml
"""

from __future__ import annotations

from pathlib import Path
from ultralytics.utils import LOGGER

import numpy as np

# Pseudo-label quality markers (stored in the `occluded` field, index 17)
OCC_PSEUDO_STEREO = 10  # stereo-matched: triangulated depth (higher quality)
OCC_PSEUDO_MONO = 20  # mono-only: depth from bbox height (lower quality)

# All 80 COCO classes: (name, [L, W, H] in meters).
# Dimensions are approximate — only height matters for mono depth (z = fx*H/h_pixels).
# Stereo-matched pseudo-labels use triangulated depth regardless.
# fmt: off
_COCO80 = [
    ("Person",       [0.88, 0.65, 1.73]),   # 0
    ("Bicycle",      [1.72, 0.60, 1.10]),   # 1
    ("Vehicle",      [4.60, 1.80, 1.50]),   # 2 — separate from real Car class
    ("Motorcycle",   [2.00, 0.80, 1.50]),   # 3
    ("Airplane",     [15.0, 15.0, 4.00]),   # 4
    ("Bus",          [12.0, 2.55, 3.20]),   # 5
    ("Train",        [15.0, 3.00, 3.50]),   # 6
    ("Truck",        [8.00, 2.50, 3.00]),   # 7
    ("Boat",         [5.00, 2.00, 1.50]),   # 8
    ("TrafficLight", [0.30, 0.30, 0.80]),   # 9
    ("FireHydrant",  [0.30, 0.30, 0.50]),   # 10
    ("StopSign",     [0.60, 0.05, 0.60]),   # 11
    ("ParkingMeter", [0.30, 0.30, 1.20]),   # 12
    ("Bench",        [1.50, 0.50, 0.80]),   # 13
    ("Bird",         [0.30, 0.15, 0.20]),   # 14
    ("Cat",          [0.50, 0.25, 0.30]),   # 15
    ("Dog",          [0.80, 0.30, 0.50]),   # 16
    ("Horse",        [2.00, 0.50, 1.60]),   # 17
    ("Sheep",        [1.00, 0.40, 0.70]),   # 18
    ("Cow",          [2.00, 0.80, 1.40]),   # 19
    ("Elephant",     [4.00, 2.00, 3.00]),   # 20
    ("Bear",         [1.50, 0.80, 1.50]),   # 21
    ("Zebra",        [2.00, 0.50, 1.50]),   # 22
    ("Giraffe",      [2.00, 1.00, 4.50]),   # 23
    ("Backpack",     [0.35, 0.20, 0.50]),   # 24
    ("Umbrella",     [1.00, 1.00, 0.80]),   # 25
    ("Handbag",      [0.35, 0.15, 0.30]),   # 26
    ("Tie",          [0.10, 0.05, 0.40]),   # 27
    ("Suitcase",     [0.55, 0.25, 0.70]),   # 28
    ("Frisbee",      [0.30, 0.30, 0.03]),   # 29
    ("Skis",         [1.80, 0.10, 0.10]),   # 30
    ("Snowboard",    [1.50, 0.30, 0.10]),   # 31
    ("SportsBall",   [0.22, 0.22, 0.22]),   # 32
    ("Kite",         [1.00, 0.60, 0.60]),   # 33
    ("BaseballBat",  [0.80, 0.07, 0.07]),   # 34
    ("BaseballGlove",[0.30, 0.20, 0.20]),   # 35
    ("Skateboard",   [0.80, 0.20, 0.10]),   # 36
    ("Surfboard",    [2.10, 0.50, 0.10]),   # 37
    ("TennisRacket", [0.70, 0.28, 0.10]),   # 38
    ("Bottle",       [0.08, 0.08, 0.25]),   # 39
    ("WineGlass",    [0.08, 0.08, 0.20]),   # 40
    ("Cup",          [0.10, 0.10, 0.12]),   # 41
    ("Fork",         [0.20, 0.03, 0.03]),   # 42
    ("Knife",        [0.25, 0.03, 0.03]),   # 43
    ("Spoon",        [0.20, 0.04, 0.03]),   # 44
    ("Bowl",         [0.20, 0.20, 0.10]),   # 45
    ("Banana",       [0.20, 0.04, 0.04]),   # 46
    ("Apple",        [0.08, 0.08, 0.08]),   # 47
    ("Sandwich",     [0.15, 0.10, 0.08]),   # 48
    ("Orange",       [0.08, 0.08, 0.08]),   # 49
    ("Broccoli",     [0.15, 0.15, 0.20]),   # 50
    ("Carrot",       [0.20, 0.03, 0.03]),   # 51
    ("HotDog",       [0.20, 0.04, 0.04]),   # 52
    ("Pizza",        [0.35, 0.35, 0.05]),   # 53
    ("Donut",        [0.10, 0.10, 0.05]),   # 54
    ("Cake",         [0.30, 0.30, 0.15]),   # 55
    ("Chair",        [0.50, 0.50, 0.90]),   # 56
    ("Couch",        [2.00, 0.90, 0.90]),   # 57
    ("PottedPlant",  [0.40, 0.40, 0.60]),   # 58
    ("Bed",          [2.00, 1.50, 0.60]),   # 59
    ("DiningTable",  [1.50, 0.80, 0.75]),   # 60
    ("Toilet",       [0.50, 0.40, 0.50]),   # 61
    ("TV",           [1.00, 0.15, 0.60]),   # 62
    ("Laptop",       [0.35, 0.25, 0.25]),   # 63
    ("Mouse",        [0.10, 0.06, 0.04]),   # 64
    ("Remote",       [0.20, 0.05, 0.05]),   # 65
    ("Keyboard",     [0.50, 0.15, 0.05]),   # 66
    ("CellPhone",    [0.15, 0.08, 0.15]),   # 67
    ("Microwave",    [0.50, 0.35, 0.30]),   # 68
    ("Oven",         [0.60, 0.60, 0.80]),   # 69
    ("Toaster",      [0.30, 0.20, 0.20]),   # 70
    ("Sink",         [0.50, 0.40, 0.20]),   # 71
    ("Refrigerator", [0.70, 0.70, 1.80]),   # 72
    ("Book",         [0.25, 0.17, 0.25]),   # 73
    ("Clock",        [0.30, 0.08, 0.30]),   # 74
    ("Vase",         [0.15, 0.15, 0.30]),   # 75
    ("Scissors",     [0.20, 0.08, 0.10]),   # 76
    ("TeddyBear",    [0.25, 0.20, 0.35]),   # 77
    ("HairDrier",    [0.25, 0.10, 0.15]),   # 78
    ("Toothbrush",   [0.20, 0.02, 0.20]),   # 79
]
# fmt: on

MARKER_PREFIX = ".auto_labeled"  # marker file: .auto_labeled (offset=1) or .auto_labeled_offset{N}


def auto_label_stereo3d(
    label_dir: str | Path,
    left_dir: str | Path,
    right_dir: str | Path,
    calib_dir: str | Path,
    detector: str = "yolo11m.pt",
    conf: float = 0.25,
    iou: float = 0.7,
    class_offset: int = 1,
    skip_coco_ids: set[int] | None = None,
    keep_coco_ids: set[int] | None = None,
) -> int:
    """Generate pseudo-3D labels for COCO classes and append to label files.

    Runs a pretrained YOLO detector on left and right stereo images. Stereo-matched
    detections get triangulated depth (occ=10). Unmatched left detections get
    monocular depth from bbox height (occ=20). This maximizes label count.

    Args:
        label_dir: Path to label directory containing *.txt files.
        left_dir: Path to left image directory.
        right_dir: Path to right image directory.
        calib_dir: Path to calibration file directory.
        detector: Pretrained YOLO 2D model for detection.
        conf: Minimum detection confidence.
        iou: NMS IoU threshold.
        class_offset: Starting class ID for pseudo-labels. For nc=1, use 1 (default).
            For nc>1, use nc so pseudo IDs don't collide with real class IDs.
        skip_coco_ids: Set of COCO class IDs to skip (overlap with real classes).
            E.g., {0, 1, 2} skips Person/Bicycle/Car when real labels have Ped/Cyc/Car.
        keep_coco_ids: If set, ONLY these COCO class IDs are used (allowlist).
            Overrides skip_coco_ids. E.g., {3, 5, 7} keeps only Motorcycle/Bus/Truck.

    Returns:
        Number of pseudo-labels generated.
    """
    label_dir = Path(label_dir)
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    calib_dir = Path(calib_dir)

    # Skip if already processed (marker includes config to avoid cross-config collision)
    if keep_coco_ids:
        keep_str = "_keep" + "_".join(str(i) for i in sorted(keep_coco_ids))
    else:
        keep_str = ""
    marker_name = MARKER_PREFIX if class_offset == 1 else f"{MARKER_PREFIX}_offset{class_offset}{keep_str}"
    marker = label_dir / marker_name
    if marker.exists():
        n = marker.read_text().strip()
        LOGGER.info(f"Auto-label: already generated ({n} pseudo-labels, offset={class_offset}), skipping")
        return 0

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        return 0

    # Build COCO class → kitti class ID mapping (contiguous, skipping overlaps)
    skip = skip_coco_ids or set()
    coco_to_kitti = {}
    next_id = class_offset
    for i in range(len(_COCO80)):
        if keep_coco_ids and i not in keep_coco_ids:
            continue
        if i in skip:
            continue
        coco_to_kitti[i] = next_id
        next_id += 1
    n_pseudo_classes = len(coco_to_kitti)
    skipped_names = [_COCO80[i][0] for i in sorted(skip) if i < len(_COCO80)]

    LOGGER.info(
        f"Auto-label: generating pseudo-labels for {len(label_files)} images using {detector} "
        f"({n_pseudo_classes} COCO classes, offset={class_offset}, skip={skipped_names})..."
    )

    from ultralytics import YOLO

    model = YOLO(detector)
    n_stereo = 0
    n_mono = 0
    batch_size = 32

    for batch_start in range(0, len(label_files), batch_size):
        batch_files = label_files[batch_start : batch_start + batch_size]

        left_paths, right_paths, calibs, image_ids = [], [], [], []
        for lf in batch_files:
            img_id = lf.stem
            lp = _find_image(left_dir, img_id)
            rp = _find_image(right_dir, img_id)
            cp = calib_dir / f"{img_id}.txt"
            if lp and rp and cp.exists():
                left_paths.append(str(lp))
                right_paths.append(str(rp))
                calibs.append(_load_calib(cp))
                image_ids.append(img_id)

        if not left_paths:
            continue

        # Detect ALL 80 COCO classes on left and right images
        left_results = model(left_paths, conf=conf, iou=iou, verbose=False)
        right_results = model(right_paths, conf=conf, iou=iou, verbose=False)

        for img_id, lr, rr, calib in zip(image_ids, left_results, right_results, calibs):
            if calib is None:
                continue
            img_h, img_w = lr.orig_img.shape[:2]
            stereo_lines, mono_lines = _generate_pseudo_labels(lr, rr, calib, img_w, img_h, coco_to_kitti)
            lines = stereo_lines + mono_lines
            if lines:
                label_path = label_dir / f"{img_id}.txt"
                # Ensure newline before appending (original file may not end with \n)
                needs_sep = label_path.stat().st_size > 0 and not label_path.read_bytes().endswith(b"\n")
                with open(label_path, "a") as f:
                    if needs_sep:
                        f.write("\n")
                    f.write("\n".join(lines) + "\n")
                n_stereo += len(stereo_lines)
                n_mono += len(mono_lines)

    # Mark as done and invalidate stale label cache
    total = n_stereo + n_mono
    marker.write_text(str(total))
    _invalidate_cache(label_dir)

    LOGGER.info(
        f"Auto-label: generated {total} pseudo-labels "
        f"({n_stereo} stereo + {n_mono} mono, "
        f"{total / max(len(label_files), 1):.1f} per image)"
    )
    return total


def _find_image(directory: Path, image_id: str) -> Path | None:
    """Find image file by ID, trying common extensions."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = directory / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _load_calib(calib_path: Path) -> dict | None:
    """Load calibration file — supports both simple key-value and KITTI P2/P3 formats."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" in line:
                key, vals = line.split(":", 1)
                data[key.strip()] = vals.strip().split()

    # Simple key-value format: fx, fy, cx, cy, baseline as direct values
    if "fx" in data and "baseline" in data:
        return {
            "fx": float(data["fx"][0]),
            "fy": float(data["fy"][0]),
            "cx": float(data["cx"][0]),
            "cy": float(data["cy"][0]),
            "baseline": float(data["baseline"][0]),
        }

    # KITTI P2/P3 matrix format (12 values each, flattened 3x4)
    if "P2" in data and "P3" in data:
        p2 = [float(v) for v in data["P2"]]
        p3 = [float(v) for v in data["P3"]]
        fx = p2[0]
        return {"fx": fx, "fy": p2[5], "cx": p2[2], "cy": p2[6], "baseline": abs(p2[3] - p3[3]) / fx}

    return None


def _format_label_line(
    kitti_cls, l_cx, l_cy, l_w, l_h, r_cx, r_cy, r_w, r_h, mean_dims, x_3d, y_bottom, z_3d, img_w, img_h, occ
):
    """Format an 18-value pseudo-label line."""
    return (
        f"{kitti_cls} "
        f"{l_cx / img_w:.6f} {l_cy / img_h:.6f} {l_w / img_w:.6f} {l_h / img_h:.6f} "
        f"{r_cx / img_w:.6f} {r_cy / img_h:.6f} {r_w / img_w:.6f} {r_h / img_h:.6f} "
        f"{mean_dims[0]:.2f} {mean_dims[1]:.2f} {mean_dims[2]:.2f} "
        f"{x_3d:.2f} {y_bottom:.2f} {z_3d:.2f} "
        f"0.00 "
        f"0.0 {occ}"
    )


def _generate_pseudo_labels(left_result, right_result, calib, img_w, img_h, coco_to_kitti):
    """Generate pseudo-labels from L/R detections: stereo-matched + mono-only fallback.

    Args:
        coco_to_kitti: Dict mapping COCO class ID → kitti class ID. Classes not in
            this dict are skipped (either out of range or in skip_coco_ids).

    Returns:
        (stereo_lines, mono_lines): Two lists of 18-value label strings.
    """
    stereo_lines = []
    mono_lines = []

    l_boxes = left_result.boxes
    if l_boxes is None or len(l_boxes) == 0:
        return stereo_lines, mono_lines

    l_xyxy = l_boxes.xyxy.cpu().numpy()
    l_cls = l_boxes.cls.cpu().numpy().astype(int)

    r_boxes = right_result.boxes
    has_right = r_boxes is not None and len(r_boxes) > 0
    r_xyxy = r_boxes.xyxy.cpu().numpy() if has_right else np.zeros((0, 4))
    r_cls = r_boxes.cls.cpu().numpy().astype(int) if has_right else np.zeros(0, dtype=int)

    fx, fy = calib["fx"], calib["fy"]
    cx_cal, cy_cal = calib["cx"], calib["cy"]
    baseline = calib["baseline"]

    r_used = set()
    stereo_matched = set()  # left indices that got stereo matches

    # Pass 1: stereo matching (higher quality depth)
    for li in range(len(l_xyxy)):
        coco_cls = l_cls[li]
        kitti_cls = coco_to_kitti.get(coco_cls)
        if kitti_cls is None:
            continue

        _, mean_dims = _COCO80[coco_cls]
        lx1, ly1, lx2, ly2 = l_xyxy[li]
        l_cx, l_cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        l_w, l_h = lx2 - lx1, ly2 - ly1

        # Find best stereo match in right image
        best_ri, best_cost = -1, float("inf")
        for ri in range(len(r_xyxy)):
            if ri in r_used or r_cls[ri] != coco_cls:
                continue
            rx1, ry1, rx2, ry2 = r_xyxy[ri]
            r_cx, r_cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            r_h = ry2 - ry1

            if r_cx >= l_cx:  # Must have positive disparity
                continue
            dy = abs(l_cy - r_cy)
            if dy > 15:  # Stereo: ~same scanline
                continue
            dh = abs(l_h - r_h) / max(l_h, 1)
            if dh > 0.3:  # Similar apparent size
                continue

            cost = dy + dh * 30
            if cost < best_cost:
                best_cost, best_ri = cost, ri

        if best_ri < 0:
            continue

        r_used.add(best_ri)
        stereo_matched.add(li)
        rx1, ry1, rx2, ry2 = r_xyxy[best_ri]
        r_cx, r_cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
        r_w, r_h = rx2 - rx1, ry2 - ry1

        # Stereo triangulation → 3D
        disparity = l_cx - r_cx
        if disparity < 1.0:
            continue
        z_3d = fx * baseline / disparity
        if z_3d < 2.0 or z_3d > 80.0:
            continue
        x_3d = (l_cx - cx_cal) * z_3d / fx
        y_3d = (l_cy - cy_cal) * z_3d / fy
        y_bottom = y_3d + mean_dims[2] / 2

        stereo_lines.append(
            _format_label_line(
                kitti_cls,
                l_cx,
                l_cy,
                l_w,
                l_h,
                r_cx,
                r_cy,
                r_w,
                r_h,
                mean_dims,
                x_3d,
                y_bottom,
                z_3d,
                img_w,
                img_h,
                OCC_PSEUDO_STEREO,
            )
        )

    # Pass 2: mono-only fallback for unmatched left detections
    for li in range(len(l_xyxy)):
        if li in stereo_matched:
            continue
        coco_cls = l_cls[li]
        kitti_cls = coco_to_kitti.get(coco_cls)
        if kitti_cls is None:
            continue

        _, mean_dims = _COCO80[coco_cls]
        lx1, ly1, lx2, ly2 = l_xyxy[li]
        l_cx, l_cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        l_w, l_h = lx2 - lx1, ly2 - ly1

        if l_h < 10:  # too small to estimate depth reliably
            continue

        # Monocular depth from bbox height: z = fx * H_real / h_pixels
        real_h = mean_dims[2]  # object height in meters
        z_3d = fx * real_h / l_h
        if z_3d < 2.0 or z_3d > 80.0:
            continue

        x_3d = (l_cx - cx_cal) * z_3d / fx
        y_3d = (l_cy - cy_cal) * z_3d / fy
        y_bottom = y_3d + real_h / 2

        # Synthesize right bbox from estimated depth
        disparity = fx * baseline / z_3d
        r_cx = l_cx - disparity
        if r_cx < 0:  # right bbox would be off-image
            continue

        mono_lines.append(
            _format_label_line(
                kitti_cls,
                l_cx,
                l_cy,
                l_w,
                l_h,
                r_cx,
                l_cy,
                l_w,
                l_h,
                mean_dims,
                x_3d,
                y_bottom,
                z_3d,
                img_w,
                img_h,
                OCC_PSEUDO_MONO,
            )
        )

    return stereo_lines, mono_lines


def _invalidate_cache(label_dir: Path):
    """Delete stale stereo label caches so labels are re-parsed with pseudo-labels."""
    for cache_file in label_dir.parent.glob("stereo3d_*.cache"):
        cache_file.unlink()
        LOGGER.info(f"Auto-label: deleted stale cache {cache_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo-labels for stereo 3D detection")
    parser.add_argument("--data", required=True, help="Dataset YAML path (e.g., kitti-stereo.yaml)")
    parser.add_argument("--detector", default="yolo11m.pt", help="2D detector weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--class-offset", type=int, default=1, help="Starting class ID for pseudo-labels")
    parser.add_argument("--keep-coco-ids", type=int, nargs="*", default=None, help="Only keep these COCO IDs")
    parser.add_argument("--skip-coco-ids", type=int, nargs="*", default=None, help="Skip these COCO IDs")
    args = parser.parse_args()

    from ultralytics.data.utils import check_det_dataset

    data_cfg = check_det_dataset(args.data, autodownload=False)
    root = Path(data_cfg["path"])

    n = auto_label_stereo3d(
        label_dir=root / "labels" / "train",
        left_dir=root / "images" / "train" / "left",
        right_dir=root / "images" / "train" / "right",
        calib_dir=root / "calib" / "train",
        detector=args.detector,
        conf=args.conf,
        class_offset=args.class_offset,
        keep_coco_ids=set(args.keep_coco_ids) if args.keep_coco_ids else None,
        skip_coco_ids=set(args.skip_coco_ids) if args.skip_coco_ids else None,
    )
    LOGGER.info(f"Generated {n} pseudo-labels")
