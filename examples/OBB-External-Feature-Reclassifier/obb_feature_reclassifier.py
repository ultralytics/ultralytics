# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
r"""
DOTA v1.0 OBB External Feature Reclassifier
============================================

Bolt-on classification improvement for YOLO OBB models using a frozen,
domain-agnostic spatial feature encoder. The encoder has never been trained
on aerial imagery or any of the DOTA object categories. It captures spatial
structure that detection models systematically discard, providing an
orthogonal information source that improves classification without modifying
the detector.

The main idea:
    1. Let YOLO detect objects normally
    2. Crop each detection using result.obb.xyxy (axis-aligned enclosure)
    3. Send crops to the Authorize Earth encoder API (returns a 920-dim vector)
    4. Concatenate YOLO's class prediction with the encoder vector
    5. Train a lightweight LightGBM classifier on the concatenated features
    6. Measure whether classification improves under 5-fold cross-validation

Setup:
    pip install ultralytics==8.4.21 lightgbm scikit-learn numpy opencv-python requests pillow tqdm

Tested with:
    Ultralytics 8.4.21, Python 3.11.9, torch 2.11.0+cu128, NVIDIA RTX 5080

Data:
    DOTA v1.0 validation set from https://captain-whu.github.io/DOTA/
    You need both the images and the labelTxt directories.

Usage:
    # Step 1: tile the full-size DOTA images into 1024x1024 patches
    python obb_feature_reclassifier.py tile \
        --images path/to/val/images \
        --labels path/to/val/labelTxt \
        --output ./tiled

    # Step 2: run the benchmark
    python obb_feature_reclassifier.py bench \
        --images ./tiled/images \
        --labels ./tiled/labels

    # With YOLO26:
    python obb_feature_reclassifier.py bench \
        --images ./tiled/images \
        --labels ./tiled/labels \
        --model yolo26l-obb.pt

    # With full 15-dim pre-NMS class scores (YOLOv8 only):
    python obb_feature_reclassifier.py bench \
        --images ./tiled/images \
        --labels ./tiled/labels \
        --full-scores

    # Skip detection on subsequent runs (uses cached results):
    python obb_feature_reclassifier.py bench \
        --images ./tiled/images \
        --labels ./tiled/labels \
        --skip-detection

Note on --full-scores:
    Extracts the raw 15-dim sigmoid class scores from YOLOv8's pre-NMS output
    and matches them back to post-NMS detections. This gives the baseline a
    stronger representation (full class distribution vs one-hot). Only valid
    for NMS-based models like YOLOv8. YOLO26's end-to-end NMS-free architecture
    uses a one-to-one prediction head, so intermediate activations are not
    interpretable class distributions. Use one-hot (default) for YOLO26.

Cross-validation grouping:
    DOTA images are large (up to 20k x 20k) and must be tiled into 1024x1024
    patches for inference. Tiles from the same original image share terrain,
    lighting, and nearby objects but contain mostly different detections
    (200px overlap out of 1024px). This benchmark uses tile-level GroupKFold:
    tiles are grouped by their x-offset pattern within each original image, so
    the classifier trains on some tiles from an image and is tested on other
    tiles from the same image at different positions. This mirrors a realistic
    deployment scenario where labeled data from the operating environment is
    available and new detections from the same region need classification.

    For strict image-level holdout (all tiles from an original image in the
    same fold), pass --strict-scene-split. Under strict splitting, the
    improvement on DOTA is marginal to slightly negative, consistent with the
    encoder capturing spatial context that transfers well within a deployment
    region but not across completely unseen aerial scenes in this dataset.
    Other benchmarks with unambiguous scene boundaries (xView2 disaster events,
    RarePlanes satellite passes) show 22-40% error reduction under strict
    scene-level splits, suggesting cross-scene transfer depends on the
    diversity of objects and conditions across scenes.

API:
    Uses a public evaluation key with no signup required. Rate-limited to
    3,000 requests/hour per IP, 15,000/hour globally. For higher limits
    contact jackk@authorize.earth.

    Encoder docs: https://authorize.earth/r&d/spatial
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

API_URL = "https://api.authorize.earth/v1/spatial/encode"
API_KEY = os.environ.get("AUTHORIZE_EARTH_API_KEY", "")
BATCH_SIZE = 64

DOTA_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

IOU_THRESH = 0.5


def parse_dota_label(path):
    """Read a DOTA annotation file.

    Args:
        path (str): Path to the DOTA label file.

    Returns:
        list[tuple[str, np.ndarray]]: List of (class_name, 4x2 polygon) tuples.
    """
    anns = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
                cls = parts[8]
                if cls not in DOTA_CLASSES:
                    continue
                anns.append((cls, np.array(coords).reshape(4, 2)))
            except (ValueError, IndexError):
                continue
    return anns


def obb_to_xyxy(pts):
    """Convert oriented bounding box points to axis-aligned [x1, y1, x2, y2].

    Args:
        pts (np.ndarray): Oriented bounding box corners with shape (4, 2).

    Returns:
        np.ndarray: Axis-aligned bounding box [x1, y1, x2, y2].
    """
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def tile_dataset(images_dir, labels_dir, output_dir, tile_size=1024, overlap=200):
    """Split DOTA images and labels into tiles for YOLO inference.

    DOTA images can be up to 20k x 20k pixels. This function tiles them into manageable patches with overlap, keeping
    only tiles that contain at least one annotated object.

    Args:
        images_dir (str): Directory containing full-size DOTA images (.png).
        labels_dir (str): Directory containing DOTA labelTxt annotation files.
        output_dir (str): Output directory for tiled images and labels.
        tile_size (int): Tile width and height in pixels.
        overlap (int): Overlap between adjacent tiles in pixels.

    Returns:
        tuple[str, str]: Paths to the output images and labels directories.
    """
    out_img = os.path.join(output_dir, "images")
    out_lbl = os.path.join(output_dir, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    image_files = sorted(Path(images_dir).glob("*.png"))
    print(f"Tiling {len(image_files)} images ({tile_size}x{tile_size}, {overlap}px overlap)")

    total = 0
    stride = tile_size - overlap

    for img_path in tqdm(image_files, desc="Tiling"):
        label_path = os.path.join(labels_dir, img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        anns = parse_dota_label(label_path) if os.path.exists(label_path) else []
        h, w = img.shape[:2]

        xs = sorted(set([*list(range(0, max(1, w - tile_size + 1), stride)), max(0, w - tile_size)]))
        ys = sorted(set([*list(range(0, max(1, h - tile_size + 1), stride)), max(0, h - tile_size)]))

        for y0 in ys:
            for x0 in xs:
                tile_anns = []
                for cls, pts in anns:
                    shifted = pts.copy()
                    shifted[:, 0] -= x0
                    shifted[:, 1] -= y0
                    cx, cy = shifted[:, 0].mean(), shifted[:, 1].mean()
                    if cx < 0 or cx >= tile_size or cy < 0 or cy >= tile_size:
                        continue
                    shifted[:, 0] = np.clip(shifted[:, 0], 0, tile_size)
                    shifted[:, 1] = np.clip(shifted[:, 1], 0, tile_size)
                    bw = shifted[:, 0].max() - shifted[:, 0].min()
                    bh = shifted[:, 1].max() - shifted[:, 1].min()
                    if bw < 3 or bh < 3:
                        continue
                    tile_anns.append((cls, shifted))

                if not tile_anns:
                    continue

                tile = img[y0 : y0 + tile_size, x0 : x0 + tile_size]
                ph, pw = tile_size - tile.shape[0], tile_size - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = cv2.copyMakeBorder(tile, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

                name = f"{img_path.stem}_{x0}_{y0}"
                cv2.imwrite(os.path.join(out_img, name + ".png"), tile)

                with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
                    for cls, pts in tile_anns:
                        coords = " ".join(f"{c:.1f}" for c in pts.flatten())
                        f.write(f"{coords} {cls} 0\n")
                total += 1

    print(f"Done: {total} tiles with annotations")
    return out_img, out_lbl


def extract_raw_scores(model, image, imgsz=1024):
    """Extract pre-NMS class scores from the raw model output.

    Runs the model backbone and head directly to get sigmoid class scores for every candidate before NMS filtering. Only
    valid for NMS-based architectures like YOLOv8. YOLO26's NMS-free head does not produce interpretable class
    distributions at this stage.

    Args:
        model: Ultralytics YOLO model instance.
        image (np.ndarray): Input image in BGR format.
        imgsz (int): Inference image size.

    Returns:
        np.ndarray: Pre-NMS class scores with shape (N_candidates, num_classes).
    """
    lb = LetterBox((imgsz, imgsz), auto=True, stride=32)
    im = lb(image=image)
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im_t = torch.from_numpy(im).unsqueeze(0).float().to(next(model.model.parameters()).device) / 255.0
    with torch.no_grad():
        raw = model.model(im_t)
    pred = raw[0].squeeze(0).T.cpu().numpy()
    nc = len(DOTA_CLASSES)
    return pred[:, 4 : 4 + nc]


def match_raw_to_detections(obb, raw_scores):
    """Match post-NMS detections back to pre-NMS candidates by class score.

    Args:
        obb: Ultralytics OBB results object.
        raw_scores (np.ndarray): Pre-NMS class scores from extract_raw_scores.

    Returns:
        np.ndarray: Full class score vectors with shape (N_detections, num_classes).
    """
    post_data = obb.data.cpu().numpy()
    full_scores = []
    used = set()

    for i in range(len(obb)):
        cls_id = int(post_data[i, 6])
        conf = post_data[i, 5]
        candidate_cls_scores = raw_scores[:, cls_id]
        conf_diff = np.abs(candidate_cls_scores - conf)
        order = np.argsort(conf_diff)
        matched = False
        for idx in order:
            if idx not in used and conf_diff[idx] < 0.02:
                used.add(idx)
                full_scores.append(raw_scores[idx])
                matched = True
                break
        if not matched:
            fallback = np.zeros(raw_scores.shape[1])
            fallback[cls_id] = conf
            full_scores.append(fallback)

    return np.array(full_scores)


def compute_iou(a, b):
    """Compute IoU between two axis-aligned bounding boxes.

    Args:
        a (np.ndarray): First box [x1, y1, x2, y2].
        b (np.ndarray): Second box [x1, y1, x2, y2].

    Returns:
        float: Intersection over Union score.
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(pred_boxes, pred_cls, pred_conf, pred_probs, gt_boxes, gt_cls):
    """Greedy IoU matching between predictions and ground truth.

    Args:
        pred_boxes (list[np.ndarray]): Predicted bounding boxes.
        pred_cls (list[str]): Predicted class names.
        pred_conf (list[float]): Prediction confidences.
        pred_probs (list[np.ndarray]): Class probability vectors per detection.
        gt_boxes (list[np.ndarray]): Ground truth bounding boxes.
        gt_cls (list[str]): Ground truth class names.

    Returns:
        list[tuple]: Matched (pred_idx, gt_idx, pred_class, gt_class, conf, probs, box) tuples.
    """
    if not pred_boxes or not gt_boxes:
        return []

    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            ious[i, j] = compute_iou(p, g)

    matches = []
    used_p, used_g = set(), set()
    while True:
        idx = np.unravel_index(ious.argmax(), ious.shape)
        if ious[idx] < IOU_THRESH:
            break
        pi, gi = idx
        if pi not in used_p and gi not in used_g:
            matches.append((pi, gi, pred_cls[pi], gt_cls[gi], pred_conf[pi], pred_probs[pi], pred_boxes[pi]))
            used_p.add(pi)
            used_g.add(gi)
        ious[pi, :] = 0
        ious[:, gi] = 0

    return matches


def run_detection(model, images_dir, labels_dir, full_scores=False, strict_scene_split=False):
    """Run YOLO OBB detection on every tile and match detections to ground truth.

    Args:
        model: Ultralytics YOLO model instance.
        images_dir (str): Directory of tiled images.
        labels_dir (str): Directory of tiled labels.
        full_scores (bool): Use full pre-NMS class scores instead of one-hot.
        strict_scene_split (bool): Group by original image instead of tile offset.

    Returns:
        list[dict]: Detection records with scene, classes, probabilities, and crops.
    """
    files = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Running detection on {len(files)} tiles")
    if full_scores:
        print("Using full 15-dim raw class scores (pre-NMS)")
        model.model.eval()
    if strict_scene_split:
        print("Using strict image-level scene grouping for cross-validation")
    else:
        print("Using tile-level grouping for cross-validation (deployment scenario)")

    records = []
    for path in tqdm(files, desc="Detecting"):
        stem = Path(path).stem
        if strict_scene_split:
            scene = stem.rsplit("_", 2)[0] if stem.count("_") >= 2 else stem
        else:
            scene = stem.rsplit("_", 1)[0] if "_" in stem else stem

        lbl = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(lbl):
            continue
        gt = parse_dota_label(lbl)
        if not gt:
            continue

        gt_cls = [a[0] for a in gt]
        gt_boxes = [obb_to_xyxy(a[1]) for a in gt]

        img = cv2.imread(path)
        if img is None:
            continue

        res = model.predict(path, verbose=False, conf=0.25, imgsz=1024)
        if not res or res[0].obb is None:
            continue

        obb = res[0].obb

        if full_scores:
            raw_scores = extract_raw_scores(model, img, imgsz=1024)
            score_matrix = match_raw_to_detections(obb, raw_scores)
        else:
            score_matrix = None

        p_boxes, p_cls, p_conf, p_probs = [], [], [], []

        for i in range(len(obb)):
            box = obb.xyxy[i].cpu().numpy()
            cid = int(obb.cls[i].cpu())
            cf = float(obb.conf[i].cpu())

            if full_scores and score_matrix is not None:
                probs = score_matrix[i]
            else:
                probs = np.zeros(len(DOTA_CLASSES))
                probs[cid] = cf

            p_boxes.append(box)
            p_cls.append(DOTA_CLASSES[cid] if cid < len(DOTA_CLASSES) else "unknown")
            p_conf.append(cf)
            p_probs.append(probs)

        for pi, gi, pc, gc, cf, probs, box in match_preds_to_gt(p_boxes, p_cls, p_conf, p_probs, gt_boxes, gt_cls):
            h, w = img.shape[:2]
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(img[y1:y2, x1:x2], (128, 128), interpolation=cv2.INTER_AREA)

            records.append(
                {
                    "scene": scene,
                    "pred_class": pc,
                    "gt_class": gc,
                    "class_probs": probs,
                    "crop": crop,
                }
            )

    print(f"Matched detections: {len(records)}")
    return records


def encode_crops(crops):
    """Send a batch of 128x128 crops to the Authorize Earth encoder API.

    Args:
        crops (list[np.ndarray]): List of BGR image crops, each 128x128.

    Returns:
        list[list[float]]: List of 920-dimensional feature vectors.
    """
    if not API_KEY:
        print("  AUTHORIZE_EARTH_API_KEY not set, returning zero vectors")
        return [np.zeros(920).tolist() for _ in crops]
    b64_list = []
    for c in crops:
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if len(c.shape) == 3 else c
        buf = io.BytesIO()
        Image.fromarray(gray).save(buf, format="JPEG", quality=95)
        b64_list.append(base64.b64encode(buf.getvalue()).decode())

    resp = requests.post(
        API_URL,
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        json={"action": "encode_batch", "images_base64": b64_list},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json().get("data", resp.json())
    feats = data["features"]
    return feats if isinstance(feats[0], list) else [feats]


def encode_all(records):
    """Encode every detection crop via the API with batching and retries.

    Args:
        records (list[dict]): Detection records containing 'crop' arrays.

    Returns:
        list[list[float]]: Feature vectors for all detections.
    """
    crops = [r["crop"] for r in records]
    features = []
    n_batches = (len(crops) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(crops), BATCH_SIZE), total=n_batches, desc="Encoding via API"):
        batch = crops[i : i + BATCH_SIZE]
        for attempt in range(3):
            try:
                features.extend(encode_crops(batch))
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2**attempt)
                else:
                    print(f"  Batch failed after 3 attempts: {e}")
                    features.extend([np.zeros(920).tolist()] * len(batch))

    return features


def evaluate(records, features, model_name):
    """Run 5-fold cross-validation comparing YOLO vs bolt-on.

    Compares YOLO's direct class predictions against a LightGBM classifier trained on YOLO class outputs concatenated
    with encoder feature vectors.

    Args:
        records (list[dict]): Detection records with ground truth and predictions.
        features (list[list[float]]): Encoder feature vectors per detection.
        model_name (str): Model name for display in results.
    """
    cls2idx = {c: i for i, c in enumerate(DOTA_CLASSES)}

    y = np.array([cls2idx[r["gt_class"]] for r in records])
    groups = np.array([r["scene"] for r in records])
    x_yolo = np.array([r["class_probs"] for r in records])
    x_enc = np.array(features)
    x_both = np.concatenate([x_yolo, x_enc], axis=1)

    yolo_direct = np.array([cls2idx.get(r["pred_class"], 0) for r in records])

    n_groups = len(set(groups))
    print(f"\n5-fold cross-validation ({len(records)} detections, {n_groups} groups)")
    print("=" * 65)

    gkf = GroupKFold(n_splits=5)
    bolt_preds = np.zeros(len(y), dtype=int)

    for fold, (tr, te) in enumerate(gkf.split(x_both, y, groups)):
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )
        clf.fit(x_both[tr], y[tr])
        bolt_preds[te] = clf.predict(x_both[te])

        f1 = f1_score(y[te], bolt_preds[te], average="weighted")
        print(f"  Fold {fold + 1}: bolt-on weighted F1 = {f1:.4f}")

    yd_wf1 = f1_score(y, yolo_direct, average="weighted")
    bt_wf1 = f1_score(y, bolt_preds, average="weighted")
    yd_mf1 = f1_score(y, yolo_direct, average="macro")
    bt_mf1 = f1_score(y, bolt_preds, average="macro")

    yd_err = 1.0 - yd_wf1
    bt_err = 1.0 - bt_wf1
    err_red = (yd_err - bt_err) / yd_err * 100 if yd_err > 0 else 0

    print(f"\n{'':─<65}")
    print(f"  {model_name + ' Direct':<20} {'Bolt-On':<20}")
    print(f"  Weighted F1:  {yd_wf1:<20.4f} {bt_wf1:<20.4f}")
    print(f"  Macro F1:     {yd_mf1:<20.4f} {bt_mf1:<20.4f}")
    print(f"  Error reduction vs {model_name} direct: {err_red:.1f}%")

    yd_per = f1_score(y, yolo_direct, average=None, labels=range(len(DOTA_CLASSES)))
    bt_per = f1_score(y, bolt_preds, average=None, labels=range(len(DOTA_CLASSES)))

    rows = []
    for i, c in enumerate(DOTA_CLASSES):
        n = np.sum(y == i)
        if n > 0:
            rows.append((c, yd_per[i], bt_per[i], bt_per[i] - yd_per[i], n))
    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"\n  {'Class':<22} {'Direct':<10} {'Bolt-On':<10} {'Delta':<10} {'n'}")
    print(f"  {'':─<62}")
    for c, d, b, delta, n in rows:
        s = "+" if delta >= 0 else ""
        print(f"  {c:<22} {d:<10.3f} {b:<10.3f} {s}{delta:<10.3f} {n}")


def main():
    """Parse arguments and run the tiling or benchmarking pipeline."""
    parser = argparse.ArgumentParser(description="OBB External Feature Reclassifier")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("tile", help="Tile full-size DOTA images into 1024x1024 patches")
    t.add_argument("--images", required=True, help="Directory of full-size DOTA images")
    t.add_argument("--labels", required=True, help="Directory of DOTA labelTxt files")
    t.add_argument("--output", required=True, help="Output directory for tiled data")
    t.add_argument("--tile-size", type=int, default=1024)
    t.add_argument("--overlap", type=int, default=200)

    b = sub.add_parser("bench", help="Run the bolt-on classification benchmark")
    b.add_argument("--images", required=True, help="Directory of tiled images")
    b.add_argument("--labels", required=True, help="Directory of tiled labels")
    b.add_argument("--model", default="yolov8l-obb.pt", help="YOLO OBB model weights")
    b.add_argument("--cache-dir", default="./dota_cache", help="Cache directory for detections and features")
    b.add_argument("--skip-detection", action="store_true", help="Load cached detections instead of re-running YOLO")
    b.add_argument("--full-scores", action="store_true", help="Use full pre-NMS class scores (YOLOv8 only)")
    b.add_argument(
        "--strict-scene-split",
        action="store_true",
        help="Group by original image for strict scene-level holdout (default: tile-level grouping)",
    )

    args = parser.parse_args()

    if args.cmd == "tile":
        tile_dataset(args.images, args.labels, args.output, args.tile_size, args.overlap)

    elif args.cmd == "bench":
        os.makedirs(args.cache_dir, exist_ok=True)
        rec_path = os.path.join(args.cache_dir, "records.npz")
        crop_path = os.path.join(args.cache_dir, "crops.npy")
        feat_path = os.path.join(args.cache_dir, "features.npy")

        if args.skip_detection and os.path.exists(rec_path) and os.path.exists(crop_path):
            print("Loading cached detections...")
            records = np.load(rec_path, allow_pickle=True)["records"].tolist()
            crops = np.load(crop_path)
            for i, r in enumerate(records):
                r["crop"] = crops[i]
            print(f"  {len(records)} detections loaded")
        else:
            model = YOLO(args.model)
            records = run_detection(
                model, args.images, args.labels,
                full_scores=args.full_scores,
                strict_scene_split=args.strict_scene_split,
            )
            np.save(crop_path, np.array([r["crop"] for r in records]))
            serializable = [
                {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in r.items() if k != "crop"}
                for r in records
            ]
            np.savez(rec_path, records=serializable)

        if os.path.exists(feat_path) and args.skip_detection:
            print("Loading cached encoder features...")
            features = np.load(feat_path).tolist()
            if len(features) != len(records):
                print(f"  Cache mismatch ({len(features)} features vs {len(records)} records), re-encoding...")
                features = encode_all(records)
                np.save(feat_path, np.array(features))
        else:
            features = encode_all(records)
            np.save(feat_path, np.array(features))

        evaluate(records, features, Path(args.model).stem)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
