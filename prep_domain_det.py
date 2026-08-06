# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the domain-detection corpus sources into standard YOLO datasets.

Each dataset keeps its OWN class space. Nothing is merged into a unified taxonomy, because the trainer slices logits
per dataset and batches are kept dataset-pure, so a shared index space would buy nothing and cost the label
contradictions it is supposed to solve.

Every source is extracted once, then ``images/<split>/`` is filled with symlinks to the extracted files rather than
copies, so a 200GB corpus does not become 400GB. Labels are written beside them and a ``data.yaml`` is emitted.

Adapters return ``{split: {image_path: [(cls, cx, cy, w, h)]}}`` plus the ordered class names, and the shared writer
handles everything after that. Add a dataset by writing one adapter and registering it. A source whose boxes are
oriented returns eight coordinates instead of four and the writer keeps them, matching the sibling SODA-A output.

Usage:
    python prep_domain_det.py --list
    python prep_domain_det.py sardet doclaynet
    python prep_domain_det.py --all
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.data.utils import exif_size
from ultralytics.utils import YAML
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path("/data/shared-datasets/domain-det")
JUNK = ("__MACOSX", "._", ".DS_Store")
VAL_EVERY = 20  # stride for sources that ship no split of their own


def is_junk(name: str) -> bool:
    """Report whether an archive member is a macOS sidecar or metadata entry rather than a real file.

    Args:
        name (str): Archive member name.

    Returns:
        (bool): True when the member should be skipped.
    """
    return any(j in name for j in JUNK)


def link_names(paths: list) -> list:
    """Name each output image by the fewest trailing source path components that keep every name unique.

    Single-source datasets keep their original filenames. Sources that merge several sub-directories into one split pick
    up as much of their path as it takes to separate them, which matters because sub-datasets routinely number their
    images from zero and would otherwise overwrite each other.

    Args:
        paths (list): Source image paths within one split.

    Returns:
        (list): One output filename per input path, in the same order.
    """
    for depth in range(1, 6):
        names = ["__".join(p.parts[-depth:]) for p in paths]
        if len(set(names)) == len(names):
            return names
    raise ValueError(f"{len(paths) - len(set(names))} images still collide using {depth} path components")


def carve_val(by_file: dict, group=lambda p: p, every: int = VAL_EVERY) -> dict:
    """Split a source that ships no split of its own into train and val.

    The val slice exists so the yaml is loadable and spot-checkable. It is a deterministic stride, so these datasets
    carry no reportable metric, which is fine because they are pretraining data.

    Sources built by slicing 3D volumes pass ``group`` to stride over scans instead of files. Striding over files would
    put slice 99 in train and slice 100 in val, which are the same picture of the same patient.

    Args:
        by_file (dict): Image path to its boxes.
        group (Callable, optional): Maps an image path to the scan it came from.
        every (int, optional): Stride over groups, lowered when a source has few enough groups that the default would
            leave val nearly empty.

    Returns:
        (dict): Split name to image path to its boxes.
    """
    val = {k for i, k in enumerate(dict.fromkeys(map(group, by_file))) if not i % every}
    return {
        "train": {p: b for p, b in by_file.items() if group(p) not in val},
        "val": {p: b for p, b in by_file.items() if group(p) in val},
    }


def read_yolo_labels(path: Path) -> list:
    """Parse a YOLO label file into rows, treating a missing file as a background image.

    Args:
        path (Path): Label file path, which need not exist.

    Returns:
        (list): One (class index, cx, cy, w, h) per row.
    """
    text = path.read_text() if path.exists() else ""
    return [(int(r[0]), *(float(v) for v in r[1:5])) for r in (x.split() for x in text.splitlines() if x.strip())]


def extract(archive: Path, dest: Path) -> Path:
    """Extract an archive once, skipping the work if it already ran.

    Args:
        archive (Path): Zip or tar to extract.
        dest (Path): Directory to extract into.

    Returns:
        (Path): The destination directory.
    """
    if (dest / ".extracted").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name}")
    bad = []
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if is_junk(n):
                    continue
                try:
                    z.extract(n, dest)
                except zipfile.BadZipFile:
                    # the DroneVehicle train zip ships 39 members with a broken CRC upstream, and a failed extract
                    # leaves the partial file behind, which every adapter would then read as a real image
                    (dest / n).unlink(missing_ok=True)
                    bad.append(n)
            if bad:
                print(f"  {len(bad)} unreadable members skipped, first {bad[0]}")
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, members=[m for m in t if not is_junk(m.name)], filter="data")
    (dest / ".extracted").write_text("\n".join(bad))  # names the members a rerun would still be missing
    return dest


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple | None:
    """Clip an absolute xyxy box to the image and normalize it, or reject it as degenerate.

    Args:
        x1 (float): Left edge.
        y1 (float): Top edge.
        x2 (float): Right edge.
        y2 (float): Bottom edge.
        w (int): Image width.
        h (int): Image height.

    Returns:
        (tuple | None): Normalized (cx, cy, bw, bh), or None if the clipped box has no area.
    """
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def coco_adapter(anno_splits: dict, image_root, drop: set = frozenset()) -> tuple[dict, list]:
    """Read COCO jsons into the adapter contract, remapping category ids to contiguous indices.

    Args:
        anno_splits (dict): Split name to the COCO json path.
        image_root (dict | Path): Split name to that split's image root, or one root shared by every split.
        drop (set, optional): Category names to exclude entirely.

    Returns:
        rows (dict): Split to image path to its list of (class index, cx, cy, w, h).
        names (list): Ordered class names.
    """
    parsed = {p: json.loads(Path(p).read_text()) for p in set(anno_splits.values())}
    cats: dict[int, str] = {}
    for d in parsed.values():
        for c in d["categories"]:
            cats[c["id"]] = c["name"]  # duplicated rows are common, last wins and they agree
    names = [n for _, n in sorted(cats.items()) if n not in drop]
    index = {i: names.index(n) for i, n in cats.items() if n not in drop}

    rows: dict[str, dict] = {}
    for split, path in anno_splits.items():
        d = parsed[path]
        root = image_root[split] if isinstance(image_root, dict) else image_root
        meta = {im["id"]: im for im in d["images"]}
        by_file: dict[Path, list] = defaultdict(list)
        for im in d["images"]:  # register every image so a label-free one becomes a background, not a miss
            by_file[Path(root) / im["file_name"]]
        for a in d["annotations"]:
            if "bbox" not in a or a["category_id"] not in index:
                continue  # CCT20 ships bbox-less annotations across every category, not only "empty"
            im = meta[a["image_id"]]
            x, y, bw, bh = a["bbox"]
            if box := xyxy_to_yolo(x, y, x + bw, y + bh, im["width"], im["height"]):
                by_file[Path(root) / im["file_name"]].append((index[a["category_id"]], *box))
        rows[split] = dict(by_file)
    return rows, names


def yolo_adapter(split_dirs: dict, names: list, image_dir: str = "images", label_dir: str = "labels") -> tuple:
    """Pass through a source that is already YOLO, pairing each image with its label file.

    Args:
        split_dirs (dict): Split name to the directory holding that split, or to a list of directories to merge.
        names (list): Ordered class names.
        image_dir (str, optional): Image subdirectory name, empty when labels sit beside the images.
        label_dir (str, optional): Label subdirectory name, empty when labels sit beside the images.

    Returns:
        rows (dict): Split to image path to its list of (class index, cx, cy, w, h).
        names (list): The class names passed in.
    """
    rows: dict[str, dict] = {}
    for split, dirs in split_dirs.items():
        by_file = {}
        for d in [dirs] if isinstance(dirs, (str, Path)) else dirs:
            d = Path(d)
            for im in sorted(p for p in (d / image_dir).iterdir() if p.suffix.lower() != ".txt"):
                by_file[im] = read_yolo_labels(d / label_dir / f"{im.stem}.txt")
        rows[split] = by_file
    return rows, names


def emit(root: Path, rows: dict, names: list, val: str, viz: int) -> Counter:
    """Link images, write labels and data.yaml, then render previews for visual inspection.

    Every split other than ``val`` becomes a training source, because these sets are pretraining data and their held-out
    splits carry no metric we report.

    Args:
        root (Path): Output dataset root.
        rows (dict): Split to image path to its boxes.
        names (list): Ordered class names.
        val (str): Which split of ``rows`` serves as val.
        viz (int): Previews to render per split.

    Returns:
        (Counter): Counters for images, boxes, backgrounds, and images missing from disk.
    """
    stats = Counter({k: 0 for k in ("images", "boxes", "background", "missing")})
    for split, by_file in rows.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        present = {p: b for p, b in by_file.items() if p.exists()}
        stats["missing"] += len(by_file) - len(present)
        for (src, boxes), out in zip(present.items(), link_names(list(present))):
            link = root / "images" / split / out
            if not link.is_symlink():
                link.symlink_to(src)
            lb = root / "labels" / split / f"{Path(out).stem}.txt"
            if boxes:
                # write however many coordinates the adapter produced, so oriented rows keep all four corners
                lb.write_text("".join(f"{b[0]} " + " ".join(f"{v:.6g}" for v in b[1:]) + "\n" for b in boxes))
            else:
                lb.unlink(missing_ok=True)
                stats["background"] += 1
            stats.update(images=1, boxes=len(boxes))

    # Adapters that reach disk with glob rather than an archive read see an empty source as an empty result, so without
    # this a missing input writes a loadable dataset holding nothing and reports success.
    assert stats["images"], f"{root.name} produced no images, so its source is missing or empty"
    assert val in rows, f"val split {val!r} is not one of {sorted(rows)}"
    train = [f"images/{s}" for s in rows if s != val]
    yaml = {"path": str(root), "train": train, "val": f"images/{val}", "names": dict(enumerate(names))}
    YAML.save(root / "data.yaml", yaml)
    stats["previews"] = visualize(root, names, viz)
    return stats


def visualize(root: Path, names: list, n: int) -> int:
    """Render boxes onto sample images from every split so a conversion regression is visible.

    Args:
        root (Path): Dataset root containing ``images/`` and ``labels/``.
        names (list): Ordered class names.
        n (int): Number of images to render per split.

    Returns:
        (int): Number of previews rendered.
    """
    out_dir = root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    written = 0
    for split_dir in sorted((root / "labels").iterdir()):
        if not split_dir.is_dir():
            continue
        labels = sorted(split_dir.glob("*.txt"))
        for lb in rng.sample(labels, min(n, len(labels))):
            hits = list((root / "images" / split_dir.name).glob(f"{lb.stem}.*"))
            im = cv2.imread(str(hits[0])) if hits else None
            if im is None:
                continue
            rows = [x.split() for x in lb.read_text().splitlines() if x.strip()]
            a = Annotator(im, line_width=max(2, im.shape[0] // 400))
            for r in rows:
                c = int(r[0])
                if len(r) == 9:  # oriented, draw the polygon rather than reading its first four numbers as a box
                    pts = (np.array(r[1:], dtype=np.float32).reshape(4, 2) * [im.shape[1], im.shape[0]]).tolist()
                    a.box_label(pts, names[c], colors(c, True))
                    continue
                xywhn = np.array([[float(v) for v in r[1:5]]], dtype=np.float32)
                a.box_label(xywhn2xyxy(xywhn, w=im.shape[1], h=im.shape[0])[0], names[c], colors(c, True))
            a.text((8, 8), f"{lb.stem}  n={len(rows)}")
            cv2.imwrite(str(out_dir / f"{split_dir.name}__{lb.stem}.jpg"), a.result())
            written += 1
    return written


# --------------------------------------------------------------------------------------------- per-dataset adapters


def sardet(root: Path) -> tuple[dict, list]:
    """SAR ship and vehicle detection, COCO with category ids already 0-based."""
    d = extract(root / "sardet-100k.zip", root / "_src") / "SARDet_100K"
    anno = {"train": d / "Annotations/train.json", "val": d / "Annotations/val.json", "test": d / "test.json"}
    return coco_adapter(anno, {s: d / "JPEGImages" / s for s in anno})


def doclaynet(root: Path) -> tuple[dict, list]:
    """Document layout, COCO over one flat PNG directory shared by all splits."""
    d = extract(root / "DocLayNet_core.zip", root / "_src")
    return coco_adapter({s: d / "COCO" / f"{s}.json" for s in ("train", "val", "test")}, d / "PNG")


def cct20(root: Path) -> tuple[dict, list]:
    """Camera-trap wildlife, COCO with uuid image ids and bbox-less rows in every category."""
    a = extract(root / "eccv_18_annotations.tar.gz", root / "_src_anno") / "eccv_18_annotation_files"
    im = extract(root / "eccv_18_all_images_sm.tar.gz", root / "_src_img") / "eccv_18_all_images_sm"
    splits = ("train", "cis_val", "trans_val", "cis_test", "trans_test")
    return coco_adapter({s: a / f"{s}_annotations.json" for s in splits}, im, drop={"empty"})


def flir(root: Path) -> tuple[dict, list]:
    """FLIR ADAS thermal only, excluding the RGB halves. Test is named video_thermal_test, not images_*."""
    d = extract(root / "flir-adas-v2.zip", root / "_src") / "FLIR_ADAS_v2"
    subs = {"train": "images_thermal_train", "val": "images_thermal_val", "test": "video_thermal_test"}
    return coco_adapter({s: d / v / "coco.json" for s, v in subs.items()}, {s: d / v for s, v in subs.items()})


def duo(root: Path) -> tuple[dict, list]:
    """Underwater creatures. Basenames collide across splits, which separate output dirs already handle."""
    d = extract(root / "duo-underwater.zip", root / "_src") / "DUO"
    anno = {s: d / "annotations" / f"instances_{s}.json" for s in ("train", "test")}
    return coco_adapter(anno, {s: d / "images" / s for s in anno})


CAMUS_FRAMES = 4  # frames kept per cardiac sequence, evenly spaced over the cycle


def camus(root: Path) -> tuple[dict, list]:
    """Echocardiography, one cardiac sequence per view sliced into frames.

    The third axis is time rather than space, so neighbouring frames are the same heart a few milliseconds apart. Four
    evenly spaced frames span one cycle, over which the myocardium contracts and the atrium fills, and that deformation
    is the only variation a sequence holds. Boxes come from the mask array the frame was cut from, so no reorientation
    sits between the two.

    Both arrays are transposed because the file stores the lateral axis first, which lays the probe apex along the left
    edge instead of the top. Mask value 1 is the ventricle cavity and 2 is the epicardial contour that encloses it, so
    the names run cavity first.
    """
    import nibabel as nib  # scope so the other adapters run without it

    names = ["left ventricle", "left ventricular myocardium", "left atrium"]
    frames = root / "_frames"
    frames.mkdir(parents=True, exist_ok=True)
    by_file = {}

    def load(z, n):
        """Read one gzipped NIfTI volume out of an open zip."""
        return nib.Nifti1Image.from_bytes(gzip.decompress(z.read(n))).get_fdata().T

    with zipfile.ZipFile(root / "Images.zip") as zi, zipfile.ZipFile(root / "Masks.zip") as zm:
        for n in sorted(x for x in zi.namelist() if x.endswith(".nii.gz")):
            im = load(zi, n)
            mask = load(zm, n.replace("Images/", "Masks/").replace(".nii.gz", "_gt.nii.gz"))
            stem, (h, w) = Path(n).name.split(".")[0], im.shape[1:]
            for t in np.linspace(0, len(im) - 1, CAMUS_FRAMES).round().astype(int):
                dst = frames / f"{stem}_f{t:02d}.png"
                if not dst.exists():
                    cv2.imwrite(str(dst), im[t].astype(np.uint8))
                boxes = []
                for i in range(len(names)):
                    ys, xs = np.where(mask[t] == i + 1)
                    if len(xs) and (box := xyxy_to_yolo(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, w, h)):
                        boxes.append((i, *box))
                by_file[dst] = boxes
    return carve_val(by_file, group=lambda p: p.name.split("_")[0]), names


def real_colon(root: Path) -> tuple[dict, list]:
    """Colonoscopy polyp boxes over the frames fetch_real_colon.py has already sampled.

    Boxes come from the per-video annotation tarballs rather than the extracted XMLs, because one XML per frame is 2.7M
    files for data worth a few thousand rows. Splitting is by video, which is one patient on one scope with one bowel
    prep, so no patient straddles the split and neither does a lesion.
    """
    want = {p.stem: p for p in (root / "_frames").glob("*.jpg")}
    have = {k.rsplit("_", 1)[0] for k in want}  # 14 of the 60 videos hold no lesion, so they are never fetched
    by_file = {}
    for t in sorted((root / "_anno").glob("*_annotations.tar.gz")):
        if t.name.split("_")[0] not in have:
            continue
        with tarfile.open(t, "r:gz") as tf:
            for m in tf:
                if not (p := want.get(Path(m.name).stem)):
                    continue
                x = ET.fromstring(tf.extractfile(m).read())
                w, h = (int(x.findtext(f"size/{k}")) for k in ("width", "height"))
                by_file[p] = [
                    (0, *box)
                    for b in x.iter("bndbox")
                    if (box := xyxy_to_yolo(*(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")), w, h))
                ]
    # 46 videos, so the default stride would leave val holding three of them
    return carve_val(by_file, group=lambda p: p.name.split("_")[0], every=5), ["polyp"]


def vision(root: Path) -> tuple[dict, list]:
    """VISION industrial defect. Fourteen sub-datasets, each its own namespace, so classes are prefixed.

    The source splits its small labelled pool near evenly between train and val and puts the bulk of its images in an
    unlabelled ``inference`` split. Both labelled splits are merged and re-carved here, because holding back half the
    boxes for a val number nobody reports would waste them.
    """
    merged: dict = {}
    names: list[str] = []
    for tar in sorted(root.glob("*.tar.gz")):
        sub = tar.name.split(".")[0]
        d = extract(tar, root / f"_src_{sub}") / sub
        anno = {s: d / s / "_annotations.coco.json" for s in ("train", "val")}  # inference/ carries no boxes
        r, n = coco_adapter(anno, {s: d / s for s in anno})
        offset = len(names)
        names += [f"{sub}-{x}" for x in n]
        for by_file in r.values():
            merged.update({p: [(c + offset, *b) for c, *b in v] for p, v in by_file.items()})
    return carve_val(merged), names


def loaf(root: Path) -> tuple[dict, list]:
    """Fisheye person detection. Subdir names inside the image tar are unverified, so images match on basename."""
    a = extract(root / "loaf_annotations.tar", root / "_src_anno") / "annotations/resolution_1k"
    im = extract(root / "images/resolution_1k.tar.bz2", root / "_src_img")
    index = {p.name: p for p in im.rglob("*.jpg")}
    rows, names = coco_adapter({s: a / f"instances_{s}.json" for s in ("train", "val", "test")}, Path("."))
    return {s: {index[p.name]: v for p, v in r.items() if p.name in index} for s, r in rows.items()}, names


def hixray(root: Path) -> tuple[dict, list]:
    """Security X-ray. Boxes are absolute xyxy in a plain txt and image width varies, so sizes are read per image."""
    d = extract(root / "hixray.zip", root / "_src")
    names = ["Cosmetic", "Laptop", "Mobile_Phone", "Nonmetallic_Lighter"]  # source ships no names file
    names += ["Portable_Charger_1", "Portable_Charger_2", "Tablet", "Water"]
    rows: dict[str, dict] = {}
    for split in ("train", "test"):
        by_file = {}
        for img in sorted((d / split / f"{split}_image").glob("*.jpg")):
            w, h = exif_size(Image.open(img))  # header read, the source txt carries no image size
            boxes = []
            lb = d / split / f"{split}_annotation" / f"{img.stem}.txt"
            for line in lb.read_text().splitlines() if lb.exists() else []:
                f = line.split()
                if len(f) >= 6 and (box := xyxy_to_yolo(*(float(v) for v in f[2:6]), w, h)):
                    boxes.append((names.index(f[1]), *box))
            by_file[img] = boxes
        rows[split] = by_file
    return rows, names


def clcxray(root: Path) -> tuple[dict, list]:
    """Security X-ray, already YOLO. The image dir is capitalized in the source."""
    d = extract(root / "clcxray-yolo.zip", root / "_src") / "CLCXray"
    names = ["blade", "scissors", "knife", "dagger", "SwissArmyKnife", "PlasticBottle"]  # from datasetCD.yaml
    names += ["Cans", "VacuumCup", "GlassBottle", "CartonDrinks", "Tin", "SprayCans"]
    return yolo_adapter({s: d / s for s in ("train", "val", "test")}, names, image_dir="Images")


def esad(root: Path) -> tuple[dict, list]:
    """Surgeon action detection, already YOLO with labels interleaved beside the images."""
    raw = root / "OpenDataLab___ESAD/raw"
    tr = extract(raw / "train.zip", root / "_src_train")
    va = extract(raw / "val.zip", root / "_src_val")
    names = (tr / "train/obj.names").read_text().split()
    dirs = {"train": [tr / "train/set1", tr / "train/set2"], "val": [va / "val/obj"]}
    return yolo_adapter(dirs, names, image_dir="", label_dir="")


def spinexr(root: Path) -> tuple[dict, list]:
    """Spine X-ray lesions, already YOLO over a flat image dir with txt split lists."""
    d = extract(root / "vindr-spinexr-full-png.zip", root / "_src")
    names = ["Osteophytes", "Disc space narrowing", "Other lesions"]  # source ships no names file
    names += ["Surgical implant", "Foraminal stenosis", "Vertebral collapse"]
    rows: dict[str, dict] = {}
    for split in ("train", "val"):
        by_file = {}
        for line in (d / f"{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            img = d / line.strip().removeprefix("./")
            by_file[img] = read_yolo_labels(d / "labels" / f"{img.stem}.txt")
        rows[split] = by_file
    return rows, names


def ead(root: Path) -> tuple[dict, list]:
    """Endoscopy artefacts, already YOLO in one flat dir. The source ships no split, so a 5% val is carved out."""
    d = extract(root / "ead2020.zip", root / "_src") / "EAD2020_train"
    names = (d / "class_list.txt").read_text().split()
    imgs = sorted((d / "allDetection_training/bbox_images").glob("*.jpg"))
    return carve_val({p: read_yolo_labels(p.with_suffix(".txt")) for p in imgs}), names


def bbbc041(root: Path) -> tuple[dict, list]:
    """Malaria blood smears. Native boxes stored row/col, so r is y and c is x."""
    d = extract(root / "malaria.zip", root / "_src") / "malaria"
    names = [
        "red blood cell",
        "trophozoite",
        "ring",
        "schizont",
        "gametocyte",
        "leukocyte",
        "difficult",
    ]  # union, leukocyte is train-only
    rows: dict[str, dict] = {}
    for split, fn in (("train", "training.json"), ("test", "test.json")):
        by_file = {}
        for rec in json.loads((d / fn).read_text()):
            shape = rec["image"]["shape"]
            h, w = shape["r"], shape["c"]
            boxes = []
            for o in rec["objects"]:
                lo, hi = o["bounding_box"]["minimum"], o["bounding_box"]["maximum"]
                if box := xyxy_to_yolo(lo["c"], lo["r"], hi["c"], hi["r"], w, h):
                    boxes.append((names.index(o["category"]), *box))
            by_file[d / rec["image"]["pathname"].lstrip("/")] = boxes
        rows[split] = by_file
    return rows, names


DRONE_BORDER = 100  # every DroneVehicle image carries this many pure-white pixels on all four sides
DRONE_NAMES = ["car", "truck", "bus", "van", "freight_car"]


def _drone_name(raw: str) -> str:
    """Normalize a DroneVehicle class string, whose source spells freight car three different ways."""
    s = raw.strip().lower().replace("_", " ").replace("*", "")
    return {"feright car": "freight_car", "freight car": "freight_car"}.get(s, s)


def _drone_corners(ob: ET.Element) -> np.ndarray | None:
    """Return the four corners of a DroneVehicle object, or None when it carries no extent.

    Most objects ship an oriented ``polygon``, 65,184 ship an axis-aligned ``bndbox`` instead, and 75 across the whole
    source are a bare ``point`` that no box can be recovered from.
    """
    if (poly := ob.find("polygon")) is not None:
        return np.array([[float(poly.findtext(f"x{i}")), float(poly.findtext(f"y{i}"))] for i in range(1, 5)])
    if (bb := ob.find("bndbox")) is not None:
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return None


def dronevehicle(root: Path) -> tuple[dict, list]:
    """Drone RGB and infrared vehicles in VOC xml, mostly oriented boxes with an axis-aligned minority.

    Both modalities are kept in one dataset because they share the five classes and the same annotation semantics, and
    each frame carries its own boxes, so the infrared view keeps the extra vehicles that are invisible in RGB.

    The source pads every image with a 100 pixel white border, which is 45% of the stored pixels, so images are cropped
    and rewritten rather than symlinked and the annotations are shifted to match.
    """
    rows: dict[str, dict] = {}
    for split, zname in (("train", "train.zip"), ("val", "val.zip"), ("test", "test.zip")):
        d = extract(root / zname, root / f"_src_{split}") / split
        by_file = {}
        for mod, suffix in (("rgb", ""), ("ir", "r")):
            img_dir, lab_dir = d / f"{split}img{suffix}", d / f"{split}label{suffix}"
            if not img_dir.is_dir():
                continue
            crop_dir = root / "_crop" / f"{split}_{mod}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            for xml in sorted(lab_dir.glob("*.xml")):
                src = img_dir / f"{xml.stem}.jpg"
                if not src.exists():
                    continue
                dst = crop_dir / f"{xml.stem}.jpg"
                if dst.exists():
                    w, h = exif_size(Image.open(dst))  # header read, matching every other adapter here
                else:
                    b = DRONE_BORDER
                    crop = cv2.imread(str(src))[b:-b, b:-b]
                    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    h, w = crop.shape[:2]
                boxes = []
                for ob in ET.parse(xml).getroot().iter("object"):
                    name = _drone_name(ob.findtext("name", ""))
                    pts = _drone_corners(ob)
                    if pts is None or name not in DRONE_NAMES:
                        continue
                    pts -= DRONE_BORDER  # annotations are stored in the padded frame
                    if not (0 <= pts[:, 0].mean() < w and 0 <= pts[:, 1].mean() < h):
                        continue  # a box whose centre left the crop was annotated inside the padding
                    pts[:, 0] = pts[:, 0].clip(0, w) / w
                    pts[:, 1] = pts[:, 1].clip(0, h) / h
                    boxes.append((DRONE_NAMES.index(name), *pts.ravel()))
                by_file[dst] = boxes
        rows[split] = by_file
    return rows, DRONE_NAMES


def locount(root: Path) -> tuple[dict, list]:
    """Retail shelf products, grouped boxes in a flat csv-style txt.

    Rows are ``x1,y1,x2,y2,ClassName,count``. The count is how many identical products the box groups, which detection
    does not model, so it is dropped and the box is kept as one instance. Image sizes are not in the txt.
    """
    tar_root = root / "OpenDataLab___Locount/raw"
    src = extract(tar_root / "Locount.tar.gz", root / "_src") / "Locount"
    parsed: dict[str, list] = {}
    vocab = set()
    for split, tag in (("train", "Train"), ("test", "Test")):
        img_dir = extract(src / f"Locount_Images{tag}.zip", root / f"_img_{split}") / f"Locount_Images{tag}"
        gt_dir = extract(src / f"Locount_GtTxts{tag}.zip", root / f"_gt_{split}") / f"Locount_GtTxts{tag}"
        by_stem = {p.stem: p for p in img_dir.iterdir()}  # one listing, not one per label file
        recs = []
        for gt in sorted(gt_dir.glob("*.txt")):
            img = by_stem.get(gt.stem)
            if img is None:
                continue
            raw = []
            for line in gt.read_text().splitlines():
                f = line.rsplit(",", 2)  # class names contain no comma but do contain spaces
                if len(f) != 3:
                    continue
                xy = f[0].split(",")
                if len(xy) != 4:
                    continue
                raw.append((f[1].strip(), [float(v) for v in xy]))
                vocab.add(f[1].strip())
            recs.append((img, raw))
        parsed[split] = recs

    names = sorted(vocab)
    rows: dict[str, dict] = {}
    for split, recs in parsed.items():
        by_file = {}
        for img, raw in recs:
            w, h = exif_size(Image.open(img))
            boxes = [(names.index(n), *b) for n, xy in raw if (b := xyxy_to_yolo(*xy, w, h))]
            by_file[img] = boxes
        rows[split] = by_file
    return rows, names


def tufts(root: Path) -> tuple[dict, list]:
    """Dental panoramic teeth. Boxes are [y1, x1, y2, x2] and External ID lowercases the real .JPG extension."""
    d = extract(root / "the-tufts-dental-database-2022.zip", root / "_src")
    recs = json.loads((d / "Segmentation/Segmentation/teeth_bbox.json").read_text())
    names = [str(i) for i in range(1, 33)] + [chr(c) for c in range(ord("A"), ord("T") + 1)]
    by_file = {}
    for rec in recs:
        img = d / "Radiographs/Radiographs" / f"{Path(rec['External ID']).stem}.JPG"
        w, h = exif_size(Image.open(img))
        boxes = []
        for o in rec["Label"]["objects"]:
            y1, x1, y2, x2 = o["bounding box"]
            if o["title"] in names and (box := xyxy_to_yolo(x1, y1, x2, y2, w, h)):
                boxes.append((names.index(o["title"]), *box))
        by_file[img] = boxes
    return carve_val(by_file), names


# dataset key: output subdirectory, adapter, which split becomes val. Sources with no val donate their test split.
ADAPTERS = {
    "sardet": ("sardet-100k", sardet, "val"),
    "doclaynet": ("doclaynet", doclaynet, "val"),
    "cct20": ("cct20", cct20, "cis_val"),
    "flir": ("flir-adas-v2", flir, "val"),
    "duo": ("duo", duo, "test"),
    "camus": ("camus", camus, "val"),
    "real_colon": ("real-colon", real_colon, "val"),
    "vision": ("vision-datasets", vision, "val"),
    "loaf": ("loaf", loaf, "val"),
    "hixray": ("hixray", hixray, "test"),
    "clcxray": ("clcxray", clcxray, "val"),
    "esad": ("esad", esad, "val"),
    "spinexr": ("vindr-spinexr-png", spinexr, "val"),
    "ead": ("ead-endoscopy", ead, "val"),
    "bbbc041": ("bbbc041-malaria", bbbc041, "test"),
    "tufts": ("dental-tufts", tufts, "val"),
    "dronevehicle": ("dronevehicle", dronevehicle, "val"),
    "locount": ("locount", locount, "test"),
}


def main() -> None:
    """Convert the named datasets, or all of them, and render previews for each."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datasets", nargs="*", help=f"any of: {' '.join(ADAPTERS)}")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--visualize", type=int, default=6, help="previews per split, always on since they catch skew")
    a = p.parse_args()

    if a.list:
        print(" ".join(ADAPTERS))
        return
    for key in ADAPTERS if a.all else a.datasets:
        subdir, fn, val = ADAPTERS[key]
        print(f"=== {key}")
        rows, names = fn(a.root / subdir)
        stats = emit(a.root / subdir, rows, names, val, a.visualize)
        print(f"  nc={len(names)}  " + "  ".join(f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
