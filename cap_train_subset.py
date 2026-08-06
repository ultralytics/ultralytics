"""Reproducibly cap a YOLO dataset's train split to at most 1000 images (seed 1337).

Non-destructive: samples the full ``images/train`` split, writes ``<dataset>/subset/train_1000.txt`` (absolute image
paths), and rewrites the ``data.yaml`` ``train:`` key to point at that file. ``val``/``test`` are untouched and every
image stays on disk, so the cap is reversible by restoring ``train: images/train``. Sampling is ``sorted()`` then a
``Random(1337)`` draw, so the selection is identical across hosts and runs. A split already at or below the cap is kept
whole (idempotent no-op). The cap always re-samples the full ``images/train`` split, so re-running after a cap (when
``train:`` already points at the subset file) reproduces the same selection rather than shrinking it.

Pass one or more ``data.yaml`` paths, or ``--list ul33.txt`` to cap every dataset in a list file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import YAML

SEED = 1337
CAP = 1000


def full_train_images(d: dict, root: Path) -> list[Path]:
    """Return the sorted full train image paths from the canonical ``images/train`` split.

    Always resolves ``<root>/images/train`` (as written by ``convert_ndjson_to_yolo``) so re-capping starts from the
    full split even after ``data.yaml`` ``train:`` was rewritten to the subset file; falls back to the ``train:`` value
    when that directory is absent.
    """
    train_dir = root / "images" / "train"
    if not train_dir.is_dir() and (train := d.get("train")):
        p = Path(train)
        train_dir = p if p.is_absolute() else root / p
    return sorted(p for p in train_dir.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)


def audit(keep: list[Path], names: dict) -> list[str]:
    """Return human-readable warnings (missing classes, out-of-range boxes) for the kept subset."""
    warnings = []
    classes_present, bad_boxes = set(), 0
    for label in img2label_paths(str(p) for p in keep):
        label = Path(label)
        if not label.is_file():
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            classes_present.add(int(float(parts[0])))
            if any(not 0.0 <= float(v) <= 1.0 for v in parts[1:5]):
                bad_boxes += 1
    if names and (missing := set(names) - classes_present):
        warnings.append(
            f"{len(missing)}/{len(names)} classes absent from the {len(keep)}-image subset: {sorted(missing)}"
        )
    if bad_boxes:
        warnings.append(f"{bad_boxes} out-of-range box coordinates in the subset")
    return warnings


def cap_one(data_yaml: Path, cap: int, seed: int) -> None:
    """Write the seed-stable train subset file and repoint ``data.yaml`` ``train:`` at it."""
    d = YAML.load(data_yaml)
    root = Path(d.get("path", data_yaml.parent))
    imgs = full_train_images(d, root)
    n = len(imgs)
    if not n:
        raise SystemExit(f"{data_yaml}: no train images found")
    # imgs is already sorted; re-sort only the random draw so the subset file is order-stable across runs
    keep = imgs if n <= cap else sorted(random.Random(seed).sample(imgs, cap))

    subset_dir = root / "subset"
    subset_dir.mkdir(exist_ok=True)
    txt = subset_dir / f"train_{cap}.txt"
    txt.write_text("\n".join(str(p.resolve()) for p in keep) + "\n")

    d["train"] = str(txt.resolve())
    YAML.save(data_yaml, d)

    tag = "kept whole (<= cap)" if n <= cap else f"capped {n} -> {cap}"
    print(f"{root.name}: {tag} -> {txt}")
    for w in audit(keep, d.get("names") or {}):
        print(f"  WARN {w}")


def resolve_targets(targets: list[str], list_file: str | None) -> list[Path]:
    """Return data.yaml paths from positional args plus an optional list file (one path per line, #-comments ok)."""
    yamls = [Path(t).expanduser().resolve() for t in targets]
    if list_file:
        for line in Path(list_file).read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                yamls.append(Path(s).expanduser().resolve())
    if missing := [y for y in yamls if not y.is_file()]:
        raise SystemExit(f"missing data.yaml files: {missing}")
    if not yamls:
        raise SystemExit("no data.yaml targets given")
    return sorted(set(yamls))


def main() -> None:
    """Cap the train split of each target ``data.yaml`` to ``--cap`` images using ``--seed``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help="data.yaml path(s)")
    parser.add_argument("--list", dest="list_file", default=None, help="file of data.yaml paths (e.g. ul33.txt)")
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for data_yaml in resolve_targets(args.targets, args.list_file):
        cap_one(data_yaml, args.cap, args.seed)


if __name__ == "__main__":
    main()
