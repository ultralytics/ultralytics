"""Mirror the ``ul33``-tagged Ultralytics Platform datasets to local/NFS YOLO format.

Per dataset (selected by the platform ``ul33`` tag, so the untagged ``cicis`` test set is excluded):
    1. Skip if the final ``data.yaml`` already exists (unless ``--force``).
    2. Resolve ``ul://fatih/datasets/<slug>`` to a signed NDJSON export via ``check_file``.
    3. Force the NDJSON stem to ``<slug>`` so ``convert_ndjson_to_yolo`` names the output folder by slug.
    4. ``convert_ndjson_to_yolo`` downloads images and writes ``images/{split}`` + ``labels/{split}`` + ``data.yaml``.
    5. Verify realized per-split image counts against the platform ``splits`` metadata.

Requires ``ULTRALYTICS_API_KEY`` (read from the env or the repo ``.env``); it gates both the list API and the
``ul://`` resolver. Run the cap step (``cap_train_subset.py``) afterwards; re-running prep with ``--force`` rewrites
``data.yaml`` and undoes the cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import urllib.request
from pathlib import Path

from ultralytics.data.converter import convert_ndjson_to_yolo
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_file

PLATFORM = "https://platform.ultralytics.com"
USERNAME = "fatih"
TAG = "ul33"
SPLITS = ("train", "val", "test")

# Fallback slug list (2026-06-26) used only when the list API is unavailable; the live tag query is preferred.
STATIC_SLUGS = [
    "annopage-doc-elements",
    "basketball-hoop-detection",
    "ceymo",
    "cheng-brain-tumor-mri",
    "ddr-lesion",
    "deeppcb",
    "dentex",
    "dspcbsd",
    "egohands-egocentric",
    "ena24",
    "fisheye8k-traffic",
    "floorplancad-2",
    "grazpedwri-dx",
    "gwhd-2021",
    "hit-uav",
    "iedxray",
    "kvasir-seg-polyp",
    "llvip",
    "plantdoc-2",
    "rdd2022",
    "rpc-retail-checkout",
    "ruod-underwater",
    "seadronessee-2",
    "sku110k-2",
    "ssdd-sar-ship",
    "taco-litter",
    "tbx11k",
    "thewayup-climbing-holds",
    "tn5000-thyroid-us",
    "txl-pbc-blood-cells",
    "uec-school-lunch-food-2",
    "visdrone",
    "webui",
]


def load_api_key(repo_root: Path) -> str:
    """Return the platform key from the env, repo ``.env``, or ultralytics SETTINGS; export it for the ``ul://``
    resolver.
    """
    key = os.environ.get("ULTRALYTICS_API_KEY")
    if not key and (env := repo_root / ".env").is_file():
        for line in env.read_text().splitlines():
            s = line.strip()
            if s.startswith("ULTRALYTICS_API_KEY") and "=" in s:
                key = s.split("=", 1)[1].strip().strip('"').strip("'")
                break
    key = key or SETTINGS.get("api_key")
    if not key:
        raise SystemExit("set ULTRALYTICS_API_KEY in the env, repo .env, or via `yolo settings api_key=...`")
    os.environ["ULTRALYTICS_API_KEY"] = key
    return key


def list_ul33(api_key: str) -> dict[str, dict]:
    """Return ``{slug: splits}`` for every ``ul33``-tagged dataset owned by ``USERNAME`` via the platform REST API."""
    url = f"{PLATFORM}/api/datasets?username={USERNAME}&limit=1000"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)
    arr = body if isinstance(body, list) else (body.get("datasets") or body.get("data") or [])
    return {d["slug"]: d.get("splits") or {} for d in arr if isinstance(d.get("tags"), list) and TAG in d["tags"]}


def count_images(images_root: Path) -> dict[str, int]:
    """Return realized per-split image counts under ``images_root/{split}``."""
    counts = {}
    for split in SPLITS:
        d = images_root / split
        counts[split] = sum(1 for p in d.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS) if d.is_dir() else 0
    return counts


def precap_train(ndjson_path: Path, cap: int, seed: int) -> tuple[int, int]:
    """Trim the NDJSON's train records to at most ``cap`` (seed-stable), keeping all val/test records and metadata.

    Sorts train records by ``file`` then draws with ``Random(seed)`` so the kept set matches what ``cap_train_subset``
    would select on the full download (train images share the ``images/train/`` prefix, so file-order == path-order).
    Avoids downloading the train images that the permanent cap would discard. Returns ``(kept_train, original_train)``.
    """
    lines = [json.loads(s) for s in ndjson_path.read_text().splitlines() if s.strip()]
    meta, records = lines[0], lines[1:]
    train = [r for r in records if r.get("split") == "train"]
    other = [r for r in records if r.get("split") != "train"]
    kept = sorted(train, key=lambda r: r["file"])
    if len(kept) > cap:
        kept = random.Random(seed).sample(kept, cap)
    with ndjson_path.open("w") as f:
        f.write("\n".join(json.dumps(r) for r in [meta, *other, *kept]) + "\n")
    return len(kept), len(train)


def prepare_one(slug: str, root: Path, precap: int, seed: int) -> dict[str, int]:
    """Pull ``ul://fatih/datasets/<slug>`` into ``root/<slug>`` as YOLO + data.yaml; return per-split image counts."""
    with tempfile.TemporaryDirectory() as tmp:
        ndjson = Path(check_file(f"ul://{USERNAME}/datasets/{slug}", download_dir=tmp))
        slug_ndjson = Path(tmp) / f"{slug}.ndjson"  # force stem so convert names the folder by slug
        if ndjson != slug_ndjson:
            ndjson.replace(slug_ndjson)
        if precap:
            precap_train(slug_ndjson, precap, seed)
        yaml_path = asyncio.run(convert_ndjson_to_yolo(slug_ndjson, output_path=root))
    return count_images(yaml_path.parent / "images")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the prep script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("/data/shared-datasets/ul33"))
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated slug filter")
    parser.add_argument("--force", action="store_true", help="re-pull even if data.yaml exists")
    parser.add_argument("--precap-train", type=int, default=1000, help="cap train records before download (0 = full)")
    parser.add_argument("--seed", type=int, default=1337, help="seed for the pre-cap train sample")
    return parser.parse_args()


def main() -> None:
    """Prepare the ``ul33``-tagged platform datasets under ``args.root``."""
    args = parse_args()
    api_key = load_api_key(Path(__file__).resolve().parent)
    try:
        meta = list_ul33(api_key)
    except Exception as e:  # API unreachable: fall back to the static slug list (no count verification)
        print(f"list API failed ({e}); using static {len(STATIC_SLUGS)}-slug fallback")
        meta = {s: {} for s in STATIC_SLUGS}

    slugs = sorted(meta)
    if args.datasets:
        wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
        if unknown := wanted - set(slugs):
            raise SystemExit(f"unknown slugs: {sorted(unknown)}")
        slugs = [s for s in slugs if s in wanted]

    args.root.mkdir(parents=True, exist_ok=True)
    total = len(slugs)
    summary: list[tuple[str, dict[str, int]]] = []
    failures: list[tuple[str, str]] = []
    for i, slug in enumerate(slugs, 1):
        ds = args.root / slug
        if (ds / "data.yaml").exists() and not args.force:  # skip the NFS tree walk on already-prepared datasets
            print(f"[{i}/{total}] {slug} ... SKIP (data.yaml exists)", flush=True)
            continue
        try:  # one dead-URL/empty dataset must not abort the multi-hour batch
            counts = prepare_one(slug, args.root, args.precap_train, args.seed)
        except Exception as e:
            print(f"[{i}/{total}] {slug} ... FAILED ({type(e).__name__}: {e})", flush=True)
            failures.append((slug, f"{type(e).__name__}: {e}"))
            continue
        expected = meta.get(slug, {})
        targets = {s: v for s in SPLITS if (v := expected.get(s))}
        if args.precap_train and "train" in targets:  # train is intentionally pre-capped; val/test stay full
            targets["train"] = min(args.precap_train, targets["train"])
        short = [f"{s} {counts[s]}/{t}" for s, t in targets.items() if counts[s] < t]
        flag = f"  SHORT: {', '.join(short)}" if short else ""
        print(f"[{i}/{total}] {slug} ... OK ({counts['train']}/{counts['val']}/{counts['test']}){flag}", flush=True)
        summary.append((slug, counts))

    print("\nslug, train, val, test")
    for slug, c in summary:
        print(f"{slug}, {c['train']}, {c['val']}, {c['test']}")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
