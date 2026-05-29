"""Market-1501 query/gallery wrapper for h8.

The heavy lifting (pid/camid parsing, file globbing, transforms) lives in
Ultralytics' ReidDataset; this module thinly wraps it to give the h8 stages a
flat iteration order with stable image_ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal


SPLITS = {"query", "gallery"}


@dataclass(frozen=True)
class MarketRecord:
    """One Market image. image_id is the filename stem (without extension)."""

    image_id: str
    split: Literal["query", "gallery"]
    pid: int
    camid: int
    img_path: str

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")


def iter_market_split(market_root: str, split: str, include_distractors: bool = False) -> Iterator[MarketRecord]:
    """Iterate Market-1501 query/ or bounding_box_test/ (=gallery).

    image_id is the filename stem (e.g. '0001_c1s1_001051_00').
    pid/camid follow Market's filename convention: <pid>_c<camid>s<seq>_<frame>_<bbox>.jpg

    By default, pid=-1 (distractors) is FILTERED OUT to match the standard
    Market-1501 protocol used by Ultralytics' ReidDataset (3,819 distractor
    files exist in bounding_box_test/; without filtering, the gallery has
    19,732 images vs the canonical 15,913). Set include_distractors=True to
    re-include them.

    pid=0 (Market noise IDs, ~2,798 files) is kept; downstream junk-filter
    in retrieval.py handles same-pid-same-cam removal.
    """
    if split == "query":
        subdir = Path(market_root) / "query"
    elif split == "gallery":
        subdir = Path(market_root) / "bounding_box_test"
    else:
        raise ValueError(f"split must be 'query' or 'gallery', got {split!r}")

    for p in sorted(subdir.glob("*.jpg")):
        stem = p.stem
        pid_str, rest = stem.split("_", 1)
        cam_str = rest.split("s", 1)[0]  # 'c1'
        pid = int(pid_str)
        camid = int(cam_str[1:])
        if pid < 0 and not include_distractors:
            continue
        yield MarketRecord(image_id=stem, split=split, pid=pid, camid=camid, img_path=str(p))
