# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stream the sampled REAL-Colon frames onto disk, ready for the ``real_colon`` adapter in prep_domain_det.py.

REAL-Colon ships 60 colonoscopy videos as 946 GB of per-video frame tarballs, one JPEG per frame at ~30 fps. Only 46 of
those videos contain a boxed lesion, and they hold 132 lesions between them, so frame count is not what limits variety.
Frames are piped through tar and only the sampled ones are written, which keeps peak disk at the size of the output
rather than the size of the archive. Members are stored unsorted, so tar still reads each archive end to end.

Sampling is one frame per second, capped per video. Neighbouring frames at 30 fps are the same polyp a few milliseconds
apart, and the cap stops a video the endoscopist lingered over from outweighing one passed quickly. Background frames
are sampled too, so the source is not uniformly polyp-present.

This is a separate entrypoint rather than an adapter body because a multi-hour 700 GB stream has no business running
under ``prep_domain_det.py --all``. Conversion, splitting, labels and previews all belong to that script.

Reference: Biffi et al., REAL-Colon, Scientific Data 2024. CC BY 4.0.

Usage:
    python fetch_real_colon.py
    python fetch_real_colon.py --videos 001-001 001-003
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ultralytics.utils.downloads import safe_download

ARTICLE = 22202866  # Figshare+ REAL-Colon
VIDEOS = 60  # videos in the release, asserted so a short file list cannot pass as a corpus with fewer lesions
ROOT = Path("/data/shared-datasets/domain-det/real-colon")
POS_CAP = 300  # sampled frames per video holding a lesion
NEG_CAP = 150  # sampled frames per video holding none
STREAMS = 3  # concurrent downloads, measured at 1.65x one connection before the link stops giving more


def files(anno: Path) -> list:
    """Return the article file list, cached so a rerun does not re-query.

    A short list is the dangerous failure here, because a missing video reads as a video with no lesion rather than as
    an error. The count is checked on the cached path too, so a truncated cache written by an earlier run cannot
    survive.
    """
    cache = anno / "files.json"
    if not cache.exists():
        anno.mkdir(parents=True, exist_ok=True)
        # page_size covers every file, since the default page returns 10 and drops the rest without saying so
        url = f"https://api.figshare.com/v2/articles/{ARTICLE}/files?page_size=200"
        safe_download(url, file=cache, unzip=False, progress=False)
    out = json.loads(cache.read_bytes())
    n = sum(x["name"].endswith("_frames.tar.gz") for x in out)
    assert n == VIDEOS, f"{cache} lists {n} of {VIDEOS} videos, delete it and rerun"
    return out


def lesion_frames(anno: Path) -> dict:
    """Map each video to its frame count and the frame indices where a lesion is boxed.

    Read straight out of the annotation tarballs. Extracting them would land 2.7M files on shared storage for data that
    is only ever needed as a list of integers.
    """
    out = {}
    for t in sorted(anno.glob("*_annotations.tar.gz")):
        video, pos, total = t.name.split("_")[0], [], 0
        with tarfile.open(t, "r:gz") as tf:
            for m in tf:
                if not m.name.endswith(".xml"):
                    continue
                total += 1
                if b"<object>" in tf.extractfile(m).read():
                    pos.append(int(Path(m.name).stem.split("_")[-1]))
        out[video] = {"total": total, "pos": sorted(pos)}
        print(f"  {video}: {len(pos):,} of {total:,} frames hold a lesion", flush=True)
    return out


def sample(idx: list, fps: float, cap: int) -> list:
    """Thin a frame index list to roughly one per second, then to at most ``cap`` evenly spaced entries."""
    thin = idx[:: max(round(fps), 1)]
    if len(thin) <= cap:
        return thin
    return [thin[round(i * (len(thin) - 1) / (cap - 1))] for i in range(cap)]


def main() -> None:
    """Fetch the annotations, decide the sample, then stream each video's chosen frames into _frames."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--videos", nargs="*", help="restrict to these video ids")
    a = p.parse_args()

    anno, frames = a.root / "_anno", a.root / "_frames"
    frames.mkdir(parents=True, exist_ok=True)
    fl = files(anno)
    for x in fl:  # annotations and metadata together are 70 MB, so always take all of them
        if x["name"].endswith(("_annotations.tar.gz", ".csv", ".md")):
            d = anno / x["name"]
            if not d.exists() or d.stat().st_size != x["size"]:
                safe_download(x["download_url"], file=d, unzip=False)

    cache = anno / "lesion_frames.json"
    if not cache.exists():
        cache.write_text(json.dumps(lesion_frames(anno)))
    lf = json.loads(cache.read_bytes())

    # fps varies per video, so the one-per-second stride has to be read rather than assumed
    rows = csv.DictReader((anno / "video_info.csv").read_text().splitlines())
    fps = {r["unique_video_name"]: float(r["fps"]) for r in rows}

    vids = [v for v in sorted(lf) if lf[v]["pos"] and (not a.videos or v in a.videos)]
    url = {x["name"]: x["download_url"] for x in fl}
    print(f"{len(vids)} videos hold a lesion, streaming each one")

    def pull(v: str) -> str | None:
        """Stream one video, keeping only its sampled frames, and name it back if it did not arrive whole.

        Frames on disk are the completion test rather than the exit status. tar reports failure when any requested
        member is absent yet still extracts the rest, and a run killed midway leaves a partial video that a
        presence check would read as done.
        """
        pos = set(lf[v]["pos"])
        wanted = set(sample(lf[v]["pos"], fps[v], POS_CAP)) | set(
            sample([n for n in range(lf[v]["total"]) if n not in pos], fps[v], NEG_CAP)
        )
        if len(list(frames.glob(f"{v}_*.jpg"))) >= len(wanted):
            print(f"  {v}: already fetched, skipping", flush=True)
            return None
        names = "\n".join(f"{v}_frames/{v}_{n}.jpg" for n in sorted(wanted))
        # -f so an HTTP error is a failure rather than an error body piped into tar, and pipefail so it reaches us
        cmd = (
            f"set -o pipefail; curl -fsSL '{url[f'{v}_frames.tar.gz']}' | tar xz -C {frames} --strip-components=1 -T -"
        )
        r = subprocess.run(["bash", "-c", cmd], input=names, text=True, capture_output=True, check=False)
        got = len(list(frames.glob(f"{v}_*.jpg")))
        short = f"  SHORT {r.stderr.strip()[:90]}" if got < len(wanted) else ""
        print(f"  {v}: {got} of {len(wanted)} frames{short}", flush=True)
        return v if got < len(wanted) else None

    # One connection does not saturate the link, and the gzip layer is near-stored over JPEG, so these wait on the wire
    # rather than on a core. Videos write disjoint filenames into one directory, so they need no coordination.
    with ThreadPoolExecutor(max_workers=STREAMS) as ex:
        short = [v for v in ex.map(pull, vids) if v]

    # Raised after every video has had its turn, so one bad source costs a rerun of that video rather than the run
    assert not short, f"{len(short)} videos came down short, rerun to finish them: {short}"
    print(f"done, {len(list(frames.glob('*.jpg'))):,} frames in {frames}")
    print("now run: python prep_domain_det.py real_colon")


if __name__ == "__main__":
    main()
