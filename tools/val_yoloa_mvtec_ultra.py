
import argparse
from pathlib import Path

from ultralytics import YOLOA

DEFAULT_DATA = "/data/shared-datasets/louis_data/MVTec-Ultra/v1"
# The project's decisive metrics, from expman/data/logbooks/yoloa_mvtec_ultra/PROTOCOL.md. Reported
# in full every time: a self-chosen subset turns a recall-for-precision trade into an apparent win.
DECISIVE = ("mAP50", "mAP50@0.25", "R50@0.25", "P50@0.25")


def main() -> None:
    """Parse arguments, run the catalogue evaluation, print the decisive metrics."""
    import ultralytics

    ap = argparse.ArgumentParser(description="Evaluate a YOLOA checkpoint on a MVTec-Ultra catalogue.")
    ap.add_argument("weights")
    ap.add_argument("--data", default=DEFAULT_DATA, help="a MVTec-Ultra root; its meta.yaml is the taxonomy")
    ap.add_argument("--groups", default="nature=structural", help="tag query, or a list of group ids")
    ap.add_argument("--out", required=True, help="directory for the csvs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", nargs="+", help='subset of "" none_ e2e_ e2e_none_; default runs all four')
    ap.add_argument("--no-e2e", action="store_true", help="skip the o2o passes (halves the time)")
    ap.add_argument("--verbose", action="store_true", help="print each product's validation table")
    ap.add_argument("--dump", help="write per-image predictions + GT here as jsonl; needs a single --passes")
    a = ap.parse_args()

    if old := sorted(p.name for p in Path(a.out).glob("*.csv")):
        raise SystemExit(f"{a.out} already holds {', '.join(old)} -- these are appended to; use a fresh --out")

    # Echoed into the log: without it the csvs are numbers of unknown provenance.
    print(f"ultralytics {ultralytics.__file__}")
    print(f"weights={a.weights}\ndata={a.data}\ngroups={a.groups}\npasses={a.passes or 'all four'}")
    r = YOLOA(a.weights).val_ood(
        a.data,
        groups=a.groups,
        e2e=not a.no_e2e,
        passes=a.passes,
        device=a.device,
        batch=a.batch,
        workers=a.workers,
        save_dir=a.out,
        verbose=a.verbose,
        dump=a.dump,
    )

    img, anom, inst = (int(r.pooled.get(k, 0)) for k in ("n_images", "n_anom_images", "n_instances"))
    print(f"\n{len(r.products)} products · {len(r.groups)} group rows · pass {r.decisive or 'heatmap'}")
    print(f"pooled over {img} images ({anom} anomalous, {inst} instances)")
    for k in DECISIVE:
        print(f"  pool_{r.decisive}{k:<12} {r.results_dict[k]:.4f}")
    print(f"\n-> {a.out}/  (ood_percat.csv · ood_pergroup.csv · pooled.csv)")


if __name__ == "__main__":
    main()
