import argparse
import json
from collections import defaultdict
from pathlib import Path

from lvis import LVIS, LVISResults, LVISEval
from ultralytics.utils import LOGGER

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("annotations_json", type=Path)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--type", default="bbox", choices=["segm", "bbox"])
    parser.add_argument("--dets-per-cat", default=10000, type=int)
    parser.add_argument("--ious", nargs="*", type=float)

    args = parser.parse_args()

    assert args.dets_per_cat > 0
    with open(args.results_json, "r") as f:
        results = json.load(f)

    by_cat = defaultdict(list)
    for ann in results:
        by_cat[ann["category_id"]].append(ann)
    results = []
    topk = args.dets_per_cat
    missing_dets_cats = set()
    for cat, cat_anns in by_cat.items():
        if len(cat_anns) < topk:
            missing_dets_cats.add(cat)
        results.extend(sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:topk])

    if args.type == "segm":
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask
        # area.  This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for x in results:
            x.pop("bbox", None)

    if missing_dets_cats:
        LOGGER.warning(
            f"\n===\n"
            f"{len(missing_dets_cats)} classes had less than {topk} detections!\n"
            f"Outputting {topk} detections for each class will improve AP further.\n"
            f"If using detectron2, please use the lvdevil/infer_topk.py script to "
            f"output a results file with {topk} detections for each class.\n"
            f"==="
        )

    gt = LVIS(args.annotations_json)
    results = LVISResults(gt, results, max_dets=-1)
    lvis_eval = LVISEval(gt, results, iou_type=args.type)
    params = lvis_eval.params
    params.max_dets = -1  # No limit on detections per image.
    if args.ious:
        params.iou_thrs = args.ious

    lvis_eval.run()
    lvis_eval.print_results()
    metrics = {k: v for k, v in lvis_eval.results.items() if k.startswith("AP")}
    LOGGER.info("copypaste: %s", ",".join(map(str, metrics.keys())))
    LOGGER.info(
        "copypaste: %s", ",".join(f"{v*100:.2f}" for v in metrics.values()),
    )


if __name__ == "__main__":
    # python tools/eval_fixed_ap.py ../datasets/lvis/annotations/lvis_v1_minival.json runs/detect/val2/predictions.json 
    main()