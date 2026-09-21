# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Score pose and pose3d checkpoints on one common 2D ruler: COCO-17 joints with the standard OKS sigmas.

A pose3d model predicts 18 keypoints — COCO-17 plus a derived root — and every pose3d dataset YAML appends a
sigma of 0.107 for that root, the most forgiving value in the table, on a joint that is the midpoint of two
joints the model already predicts. Its `pose mAP` is therefore not comparable with a 2D pose model's, and the
label sets differ in width as well: 18x4 for the teacher's pseudo-labels, 17x3 for COCO's own annotation.

This drops the root joint and the depth channel from *both* sides — predictions and labels — and forces
`OKS_SIGMA` on both, so any checkpoint is scored by one metric on one set of images. Every model goes through
the same class, so a harness bug lands on all of them equally.

Usage:
    python research/src/eval_2d_parity.py --data <ruler.yaml> --models pose2d=<a.pt> E_refocal=<b.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import OKS_SIGMA

COCO_KPT = 17  # joints kept from either layout
KPT_LAYOUT = {51: (17, 3), 54: (18, 3), 68: (17, 4), 72: (18, 4)}  # flattened keypoint width -> (nkpt, ndim)


class Common2DValidator(PoseValidator):
    """Pose validator that scores every model on COCO-17 with `OKS_SIGMA`, whatever layout it was trained in."""

    def init_metrics(self, model) -> None:
        """Initialize pose metrics, then pin the keypoint layout and the sigmas to the COCO-17 ruler."""
        super().init_metrics(model)
        self.kpt_shape = [COCO_KPT, 3]  # what this validator reports on, not what the data carries
        self.sigma = OKS_SIGMA
        # Both rulers match `is_coco`, which forces `save_json` on and hands the metrics to pycocotools — but
        # only one of them ships an annotations/ dir, so the two columns would be scored by two evaluators.
        self.args.save_json = False

    def postprocess(self, preds):
        """Reshape predicted keypoints using the model's own layout, then truncate to COCO-17 (x, y, visible)."""
        preds = DetectionValidator.postprocess(self, preds)  # not PoseValidator's: it views with self.kpt_shape
        for pred in preds:
            extra = pred.pop("extra")
            nkpt, ndim = KPT_LAYOUT[extra.shape[1]]
            pred["keypoints"] = extra.view(-1, nkpt, ndim)[:, :COCO_KPT, :3].contiguous()
        return preds

    def _prepare_batch(self, si: int, batch):
        """Truncate ground-truth keypoints to COCO-17 (x, y, visible), so 18x4 labels score like 17x3 ones."""
        pbatch = super()._prepare_batch(si, batch)
        pbatch["keypoints"] = pbatch["keypoints"][:, :COCO_KPT, :3].contiguous()
        return pbatch


def evaluate(weights: str, data: str, device: str, imgsz: int, batch: int) -> dict[str, float]:
    """Validate one checkpoint on one ruler and return its 2D metrics."""
    validator = Common2DValidator(
        args=dict(
            model=weights,
            data=data,
            task="pose",  # the checkpoint may say pose3d; the dataloader and metrics here are 2D
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=False,
            save_json=False,
            verbose=False,
        )
    )
    stats = validator(model=weights)
    return {k: round(float(v), 4) for k, v in stats.items() if k.startswith("metrics/")}


def main(models: list[str], data: str, device: str, imgsz: int, batch: int, out: Path | None) -> None:
    """Evaluate every model on the ruler and print one markdown table."""
    rows = {}
    for spec in models:
        name, _, weights = spec.partition("=")
        if not weights:
            raise ValueError(f"--models takes name=path pairs, got {spec!r}")
        print(f"\n=== {name}: {weights} on {data} ===", flush=True)
        rows[name] = evaluate(weights, data, device, imgsz, batch)

    keys = ["metrics/mAP50-95(B)", "metrics/mAP50(P)", "metrics/mAP50-95(P)"]
    print(f"\n## {data}\n")
    print("| model | box mAP50-95 | pose mAP50 | pose mAP50-95 |")
    print("| --- | ---: | ---: | ---: |")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{r.get(k, float('nan')):.4f}" for k in keys) + " |")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"data": data, "imgsz": imgsz, "results": rows}, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name=path pairs, pose or pose3d checkpoints")
    parser.add_argument("--data", required=True, help="dataset YAML; its labels may be 17x3 or 18x4")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    main(**vars(parser.parse_args()))
