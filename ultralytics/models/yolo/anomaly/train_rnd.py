# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Internal R&D trainer for YOLO Anomaly.

Extends ``AnomalyTrainer`` with periodic cross-dataset OOD validation. This is
not intended for normal users; it exists for internal experiments where best.pt
selection should be driven by a macro-average OOD metric (e.g. MVTec mAP50).

The OOD loop is intentionally simple:
  * resolve a list of data yamls,
  * for each yaml, build a memory bank from its normal train split,
  * run ``YOLOAnomalyValidator`` on its val split,
  * macro-average the metrics and use the result as training fitness.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from ultralytics.models.yolo.anomaly.train import AnomalyTrainer
from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
from ultralytics.utils import LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

_MVTEC_ROOT_CANDIDATES = (
    "/data/shared-datasets/louis_data/MVTec-YOLO",
    "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
    "/home/laughing/codes/datasets/MVTec-YOLO",
)


def _normal_dir_from_yaml(yaml_path: str | Path) -> Path:
    """Resolve the directory of normal images referenced by a data yaml.

    Follows the MVTec convention: if ``<train>/good`` exists, use it; otherwise
    use ``<train>``.
    """
    data = YAML.load(yaml_path)
    root = Path(data.get("path", Path(yaml_path).parent))
    train = Path(data["train"])
    if not train.is_absolute():
        train = root / train
    good = train / "good"
    return good if good.is_dir() else train


@contextmanager
def _frozen_rng():
    """Run a block, then rewind every global RNG it touched.

    Used to keep ``ood_end2end``'s extra eval passes purely ADDITIVE. The memory-bank build
    draws on global RNG and is reproducible only by replaying the same draw sequence: three
    back-to-back ``_run_ood_eval`` calls on one model give prior-ON mAP10_50
    0.1957 / 0.2026 / 0.2019, while prior-OFF (bank disabled) stays at 0.195577 to six
    decimals — so it is the bank, not the detector. Without this, an extra pass would shift
    every later category's prior-ON number and ``ood_end2end=True`` runs would stop being
    comparable with ``False`` ones on the o2m columns they are supposed to share.
    """
    state = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        random.getstate(),
        np.random.get_state(),
    )
    try:
        yield
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2])
        np.random.set_state(state[3])


def _average_ood_rows(rows: list[dict]) -> dict[str, float]:
    """Macro-average over categories, ignoring NaNs and non-numeric fields (e.g. ``category``).

    Averages every numeric key present, so both the heatmap-prior metrics (``mAP50`` …) and the
    none-prior metrics (``none_mAP50`` …) are aggregated in one pass.
    """
    keys = [k for k in rows[0] if isinstance(rows[0].get(k), (int, float))]
    out: dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        out[key] = sum(vals) / len(vals) if vals else math.nan
    return out


class AnomalyRNDTrainer(AnomalyTrainer):
    """Trainer for YOLOAnomaly with periodic cross-dataset OOD validation."""

    def validate(self):
        """Run normal validation, then periodic OOD validation; fitness = OOD mAP50."""
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:  # non-rank-0 DDP workers get no metrics — mirror BaseTrainer.validate()
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if RANK not in (-1, 0) or self.ema is None:
            return metrics, fitness

        # In-domain on the OTHER branch. The domain val above inherits ``end2end`` from the model
        # yaml (True for yolo26), so ``metrics/*`` is the o2o branch; this adds the o2m mirror as
        # ``metrics/o2m_*``. Together with the ``e2e_*`` OOD keys below a run then reports the full
        # 2x2 — {in-domain, OOD} x {o2m, o2o} — instead of one cell of each.
        if getattr(self.args, "ood_end2end", False):
            metrics.update(self._domain_other_branch())

        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly", {})
        freq = int(v2_cfg.get("test_val_freq", 0))
        if freq <= 0 or (self.epoch + 1) % freq != 0:
            return metrics, fitness

        yamls = self._resolve_test_yamls(v2_cfg)
        if not yamls:
            return metrics, fitness

        ema_eval = deepcopy(self.ema.ema).eval()
        try:
            rows = self._run_ood_eval(ema_eval, yamls, v2_cfg)
            if rows:
                avg = _average_ood_rows(rows)
                avg_metrics = {f"ood/{k}": v for k, v in avg.items()}
                # Fitness keeps its historical definition -- mAP50 measured at conf>=0.25 -- so
                # best.pt selection stays comparable with every earlier yoloa run. The bare
                # ``mAP50`` key is now the threshold-free value and is reported, not selected on.
                fitness = float(avg.get("mAP50@0.25", avg["mAP50"]))
                metrics["fitness"] = fitness
                metrics.update(avg_metrics)
                self.best_fitness = max(self.best_fitness or -math.inf, fitness)
                LOGGER.info(
                    f"OOD eval @ep{self.epoch + 1}: [heatmap] mAP50={avg['mAP50']:.4f} "
                    f"(@.25={fitness:.4f}) mAP10={avg['mAP10']:.4f} "
                    f"| [none] mAP50={avg.get('none_mAP50', float('nan')):.4f} "
                    f"mAP10={avg.get('none_mAP10', float('nan')):.4f} "
                    f"(fitness=heatmap mAP50@0.25; bare keys are threshold-free; n={len(rows)} categories)"
                )
        finally:
            del ema_eval

        return metrics, fitness

    def _domain_other_branch(self) -> dict[str, float]:
        """Re-run in-domain val on whichever of o2m/o2o the main pass did not use.

        A fresh validator on the same ``test_loader`` rather than re-calling ``self.validator``:
        the trainer-mode call writes back into trainer state (loss, plots, speed), so invoking it
        twice per epoch would corrupt the numbers the first pass produced. ``trainer=None`` with an
        explicit model is the same pattern ``_run_ood_eval`` uses.

        Failures are swallowed — this is a reporting extra and must never take a run down.
        """
        from copy import copy

        try:
            args = copy(self.args)
            # The main pass took the model yaml's value (True for yolo26); take the other one.
            main_e2e = bool(getattr(unwrap_model(self.model), "end2end", True))
            args.end2end = not main_e2e
            args.plots = False
            args.verbose = False
            tag = "o2m" if main_e2e else "o2o"
            with _frozen_rng():  # additive only — this pass runs before the OOD loop, see _frozen_rng
                v = YOLOAnomalyValidator(self.test_loader, save_dir=self.save_dir, args=args)
                res = v(trainer=None, model=deepcopy(self.ema.ema).eval())
            return {k.replace("metrics/", f"metrics/{tag}_"): val for k, val in (res or {}).items()
                    if k.startswith("metrics/")}
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"in-domain other-branch val failed: {type(e).__name__}: {e}")
            return {}

    def _resolve_test_yamls(self, v2_cfg: dict) -> list[Path]:
        """Resolve explicit ``test_data_yamls`` or expand ``test_root`` + ``test_categories``."""
        if explicit := v2_cfg.get("test_data_yamls"):
            return [Path(p) for p in explicit]

        root = v2_cfg.get("test_root")
        if not root:
            for candidate in _MVTEC_ROOT_CANDIDATES:
                if Path(candidate).is_dir():
                    root = candidate
                    break
        if not root:
            LOGGER.warning("AnomalyRNDTrainer: no test_data_yamls or test_root configured; skipping OOD eval.")
            return []

        cats = v2_cfg.get("test_categories") or MVTEC_CATEGORIES
        yamls = []
        for cat in cats:
            for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
                p = Path(root) / cat / name
                if p.exists():
                    yamls.append(p)
                    break
            else:
                LOGGER.warning(f"AnomalyRNDTrainer: no yaml found for category '{cat}' under {root}")
        return yamls

    def _run_ood_eval(self, model, yamls: list[Path], v2_cfg: dict) -> list[dict]:
        """Fit bank per yaml and validate; return per-category metric rows."""
        rows = []
        batch = int(v2_cfg.get("test_batch", 8))
        device = self.device
        workers = self.args.workers

        # Confidence source for OOD only (the ``ood_scoring`` train arg). Training loss, domain
        # val and fitness definitions are untouched; the head scoring is restored on exit so the
        # next domain val reads cls again. ``obj`` is the class-agnostic objectness confidence —
        # its single channel is not diluted across nc sigmoids, so the conf=0.25 floor survives.
        scoring = getattr(self.args, "ood_scoring", "cls") or "cls"
        if scoring not in {"cls", "obj"}:
            LOGGER.warning(f"ood_scoring={scoring!r} invalid; falling back to 'cls'")
            scoring = "cls"
        e2e = bool(getattr(self.args, "ood_end2end", False))
        head = model.model[-1]
        saved_scoring = getattr(head, "scoring", "cls")
        head.scoring = scoring
        try:
            for yaml in yamls:
                source = _normal_dir_from_yaml(yaml)
                try:
                    model.memory_bank.reset()
                    n = model.build_memory_bank(str(source), imgsz=640, device=device, batch=batch)
                    if not n:
                        LOGGER.warning(f"OOD eval: empty bank for {yaml.name}; skipping.")
                        continue

                    overrides = {
                        "task": "detect",
                        "mode": "val",
                        "data": str(yaml),
                        "split": "val",
                        "imgsz": 640,
                        "batch": batch,
                        "workers": workers,
                        "device": str(device) if device is not None else None,
                        "rect": False,
                        "plots": False,
                        "verbose": False,
                        "save_json": False,
                        "single_cls": True,
                        "iou": 0.2,
                        # Score everything the head emits. AP is threshold-free, so the old 0.25 floor
                        # was deleting ~73% of the correct detections (their median score is 0.065)
                        # before AP was computed, understating OOD by ~3.7x. The validator re-derives
                        # the 0.25 numbers by masking, so nothing is lost and fitness is unchanged.
                        "conf": 0.001,
                        "end2end": False,
                    }
                    # Pass 1: heatmap prior (memory bank active) — the yoloa_clean fitness signal.
                    validator = YOLOAnomalyValidator(args=overrides)
                    validator(trainer=None, model=model)
                    row = {"category": yaml.parent.name, **validator._ood_map_metrics()}

                    # Pass 2: none prior (bank disabled via the ``building`` flag, same toggle the viz
                    # path uses) — bare-detector baseline, logged as ``none_*`` so the per-category
                    # fusion lift (heatmap - none) is visible. Does not change fitness.
                    mb = model.memory_bank
                    saved_building = mb.building
                    mb.building = True
                    try:
                        validator_none = YOLOAnomalyValidator(args=overrides)
                        validator_none(trainer=None, model=model)
                        row.update({f"none_{k}": v for k, v in validator_none._ood_map_metrics().items()})
                    finally:
                        mb.building = saved_building

                    # Passes 3-4 (``ood_end2end``): the same two passes on the ONE2ONE branch, i.e.
                    # the NMS-free path deployment actually uses. Everything above is o2m, so
                    # without this a run reports no number for what it would ship.
                    if e2e:
                        # agnostic_nms is REQUIRED here whenever scoring='obj'. obj is broadcast
                        # across all nc channels to keep the [4+nc] contract, so every channel of
                        # an anchor is identical; Detect.get_topk_index's per-class path then takes
                        # top-k anchors, flattens (k x nc) and takes top-k of THAT, filling every
                        # max_det slot with ~max_det/nc distinct boxes — measured at 6 unique boxes
                        # out of 300, which reads as a 100x "collapse" that is pure postprocessing.
                        # o2m never shows it because NMS dedups the copies by IoU.
                        e2e_overrides = {**overrides, "end2end": True, "agnostic_nms": scoring == "obj"}
                        with _frozen_rng():  # keeps the o2m columns identical to an ood_end2end=False run
                            v_e2e = YOLOAnomalyValidator(args=e2e_overrides)
                            v_e2e(trainer=None, model=model)
                            row.update({f"e2e_{k}": v for k, v in v_e2e._ood_map_metrics().items()})

                            mb.building = True
                            try:
                                v_e2e_none = YOLOAnomalyValidator(args=e2e_overrides)
                                v_e2e_none(trainer=None, model=model)
                                row.update({f"e2e_none_{k}": v for k, v in v_e2e_none._ood_map_metrics().items()})
                            finally:
                                mb.building = saved_building

                    rows.append(row)
                except Exception as e:
                    LOGGER.warning(f"OOD eval failed for {yaml}: {type(e).__name__}: {e}")
        finally:
            head.scoring = saved_scoring

        return rows
