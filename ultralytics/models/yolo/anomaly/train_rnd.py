# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Internal R&D trainer for YOLO Anomaly.

Extends ``AnomalyTrainer`` with periodic cross-dataset OOD validation, so best.pt is selected on a
whole anomaly catalogue rather than the training set. The evaluation itself lives in ``val_rnd.py``;
this only schedules it and turns its result into fitness.
"""

from __future__ import annotations

import math
from copy import deepcopy

from torch import distributed as dist

from ultralytics.models.yolo.anomaly.train import AnomalyTrainer
from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
from ultralytics.models.yolo.anomaly.val_rnd import OODEvaluator, _average_ood_rows, _frozen_rng
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyRNDTrainer(AnomalyTrainer):
    """Trainer for YOLOAnomaly with periodic cross-dataset OOD validation."""

    def validate(self):
        """Run normal validation, then periodic OOD validation, and select best.pt on its result.

        Fitness is ``mAP50@0.25`` on three axes, each set by an arg and each choosing a different
        number from the same passes:

        - ``fitness_branch`` (o2m / o2o): o2m is the NMS path, o2o the NMS-free one a deployment
          ships. They do not peak together -- on 26s, o2o OOD tops out at ep5 and decays
          0.2127 -> 0.1925 while o2m stays flat through ep10 -- so an o2m-selected best.pt is past
          o2o's optimum. o2o needs ``ood_end2end=True``.
        - ``fitness_prior`` (heatmap / none): the memory-bank prior. ``p_drop=1.0`` runs drop it on
          every training sample, so prior-OFF is their native regime.
        - ``fitness_groups``: unset keeps the macro mean over products; set makes fitness the pooled
          binary AP of the selected groups. The two are not comparable -- pooling is one global
          ranked list and sits below the mean of per-product APs -- so turning it on is a new
          project (experiment_protocol.md section 0).

        Returns:
            (tuple[dict, float]): Validation metrics and the fitness value.
        """
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

        if getattr(self.args, "ood_end2end", False):
            metrics.update(self._domain_other_branch())

        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly", {})
        freq = int(v2_cfg.get("test_val_freq", 0))
        if freq <= 0 or (self.epoch + 1) % freq != 0:
            return metrics, fitness

        ev = OODEvaluator(
            v2_cfg,
            groups=(getattr(self.args, "fitness_groups", "") or "").strip(),
            e2e=bool(getattr(self.args, "ood_end2end", False)),
            device=self.device,
            workers=self.args.workers,
            save_dir=self.save_dir,
        )
        if not ev.resolve():
            return metrics, fitness

        ema_eval = deepcopy(self.ema.ema).eval()
        try:
            res = ev(ema_eval, epoch=self.epoch)
            rows = res.products
            if rows:
                avg = _average_ood_rows(rows)
                avg.update(res.pooled)
                avg_metrics = {f"ood/{k}": v for k, v in avg.items()}
                branch = getattr(self.args, "fitness_branch", "o2m") or "o2m"
                if branch not in {"o2m", "o2o"}:
                    LOGGER.warning(f"fitness_branch={branch!r} invalid; falling back to 'o2m'")
                    branch = "o2m"
                # CLI `fitness_prior=none` arrives as Python None, so `or <default>` would eat it.
                prior = getattr(self.args, "fitness_prior", "heatmap")
                prior = "none" if prior is None else prior
                if prior not in {"heatmap", "none"}:
                    LOGGER.warning(f"fitness_prior={prior!r} invalid; falling back to 'heatmap'")
                    prior = "heatmap"
                pre = ("pool_" if (getattr(self.args, "fitness_groups", "") or "").strip() else "") + (
                    "e2e_" if branch == "o2o" else ""
                ) + ("none_" if prior == "none" else "")
                if pre and f"{pre}mAP50@0.25" not in avg:
                    LOGGER.warning(
                        f"fitness_branch={branch!r} fitness_prior={prior!r} unavailable (no {pre}* metrics; "
                        "o2o needs ood_end2end=True, none needs test_none_prior, pool_ needs "
                        "anomaly.meta_yaml + a fitness_groups query that matched); "
                        "falling back to the o2m heatmap MACRO mean -- not the same number"
                    )
                    pre, branch, prior = "", "o2m", "heatmap"
                fitness = float(avg.get(f"{pre}mAP50@0.25", avg[f"{pre}mAP50"]))
                metrics["fitness"] = fitness
                metrics.update(avg_metrics)
                self.best_fitness = max(self.best_fitness or -math.inf, fitness)
                LOGGER.info(
                    f"OOD eval @ep{self.epoch + 1}: [heatmap] mAP50={avg['mAP50']:.4f} "
                    f"mAP10={avg['mAP10']:.4f} "
                    f"| [none] mAP50={avg.get('none_mAP50', float('nan')):.4f} "
                    f"mAP10={avg.get('none_mAP10', float('nan')):.4f} "
                    f"| fitness={fitness:.4f} "
                    f"({'pooled ' if pre.startswith('pool_') else 'macro '}{branch} {prior} mAP50@0.25); "
                    f"bare keys are threshold-free; n={len(rows)} categories"
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
            return {
                k.replace("metrics/", f"metrics/{tag}_"): val
                for k, val in (res or {}).items()
                if k.startswith("metrics/")
            }
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"in-domain other-branch val failed: {type(e).__name__}: {e}")
            return {}

