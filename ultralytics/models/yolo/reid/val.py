# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.data import build_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.metrics import ReidMetrics
from ultralytics.utils.plotting import plot_reid_retrieval
from ultralytics.utils.torch_utils import autocast, smart_inference_mode


def select_diagnostic_queries(hits, n: int = 8) -> list[int]:
    """Pick a diagnostic mix of correct and incorrect rank-1 queries to visualize.

    Splits query indices by whether their rank-1 retrieval was a true match (``hits[i]``), then takes an evenly-strided
    sample of each half so the plot surfaces both successes and
    failures. Falls back to the other class when one is short. Fully deterministic (no RNG)
    so the same eval reproduces the same panel across epochs.

    Args:
        hits (np.ndarray): Boolean array (len Q); True iff the query's rank-1 match is correct.
        n (int): Target number of queries to show.

    Returns:
        (list[int]): Up to ``n`` query indices, sorted ascending.
    """
    hits = np.asarray(hits, dtype=bool)
    q = len(hits)
    if q <= n:
        return list(range(q))

    miss_idx = np.flatnonzero(~hits)
    hit_idx = np.flatnonzero(hits)

    def _stride(arr, count):
        if count <= 0 or len(arr) == 0:
            return []
        if len(arr) <= count:
            return arr.tolist()
        pos = np.linspace(0, len(arr) - 1, count).round().astype(int)
        return arr[np.unique(pos)].tolist()

    want_miss = n // 2
    want_hit = n - want_miss
    chosen = set(_stride(miss_idx, want_miss)) | set(_stride(hit_idx, want_hit))

    # Backfill from whichever pool still has unused indices until we reach n (or run out).
    pool = [i for i in (*hit_idx.tolist(), *miss_idx.tolist()) if i not in chosen]
    for i in pool:
        if len(chosen) >= n:
            break
        chosen.add(i)
    return sorted(chosen)


class ReidValidator(ClassificationValidator):
    """Validator for person re-identification models.

    Accumulates embeddings, person IDs, and camera IDs during validation, then computes mAP and CMC metrics using the
    standard Market-1501 protocol.

    Attributes:
        query_feats (list): Accumulated query feature embeddings.
        query_pids (list): Accumulated query person IDs.
        query_camids (list): Accumulated query camera IDs.
        gallery_feats (list): Accumulated gallery feature embeddings.
        gallery_pids (list): Accumulated gallery person IDs.
        gallery_camids (list): Accumulated gallery camera IDs.
        metrics (ReidMetrics): Metrics calculator.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidValidator
        >>> args = dict(model="yolo26n-reid.pt", data="Market-1501.yaml")
        >>> validator = ReidValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize ReidValidator.

        Args:
            dataloader (Any, optional): DataLoader for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Validation configuration.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "reid"
        self.metrics = ReidMetrics()
        self._feats = []
        self._pids = []
        self._camids = []
        self._paths = []

    def get_desc(self) -> str:
        """Return formatted description string."""
        return ("%22s" + "%11s" * 4) % ("", "mAP", "Rank-1", "Rank-5", "Rank-10")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize tracking containers.

        Caches gallery feature extraction by ``(gallery_path, id(model), imgsz, scales, tta)``
        for standalone re-validation of an unchanged model object: reuses the cached
        (feats, pids, camids, paths) when the key matches; rebuilds otherwise. During training
        the cache is bypassed entirely — the trainer passes the SAME EMA module every epoch and
        mutates its weights in place, so ``id(model)`` can never detect the change.
        """
        self._model = model  # store reference for gallery feature extraction
        self.names = model.names
        self.nc = len(model.names)
        self._feats = []
        self._pids = []
        self._camids = []
        self._paths = []
        self._gallery_paths = []
        # Build gallery dataset and extract features
        gallery_path = self.data.get("gallery", self.data.get("test", ""))
        if gallery_path:
            gallery_path = Path(self.data["path"]) / gallery_path
            scales = getattr(self.args, "reid_scales", None) or ""
            cache_key = (
                str(gallery_path),
                id(model),
                int(self.args.imgsz) if self.args.imgsz else 0,
                str(scales),
                bool(getattr(self.args, "reid_tta", False)),
            )
            # In-train the trainer hands us the SAME EMA module every epoch (weights mutate
            # in place), so id(model) in the key can never detect the change — always
            # re-extract. The cache only helps (and is only correct) for standalone re-val
            # of an unchanged model object.
            cached = None if getattr(self, "training", False) else getattr(self, "_gallery_cache", None)
            if cached is not None and cached[0] == cache_key:
                feats, pids, camids, paths = cached[1]
            else:
                feats, pids, camids, paths = self._extract_gallery_features(gallery_path)
                self._gallery_cache = (cache_key, (feats, pids, camids, paths))
            self._gallery_paths = paths
            self.metrics.update_gallery(feats, pids, camids)

    def _embed(self, img: torch.Tensor) -> torch.Tensor:
        """Compute the final embedding for a batch of images, applying optional flip and multi-scale TTA.

        Flip TTA (``reid_tta=True``) sums the embeddings of the image and its horizontal mirror.
        Multi-scale TTA (``reid_scales="320,384"``) additionally sums embeddings extracted at each
        square input size, bilinearly resizing the **already-preprocessed** (normalized) batch to
        each scale. The fused embedding is L2-renormalised so the returned tensor stays unit-norm.

        Note on multi-scale: standard ReID multi-scale TTA resizes the *raw RGB image* and then
        re-normalises; this implementation resizes the post-Normalize tensor, which is cheaper but
        is NOT equivalent to averaging unit-norm vectors — the per-scale views see slightly
        perturbed channel statistics. In practice this trades a small mAP gain for code simplicity.
        If you need paper-faithful multi-scale TTA, run val separately at each imgsz and average
        the embeddings externally.

        Args:
            img (torch.Tensor): Preprocessed image batch (B, 3, H, W) on the model device.

        Returns:
            (torch.Tensor): L2-normalized fused embedding (B, D).
        """
        # Match the model's weight dtype so a fp16 batch (in-train AMP val) doesn't hit unimplemented
        # fp16 CPU conv kernels against an fp32 EMA model; CUDA autocast still handles mixed precision.
        param = next(self._model.parameters(), None)
        if param is not None and img.dtype != param.dtype:
            img = img.to(param.dtype)

        scales = getattr(self.args, "reid_scales", None)
        if isinstance(scales, str):
            scales = [int(s) for s in scales.replace(" ", "").split(",") if s]
        if not scales:
            scales = [img.shape[-1]]  # native size only (default behavior)
        tta = getattr(self.args, "reid_tta", False)

        fused = None
        for s in scales:
            x = img if s == img.shape[-1] else F.interpolate(img, size=(s, s), mode="bilinear", align_corners=False)
            out = self._model(x)
            emb = out[0] if isinstance(out, (list, tuple)) else out
            if tta:
                out_flip = self._model(x.flip(dims=[3]))
                emb_flip = out_flip[0] if isinstance(out_flip, (list, tuple)) else out_flip
                emb = emb + emb_flip
            fused = emb if fused is None else fused + emb
        # Renorm is a no-op when scales is single-element AND tta is False (head already L2-norms),
        # but the cost is one extra kernel per batch — kept for uniformity. Use dim=-1 for
        # robustness against any future (D,) outputs.
        return F.normalize(fused, dim=-1)

    def _tta_active(self) -> bool:
        """Return True iff multi-scale or flip TTA is enabled (requires re-running _embed)."""
        scales = getattr(self.args, "reid_scales", None)
        if isinstance(scales, str):
            scales = [int(s) for s in scales.replace(" ", "").split(",") if s]
        return bool(scales) or bool(getattr(self.args, "reid_tta", False))

    def update_metrics(self, preds, batch: dict[str, Any]) -> None:
        """Accumulate query embeddings and metadata.

        Default no-TTA path consumes the ``preds`` already computed by the validator loop —
        avoids a redundant second forward pass per batch. When ``reid_tta`` or ``reid_scales``
        is set, _embed re-runs the model at each augmented view.

        Args:
            preds (Any): Model output from the validator loop (Tensor or 2-tuple (emb, feat_bn)).
            batch (dict): Batch with 'img', 'cls' and 'camid' keys.
        """
        if self._tta_active():
            # Same dtype guard as gallery extraction: update_metrics runs outside the engine
            # loop's autocast block.
            with autocast(self.training and self.args.half, device=self.device.type):
                emb = self._embed(batch["img"])
            emb = emb.float()
        else:
            # Unwrap eval-mode 2-tuple; the head emits already-L2-normalised embeddings, so the
            # extra F.normalize step from _embed is unnecessary in this default path.
            emb = preds[0] if isinstance(preds, (list, tuple)) else preds

        self._feats.append(emb.cpu())
        self._pids.append(batch["cls"].cpu())
        self._camids.append(
            torch.tensor([batch["camid"][i] for i in range(len(batch["camid"]))])
            if isinstance(batch["camid"], list)
            else batch["camid"].cpu()
        )
        self._paths.extend(batch["im_file"])

    def postprocess(self, preds):
        """Extract primary prediction from model output."""
        return preds

    def finalize_metrics(self) -> None:
        """Finalize metrics with speed info, then render the query->retrieval diagnostic grid.

        Note: does NOT call super().finalize_metrics() — the ClassificationValidator base
        dereferences confusion_matrix/pred/targets which ReidValidator never populates (it
        accumulates embeddings, not class predictions). The two lines below are the only base
        behavior ReID needs.
        """
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self._plot_retrieval()

    def _plot_retrieval(self) -> None:
        """Save runs/.../val_retrieval0.jpg: sampled queries (rows) x top-k gallery matches (cols).

        Reuses the embeddings/paths/ranking already computed during validation. Skips silently
        when plotting is off, on non-zero DDP ranks, in the no-gallery fallback, or when the data
        needed to plot is unavailable.
        """
        if not self.args.plots or RANK not in {-1, 0}:
            return
        match_indices = getattr(self.metrics, "match_g_indices", [])
        if not match_indices or not getattr(self.metrics, "_gallery_provided", False):
            return
        q_paths = getattr(self, "_paths", [])
        g_paths = getattr(self, "_gallery_paths", [])
        q_pids = getattr(self, "_query_pids", None)
        q_camids = getattr(self, "_query_camids", None)
        g_pids = self.metrics.gallery_pids
        if q_pids is None or q_camids is None or not q_paths or not g_paths or g_pids is None:
            return

        # Rank-1 correctness per query (empty rankings count as misses).
        hits = np.array(
            [len(match_indices[i]) > 0 and g_pids[match_indices[i][0]] == q_pids[i] for i in range(len(match_indices))]
        )
        selected = select_diagnostic_queries(hits, n=8)
        if not selected:
            return

        blue, green, red = (80, 170, 255), (70, 200, 120), (215, 95, 95)
        rows = []
        for qi in selected:
            q_label = f"QUERY  pid={int(q_pids[qi])} c={int(q_camids[qi])}"
            row = [(q_paths[qi], q_label, blue)]
            dists = self.metrics.match_dists[qi]
            for rank, gi in enumerate(match_indices[qi], start=1):
                is_match = g_pids[gi] == q_pids[qi]
                color = green if is_match else red
                row.append((g_paths[gi], f"#{rank}  pid={int(g_pids[gi])}  d={float(dists[rank - 1]):.3f}", color))
            rows.append(row)

        out_path = self.save_dir / "val_retrieval0.jpg"
        plot_reid_retrieval(rows, out_path)
        self.on_plot(out_path)

    def get_stats(self) -> dict[str, float]:
        """Compute mAP and CMC from accumulated features.

        This performs the full query-vs-gallery evaluation. The dataloader iterates over
        the query split. For gallery evaluation, we build a separate dataset.
        """
        if not self._feats:
            return self.metrics.results_dict

        # Current accumulated features are from the val split (query)
        query_feats = torch.cat(self._feats, dim=0).float().numpy()  # fp32: match forced-fp32 gallery feats
        query_pids = torch.cat(self._pids, dim=0).numpy()
        query_camids = torch.cat(self._camids, dim=0).numpy()
        self._query_pids = query_pids
        self._query_camids = query_camids

        reranking = getattr(self.args, "reid_reranking", False)
        self.metrics.process(query_feats, query_pids, query_camids, reranking=reranking)
        return self.metrics.results_dict

    @smart_inference_mode()
    def _extract_gallery_features(self, gallery_path: str):
        """Extract features from gallery set.

        Args:
            gallery_path (str): Path to gallery images.

        Returns:
            Tuple of (features, pids, camids) as numpy arrays plus the list of gallery image paths.
        """
        dataset = build_yolo_dataset(self.args, gallery_path, self.args.batch, self.data, mode="gallery")
        loader = build_dataloader(dataset, self.args.batch, self.args.workers, rank=-1)

        feats, pids, camids, paths = [], [], [], []
        bar = TQDM(loader, desc=f"{'Extracting gallery':>22s}", total=len(loader))
        for batch in bar:
            batch = self.preprocess(batch)
            # In-train val: fp16 inputs (args.half=trainer.amp) but fp32 EMA weights — mirror the
            # engine loop's autocast guard (validator.py) or CUDA conv crashes on dtype mismatch.
            with autocast(self.training and self.args.half, device=self.device.type):
                emb = self._embed(batch["img"])
            emb = emb.float()
            # self.preprocess (inherited from ClassificationValidator) moves cls onto the device
            # alongside img, so all three accumulators must .cpu() before the final torch.cat().numpy().
            feats.append(emb.cpu())
            pids.append(batch["cls"].cpu())
            camids.append(
                torch.tensor([batch["camid"][i] for i in range(len(batch["camid"]))])
                if isinstance(batch["camid"], list)
                else batch["camid"].cpu()
            )
            paths.extend(batch["im_file"])

        return (
            torch.cat(feats, dim=0).numpy(),
            torch.cat(pids, dim=0).numpy(),
            torch.cat(camids, dim=0).numpy(),
            paths,
        )

    def gather_stats(self) -> None:
        """Gather stats from all GPUs for DDP."""
        if RANK == 0:
            gathered_feats = [None] * dist.get_world_size()
            gathered_pids = [None] * dist.get_world_size()
            gathered_camids = [None] * dist.get_world_size()
            gathered_paths = [None] * dist.get_world_size()
            dist.gather_object(self._feats, gathered_feats, dst=0)
            dist.gather_object(self._pids, gathered_pids, dst=0)
            dist.gather_object(self._camids, gathered_camids, dst=0)
            dist.gather_object(self._paths, gathered_paths, dst=0)
            self._feats = [f for rank in gathered_feats for f in rank]
            self._pids = [p for rank in gathered_pids for p in rank]
            self._camids = [c for rank in gathered_camids for c in rank]
            self._paths = [p for rank in gathered_paths for p in rank]
        elif RANK > 0:
            dist.gather_object(self._feats, None, dst=0)
            dist.gather_object(self._pids, None, dst=0)
            dist.gather_object(self._camids, None, dst=0)
            dist.gather_object(self._paths, None, dst=0)

    def build_dataset(self, img_path: str):
        """Create a ReidDataset instance for the query split via centralized build_yolo_dataset()."""
        return build_yolo_dataset(self.args, img_path, self.args.batch, self.data, mode="query")

    def print_results(self) -> None:
        """Print evaluation metrics."""
        pf = "%22s" + "%11.4g" * 4
        LOGGER.info(pf % ("Results", self.metrics.mAP, self.metrics.rank1, self.metrics.rank5, self.metrics.rank10))

    def plot_predictions(self, batch: dict[str, Any], preds, ni: int) -> None:
        """Plot predictions (no-op for ReID, embeddings are not visual)."""
        pass
