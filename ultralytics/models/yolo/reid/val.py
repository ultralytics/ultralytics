# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.data import build_dataloader
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.metrics import ReidMetrics
from ultralytics.utils.torch_utils import smart_inference_mode


class ReidValidator(ClassificationValidator):
    """Validator for person re-identification models.

    Accumulates embeddings, person IDs, and camera IDs during validation, then computes
    mAP and CMC metrics using the standard Market-1501 protocol.

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
            dataloader: DataLoader for validation.
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

    def get_desc(self) -> str:
        """Return formatted description string."""
        return ("%22s" + "%11s" * 4) % ("", "mAP", "Rank-1", "Rank-5", "Rank-10")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize tracking containers.

        Caches gallery feature extraction by ``(gallery_path, id(model), imgsz, scales, tta)``:
        Validator.__call__ fires every epoch's val during training, but the gallery split is
        static and embeddings only change when the model weights do — which the trainer
        reflects in a new model object passed here. Reuses the cached (feats, pids, camids)
        when the key matches; rebuilds otherwise.
        """
        self._model = model  # store reference for gallery feature extraction
        self.names = model.names
        self.nc = len(model.names)
        self._feats = []
        self._pids = []
        self._camids = []
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
            cached = getattr(self, "_gallery_cache", None)
            if cached is not None and cached[0] == cache_key:
                feats, pids, camids = cached[1]
            else:
                feats, pids, camids = self._extract_gallery_features(gallery_path)
                self._gallery_cache = (cache_key, (feats, pids, camids))
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
            preds: Model output from the validator loop (Tensor or 2-tuple (emb, feat_bn)).
            batch (dict): Batch with 'img', 'cls' and 'camid' keys.
        """
        if self._tta_active():
            emb = self._embed(batch["img"])
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

    def postprocess(self, preds):
        """Extract primary prediction from model output."""
        return preds

    def finalize_metrics(self) -> None:
        """Finalize metrics with speed info."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, float]:
        """Compute mAP and CMC from accumulated features.

        This performs the full query-vs-gallery evaluation. The dataloader iterates over
        the query split. For gallery evaluation, we build a separate dataset.
        """
        if not self._feats:
            return self.metrics.results_dict

        # Current accumulated features are from the val split (query)
        query_feats = torch.cat(self._feats, dim=0).numpy()
        query_pids = torch.cat(self._pids, dim=0).numpy()
        query_camids = torch.cat(self._camids, dim=0).numpy()

        reranking = getattr(self.args, "reid_reranking", False)
        self.metrics.process(query_feats, query_pids, query_camids, reranking=reranking)
        return self.metrics.results_dict

    @smart_inference_mode()
    def _extract_gallery_features(self, gallery_path: str):
        """Extract features from gallery set.

        Args:
            gallery_path (str): Path to gallery images.

        Returns:
            Tuple of (features, pids, camids) as numpy arrays.
        """
        dataset = build_yolo_dataset(self.args, gallery_path, self.args.batch, self.data, mode="gallery")
        loader = build_dataloader(dataset, self.args.batch, self.args.workers, rank=-1)

        feats, pids, camids = [], [], []
        bar = TQDM(loader, desc=f"{'Extracting gallery':>22s}", total=len(loader))
        for batch in bar:
            batch = self.preprocess(batch)
            emb = self._embed(batch["img"])
            # self.preprocess (inherited from ClassificationValidator) moves cls onto the device
            # alongside img, so all three accumulators must .cpu() before the final torch.cat().numpy().
            feats.append(emb.cpu())
            pids.append(batch["cls"].cpu())
            camids.append(
                torch.tensor([batch["camid"][i] for i in range(len(batch["camid"]))])
                if isinstance(batch["camid"], list)
                else batch["camid"].cpu()
            )

        return torch.cat(feats, dim=0).numpy(), torch.cat(pids, dim=0).numpy(), torch.cat(camids, dim=0).numpy()

    def gather_stats(self) -> None:
        """Gather stats from all GPUs for DDP."""
        if RANK == 0:
            gathered_feats = [None] * dist.get_world_size()
            gathered_pids = [None] * dist.get_world_size()
            gathered_camids = [None] * dist.get_world_size()
            dist.gather_object(self._feats, gathered_feats, dst=0)
            dist.gather_object(self._pids, gathered_pids, dst=0)
            dist.gather_object(self._camids, gathered_camids, dst=0)
            self._feats = [f for rank in gathered_feats for f in rank]
            self._pids = [p for rank in gathered_pids for p in rank]
            self._camids = [c for rank in gathered_camids for c in rank]
        elif RANK > 0:
            dist.gather_object(self._feats, None, dst=0)
            dist.gather_object(self._pids, None, dst=0)
            dist.gather_object(self._camids, None, dst=0)

    def build_dataset(self, img_path: str):
        """Create a ReidDataset instance for the query split via centralised build_yolo_dataset()."""
        return build_yolo_dataset(self.args, img_path, self.args.batch, self.data, mode="query")

    def print_results(self) -> None:
        """Print evaluation metrics."""
        pf = "%22s" + "%11.4g" * 4
        LOGGER.info(pf % ("Results", self.metrics.mAP, self.metrics.rank1, self.metrics.rank5, self.metrics.rank10))

    def plot_predictions(self, batch: dict[str, Any], preds, ni: int) -> None:
        """Plot predictions (no-op for ReID, embeddings are not visual)."""
        pass
