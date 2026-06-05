# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.reid import retrieval
from ultralytics.utils import DEFAULT_CFG, ops


class ReidPredictor(ClassificationPredictor):
    """Predictor for person re-identification models.

    Default behavior wraps each image's L2-normalized embedding in ``Results.embeddings``.
    When a ``gallery`` argument is supplied, the predictor instead performs retrieval: it embeds
    the gallery once (optionally cached via ``reid_cache``), ranks each streamed query against it
    by cosine similarity, and attaches the top-``topk`` ``(path, score)`` matches to
    ``Results.matches`` (a montage is saved in ``write_results``).

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="query.jpg", gallery="gallery/", topk=5)
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor, re-set task to 'reid', and clear the lazy gallery index."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"
        self.gallery_paths = None
        self.gallery_embs = None

    def _embed_paths(self, paths: list) -> np.ndarray:
        """Embed a list of image paths in batches using this predictor's model and transforms.

        Returns a (N, D) float32 array (un-normalized; ``build_gallery`` L2-normalizes).
        """
        bs = max(int(getattr(self.args, "batch", 16) or 16), 1)
        embs: list[np.ndarray] = []
        for start in range(0, len(paths), bs):
            chunk = paths[start : start + bs]
            ims = [cv2.imread(str(p)) for p in chunk]
            im = self.preprocess(ims)
            with torch.no_grad():
                preds = self.model(im)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            embs.append(preds.detach().cpu().float().numpy())
        return np.concatenate(embs, axis=0)

    def _ensure_gallery(self) -> None:
        """Build the gallery embedding index once (lazy), honoring the optional cache."""
        if self.gallery_embs is None:
            self.gallery_paths, self.gallery_embs = retrieval.build_gallery(
                self._embed_paths,
                getattr(self.args, "gallery"),
                cache=getattr(self.args, "reid_cache", None),
                model_id=str(getattr(self.args, "model", "")),
                imgsz=self.imgsz,
            )

    def postprocess(self, preds, img, orig_imgs):
        """Wrap embeddings in Results; when a gallery is set, also attach ranked matches.

        Args:
            preds (torch.Tensor | tuple): (B, D) embeddings or an ``(embedding, feat_bn)`` tuple.
            img (torch.Tensor): Preprocessed input batch.
            orig_imgs (list[np.ndarray] | torch.Tensor): Original images.

        Returns:
            (list[Results]): One Results per query, each with ``embeddings`` and (when a gallery
                is supplied) ``matches`` populated.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        matches_per_query = [None] * len(preds)
        if getattr(getattr(self, "args", None), "gallery", None):
            self._ensure_gallery()
            query = retrieval.l2_normalize(preds.detach().cpu().float().numpy())
            topk = int(getattr(self.args, "topk", 5) or 5)
            idx, scores = retrieval.cosine_topk(query, self.gallery_embs, topk)
            matches_per_query = [
                [(str(self.gallery_paths[j]), float(s)) for j, s in zip(idx[q], scores[q])]
                for q in range(len(preds))
            ]

        return [
            Results(orig_img, path=img_path, names=self.model.names, embeddings=pred, matches=matches)
            for pred, orig_img, img_path, matches in zip(preds, orig_imgs, self.batch[0], matches_per_query)
        ]
