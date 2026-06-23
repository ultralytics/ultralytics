# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class ReidPredictor(ClassificationPredictor):
    """Predictor for person re-identification models.

    Wraps each image's L2-normalized embedding in ``Results.embeddings``. Gallery retrieval and visualization are
    intentionally out of scope here — use ``ultralytics.solutions.ReIDVisualizer``.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="query.jpg")
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor and re-set task to 'reid'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"

    def postprocess(self, preds, img, orig_imgs):
        """Wrap each model embedding in a ``Results`` object.

        Args:
            preds (torch.Tensor | tuple): (B, D) embeddings or an ``(embedding, feat_bn)`` tuple.
            img (torch.Tensor): Preprocessed input batch.
            orig_imgs (list[np.ndarray] | torch.Tensor): Original images.

        Returns:
            (list[Results]): One ``Results`` per image with ``embeddings`` populated.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, embeddings=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
