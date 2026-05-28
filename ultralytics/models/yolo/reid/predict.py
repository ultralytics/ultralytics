# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class ReidPredictor(ClassificationPredictor):
    """Predictor for person re-identification models.

    Inherits image preprocessing and source setup from ClassificationPredictor; overrides
    ``postprocess`` to wrap the L2-normalized embedding in a typed ``Results.embeddings``
    field instead of overloading ``Results.probs``. ``Results.verbose()`` then automatically
    emits ``embedding({D}-d),`` for the per-image log line, so no ``write_results`` override
    is needed.

    Model outputs may be a tensor or an ``(embedding, feat_bn)`` tuple; the ``preds[0]``
    convention takes the first element when a tuple is returned.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="path/to/query/")
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor and re-set task to 'reid' (parent pinned it to 'classify')."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"

    def postprocess(self, preds, img, orig_imgs):
        """Wrap each per-image embedding in a Results object via the typed ``embeddings`` field.

        Args:
            preds (torch.Tensor | tuple): Model output — a (B, D) embedding tensor, or a
                ``(embedding, feat_bn)`` tuple from which the first element is taken.
            img (torch.Tensor): Input image batch after preprocessing.
            orig_imgs (list[np.ndarray] | torch.Tensor): Original images before preprocessing.

        Returns:
            (list[Results]): One Results per image, each carrying its embedding vector.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, embeddings=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
