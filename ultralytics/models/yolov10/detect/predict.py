# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class YOLOv10DetectionPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on YOLOv10 models.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolov10.detect import YOLOv10DetectionPredictor

        args = dict(model='yolov10n.pt', source=ASSETS)
        predictor = YOLOv10DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred = pred[pred[:, 4] > self.args.conf]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
