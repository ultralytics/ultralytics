# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class MultiTaskPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a multi-task model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import MultiTaskPredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = MultiTaskPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'multi-task'
        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""

        preds_nmsed = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        # input images are a torch.Tensor, not a list
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # second output is len 4 if pt, but only 1 if exported
        proto = preds[1][-1] if len(preds[1]) == 4 else preds[1]

        kpt_shape = self.model.kpt_shape

        results = []
        for i, pred in enumerate(preds_nmsed):
            orig_img = orig_imgs[i]
            # scale boxes to original image size
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()

            # scale keypoints to original image size
            pred_kpts = pred[:, 6:6 + np.prod(kpt_shape)]
            pred_kpts = pred_kpts.view(len(pred), *self.model.kpt_shape) if len(pred) else pred_kpts
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

            img_path = self.batch[0][i]

            if not len(pred):  # save empty boxes
                masks = None
            else:
                masks = pred[:, 6 + np.prod(kpt_shape):]
                if self.args.retina_masks:
                    masks = ops.process_mask_native(proto[i], masks, pred[:, :4], orig_img.shape[:2])  # HWC
                else:
                    masks = ops.process_mask(proto[i], masks, pred[:, :4], img.shape[2:], upsample=True)  # HWC

            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    keypoints=pred_kpts,
                    masks=masks,
                ))
        return results
