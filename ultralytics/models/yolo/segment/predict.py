# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.post_process_utils import decode_bbox

class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        if self.separate_outputs:  # Quant friendly export with separated outputs
            mcv = float('-inf')
            lci = -1
            for idx, s in enumerate(preds):
                dim_1 = s.shape[1]
                if dim_1 > mcv:
                    mcv = dim_1
                    lci = idx
                if len(s.shape) == 4:
                    proto = s
                    pidx = idx  
            mask = preds[lci]
            proto = proto.permute(0,3,1,2) 
            pred_order = [item for index, item in enumerate(preds) if index not in [pidx, lci]]
            preds_decoded = decode_bbox(pred_order, img_shape, self.device)
            preds_decoded = torch.cat([preds_decoded, mask.permute(0, 2, 1)], 1)
            p = ops.non_max_suppression(preds_decoded, #preds[0],
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
                                        nc=self.nc)
        else:
            p = ops.non_max_suppression(preds[0],
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
                                        nc=self.nc)
            proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        
        results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                if is_list:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if is_list:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
