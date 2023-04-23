import cv2

from ultralytics.vit.engine import BasePredictor
from ultralytics.yolo.engine.results import Results
from .build import build_sam
from .modules.mask_generator import SamAutomaticMaskGenerator

class Predictor(BasePredictor):
    def setup(self):
        model = self.args.model
        print(model)
        if model and not (model.endswith(".pt") or model.endswith(".pth")):
            # Should raise AssertionError instead?
            raise NotImplementedError("Segment anything prediction requires pre-trained checkpoint")
        self.model = build_sam(model)
        self.predictor = SamAutomaticMaskGenerator(self.model, pred_iou_thresh=self.args.iou)

    def __call__(self, source=None):
        # TODO: support all sources as yolo
        source = source or self.args.source
        frame = cv2.imread(source)
        result = self.predictor.generate(frame)
        # result = Results(orig_img=frame, ) # TODO: integrate Results with sam output
        return result
        
        

