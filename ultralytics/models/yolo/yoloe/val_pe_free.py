from ultralytics.models.yolo.detect import DetectionValidator

class YOLOEPEFreeValidatorMixin:
    def eval_json(self, stats):
        return stats

class YOLOEPEFreeDetectValidator(YOLOEPEFreeValidatorMixin, DetectionValidator):
    pass