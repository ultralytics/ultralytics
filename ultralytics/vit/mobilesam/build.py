# Ultralytics YOLO 🚀, AGPL-3.0 license
from .setup_mobile_sam import setup_model


def build_sam_vit_tiny(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) b-size model."""
    return setup_model(checkpoint)

sam_model_map = {'mobile_sam.pt': build_sam_vit_tiny}

def build_sam(ckpt='mobile_sam.pt'):
    """Build a SAM model specified by ckpt."""
    model_builder = None
    for k in sam_model_map.keys():
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)

    if not model_builder:
        raise FileNotFoundError(f'{ckpt} is not a supported sam model. Available models are: \n {sam_model_map.keys()}')

    return model_builder(ckpt)
