import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.plotting import Annotator, colors

from copy import deepcopy


def visualize(img,
              results: Results,
              model=None,
              labels=None,
              device='cpu',
              show_conf=True,
              line_width=None,
              font_size=None,
              font='Arial.ttf',
              pil=False,
              example='abc'):
    """
    Plots the given result on an input RGB image. Accepts cv2(numpy) or PIL Image

    Args:
      img (): Image
      results (Results): The result/prediction of a model
      line_width (Float): The line width of boxes. Automatically scaled to img size if not provided
      model (): The model used to predict the results. Used to extract class labels if `class_map` is not provided
      labels (): The list of class names
      show_conf (bool): Show the confidence
      font_size (Float): The font size of . Automatically scaled to img size if not provided
    """
    img = deepcopy(img)
    if isinstance(img, Image.Image):  # handle PILLOW image
        img = np.asarray(img)[:, :, ::-1]
        img = np.ascontiguousarray(img)

    annotator = Annotator(img, line_width, font_size, font, pil, example)
    boxes = results.boxes
    masks = results.masks
    logits = results.probs
    if not (model or labels):
        LOGGER.info("Both model and labels not provided! Class indeces will be used to identify predictions.")
    names = labels or model.names

    if boxes is not None:
        for d in reversed(boxes):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            c = int(cls)
            label = (f'{names[c]}' if names else f'{c}') + (f'{conf:.2f}' if show_conf else '')
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    if masks is not None:
        im_gpu = torch.as_tensor(img, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()
        im_gpu = F.resize(im_gpu, masks.data.shape[1:]) / 255
        annotator.masks(masks.data, colors=[colors(x, True) for x in boxes.cls], im_gpu=im_gpu)
    
    if logits is not None:
        top5i = logits.argsort(0, descending=True)[:5].tolist()  # top 5 indices
        text = f"{', '.join(f'{names[j] if names else j} {logits[j]:.2f}' for j in top5i)}, "
        annotator.text((32, 32), text, txt_color=(255, 255, 255)) # TODO: allow setting colors

    return img


if __name__ == "__main__":
    from ultralytics import YOLO
    from ultralytics.yolo.utils import ROOT
    import cv2

    model = YOLO("yolov8n-cls.pt")
    source = str(ROOT / "assets/bus.jpg")
    img_cv = cv2.imread(source)
    img_pil = Image.open(source)

    res = model(img_cv)
    resimg = visualize(img_cv, res[0], model)

    cv2.imshow("res", resimg)
    cv2.waitKey(0)
