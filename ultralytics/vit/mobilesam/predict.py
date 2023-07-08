# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.torch_utils import select_device

from ..sam.modules.mask_generator import SamAutomaticMaskGenerator
from ..sam.modules.prompt_predictor import PromptPredictor


class Predictor(BasePredictor):

    def preprocess(self, im):
        """Prepares input image for inference."""
        # TODO: Only support bs=1 for now
        # im = ResizeLongestSide(1024).apply_image(im[0])
        # im = torch.as_tensor(im, device=self.device)
        # im = im.permute(2, 0, 1).contiguous()[None, :, :, :]
        return im[0]

    def setup_model(self, model):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        self.model = SamAutomaticMaskGenerator(model.to(device).eval(),
                                               pred_iou_thresh=self.args.conf,
                                               box_nms_thresh=self.args.iou)
        self.device = device
        # TODO: Temporary settings for compatibility
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def predict_point(self, model, source, input_point, input_label):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        save_dir = self.save_dir
        predictor = PromptPredictor(model)
        predictor.set_image(source)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(source)
        show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        save_dir = str(save_dir) + '_point/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('{}mobile_test.jpg'.format(save_dir), bbox_inches='tight', pad_inches=0.0)

        self.device = device
        # TODO: Temporary settings for compatibility
        # self.model.pt = False
        # self.model.triton = False
        # self.model.stride = 32
        # self.model.fp16 = False
        # self.done_warmup = True
        return 0

    def predict_box(self, model, source, input_box):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        save_dir = self.save_dir
        predictor = PromptPredictor(model)
        predictor.set_image(source)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(source)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
        save_dir = str(save_dir) + '_box/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig('{}mobile_test.jpg'.format(save_dir), bbox_inches='tight', pad_inches=0.0)

        self.device = device
        # TODO: Temporary settings for compatibility
        # self.model.pt = False
        # self.model.triton = False
        # self.model.stride = 32
        # self.model.fp16 = False
        # self.done_warmup = True
        return 0

    def postprocess(self, preds, path, orig_imgs):
        """Postprocesses inference output predictions to create detection masks for objects."""
        names = dict(enumerate(list(range(len(preds)))))
        results = []
        # TODO
        for i, pred in enumerate([preds]):
            masks = torch.from_numpy(np.stack([p['segmentation'] for p in pred], axis=0))
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=names, masks=masks))
        return results


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
