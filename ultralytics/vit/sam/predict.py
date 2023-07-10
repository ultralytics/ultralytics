# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.torch_utils import select_device
from .modules.mask_generator import SamAutomaticMaskGenerator


class Predictor(BasePredictor):

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img = (img - self.mean) / self.std
        return img

    def preprocess(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        return [LetterBox(self.imgsz, auto=False)(image=x) for x in im]

    def inference(self, im, boxes=None, points=None, labels=None):
        """
        Predict masks for the given input prompts, using the currently set image.

        Args:
          box (np.ndarray, None): A length 4 array given a box prompt to the
            model, in XYXY format.
          point_coords (np.ndarray, None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray, None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError('An image must be set with .set_image(...) before mask prediction.')

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (point_labels is not None), 'point_labels must be supplied if point_coords is supplied.'
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        pass

    def setup_model(self, model):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        model.eval()
        # self.model = SamAutomaticMaskGenerator(model.to(device),
        #                                        pred_iou_thresh=self.args.conf,
        #                                        box_nms_thresh=self.args.iou)
        self.model = model
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        # TODO: Temporary settings for compatibility
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

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

    def set_image(self, image):
        """Set image in advance.
        Args:

            im (torch.Tensor | np.ndarray): BCHW for tensor, HWC for ndarray.
        """
        im = self.preprocess(image)
        self.features = self.model.image_encoder(im)

    def reset_image(self):
        self.features = None
