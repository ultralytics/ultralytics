# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model, conf=0.25, iou=0.7, imgsz=640):
        """
        Initializes the object detection model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou (float, optional): IoU threshold for NMS. Defaults to 0.7.
            imgsz (int | Tuple): Input image size of the model.

        Attributes:
            session (ort.InferenceSession): ONNX Runtime session.
            imgsz (Tuple): Input image size of the model.
            classes (dict): Class mappings from the COCO dataset.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold used by NMS.
        """
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"],
        )

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]
        self.conf = conf
        self.iou = iou

    def __call__(self, img):
        """
        Runs inference on the input image using the ONNX model.

        Args:
            img (numpy.ndarray): The original input image in BGR format.

        Returns:
            list: Processed detection results after post-processing.
        """
        prep_img = self.preprocess(img, self.imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs)

    def letterbox(self, img, new_shape=(640, 640)):
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img

    def preprocess(self, img, new_shape):
        """
        Preprocesses the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (Tuple or List, optional): The target shape for resizing. Defaults to (640, 640).

        Returns:
            np.ndarray: Preprocessed image ready for model inference.
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255
        return img

    def postprocess(self, img, prep_img, outs):
        """
        Post-processes model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs.

        Returns:
            list: Processed detection results.
        """
        preds, protos = [torch.from_numpy(p) for p in outs]
        preds = ops.non_max_suppression(preds, self.conf, self.iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))

        return results

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

        Args:
            protos (torch.Tensor): [mask_dim, mask_h, mask_w]
            masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms.
            bboxes (torch.Tensor): [n, 4], n is number of masks after nms.
            shape (Tuple): The size of the input image (h,w).

        Returns:
            masks (torch.Tensor): The returned masks with dimensions [h, w, n].
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        masks = ops.scale_masks(masks[None], shape)[0]  # CHW
        masks = ops.crop_mask(masks, bboxes)  # CHW
        return masks.gt_(0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLOv8Seg(args.model, args.conf, args.iou)
    img = cv2.imread(args.source)
    results = model(img)

    cv2.imshow("Segmented Image", results[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
