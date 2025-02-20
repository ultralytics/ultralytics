# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml


class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model, conf_threshold=0.4):
        """
        Initializes the object detection model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.4.

        Attributes:
            session (ort.InferenceSession): ONNX Runtime session for running inference.
            ndtype (numpy.dtype): Data type for model input (FP16 or FP32).
            model_height (int): Height of the model's input image.
            model_width (int): Width of the model's input image.
            classes (list): List of class names from the COCO dataset.
            device (str): Specifies whether inference runs on CPU or GPU.
            conf_threshold (float): Confidence threshold for filtering detections.
        """
        # Build Ort session
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # Load COCO class names
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # Device
        self.device = "cuda:0" if ort.get_device().lower() == "gpu" else "cpu"

        # Confidence
        self.conf_threshold = conf_threshold

    def __call__(self, im0):
        """
        Runs inference on the input image using the ONNX model.

        Args:
            im0 (numpy.ndarray): The original input image in BGR format.

        Returns:
            list: Processed detection results after post-processing.

        Example:
            >>> detector = Model("yolov8.onnx")
            >>> results = detector(image)  # Runs inference and returns detections.
        """
        # Pre-process
        processed_image = self.preprocess(im0)

        # Ort inference
        predictions = self.session.run(None, {self.session.get_inputs()[0].name: processed_image})

        # Post-process
        return self.postprocess(im0, processed_image, predictions)

    def preprocess(self, image, new_shape: Union[Tuple, List] = (640, 640)):
        """
        Preprocesses the input image before feeding it into the model.

        Args:
            image (np.ndarray): The input image in BGR format.
            new_shape (Tuple or List, optional): The target shape for resizing. Defaults to (640, 640).

        Returns:
            np.ndarray: Preprocessed image ready for model inference.

        Example:
            >>> processed_img = model.preprocess(image)
        """
        image, _, _ = self.__resize_and_pad_image(image=image, new_shape=new_shape)
        image = self.__reshape_image(image=image)
        return image[None] if len(image.shape) == 3 else image

    def __reshape_image(self, image: np.ndarray) -> np.ndarray:
        """
        Reshapes the image by changing its layout and normalizing pixel values.

        Args:
            image (np.ndarray): The image to be reshaped.

        Returns:
            np.ndarray: Reshaped and normalized image.

        Example:
            >>> reshaped_img = model.__reshape_image(image)
        """
        image = image.transpose([2, 0, 1])
        image = image[np.newaxis, ...]
        return np.ascontiguousarray(image).astype(np.float32) / 255

    def __resize_and_pad_image(
        self, image=np.ndarray, new_shape: Union[Tuple, List] = (640, 640), color: Union[Tuple, List] = (114, 114, 114)
    ):
        """
        Resizes and pads the input image while maintaining the aspect ratio.

        Args:
            image (np.ndarray): The input image.
            new_shape (Tuple or List, optional): Target shape (width, height). Defaults to (640, 640).
            color (Tuple or List, optional): Padding color. Defaults to (114, 114, 114).

        Returns:
            Tuple[np.ndarray, float, float]: The resized image along with padding values.

        Example:
            >>> resized_img, dw, dh = model.__resize_and_pad_image(image)
        """
        shape = image.shape[:2]  # original image shape

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        ratio = min(new_shape[0] / shape[1], new_shape[1] / shape[0])

        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        delta_width, delta_height = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]

        # Divide padding into 2 sides
        delta_width /= 2
        delta_height /= 2

        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] == new_unpad else image

        top, bottom = int(round(delta_height - 0.1)), int(round(delta_height + 0.1))
        left, right = int(round(delta_width - 0.1)), int(round(delta_width + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image, delta_width, delta_height

    def postprocess(self, image, processed_image, predictions):
        """
        Post-processes model predictions to extract meaningful results.

        Args:
            image (np.ndarray): The original input image.
            processed_image (np.ndarray): The preprocessed image used for inference.
            predictions (list): Model output predictions.

        Returns:
            list: Processed detection results.

        Example:
            >>> results = model.postprocess(image, processed_image, predictions)
        """
        torch_tensor_predictions = [torch.from_numpy(output) for output in predictions]
        torch_tensor_boxes_confidence_category_predictions = torch_tensor_predictions[0]
        masks_predictions_tensor = torch_tensor_predictions[1].to(self.device)

        nms_boxes_confidence_category_predictions_tensor = ops.non_max_suppression(
            torch_tensor_boxes_confidence_category_predictions,
            conf_thres=self.conf_threshold,
            nc=len(self.classes),
            agnostic=False,
            max_det=100,
            max_time_img=0.001,
            max_nms=1000,
        )

        results = []
        for idx, predictions in enumerate(nms_boxes_confidence_category_predictions_tensor):
            predictions = predictions.to(self.device)
            masks = self.__process_mask(
                masks_predictions_tensor[idx],
                predictions[:, 6:],
                predictions[:, :4],
                processed_image.shape[2:],
                upsample=True,
            )  # HWC
            predictions[:, :4] = ops.scale_boxes(processed_image.shape[2:], predictions[:, :4], image.shape)
            results.append(Results(image, path="", names=self.classes, boxes=predictions[:, :6], masks=masks))

        return results

    def __process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Processes segmentation masks from the model output.

        Args:
            protos (torch.Tensor): The prototype mask predictions from the model.
            masks_in (torch.Tensor): The raw mask predictions.
            bboxes (torch.Tensor): Bounding boxes for the detected objects.
            shape (Tuple): Target shape for mask resizing.
            upsample (bool, optional): Whether to upscale masks to match the original image size. Defaults to False.

        Returns:
            torch.Tensor: Processed binary masks.

        Example:
            >>> masks = model.__process_mask(protos, masks_in, bboxes, shape, upsample=True)
        """
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
        width_ratio = mw / iw
        height_ratio = mh / ih

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= width_ratio
        downsampled_bboxes[:, 2] *= width_ratio
        downsampled_bboxes[:, 3] *= height_ratio
        downsampled_bboxes[:, 1] *= height_ratio

        masks = ops.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
        return masks.gt_(0.5).to(self.device)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # Build model
    model = YOLOv8Seg(args.model, args.conf)

    # Read image by OpenCV
    img = cv2.imread(args.source)
    img = cv2.resize(img, (640, 640))  # Can be changed based on your models expected size

    # Inference
    results = model(img)

    cv2.imshow("Segmented Image", results[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
