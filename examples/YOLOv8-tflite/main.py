import argparse

import cv2
import numpy as np
import supervision as sv
from tflite_runtime.interpreter import Interpreter

from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml


class YOLOv8Tflite:

    def __init__(self, conf_thresh=0.1, iou_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.is_initialized = False

        self.input_details = None
        self.output_details = None
        self.input_zero = None
        self.input_scale = None
        self.output_zero = None
        self.output_scale = None
        self.is_int8 = False
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

    def load_model(self, model_path: str) -> None:
        """
        Internal function that loads the tflite file and creates
        the interpreter that deals with the EdgetPU hardware.
        """
        # Load the model and allocate

        self.model = Interpreter(model_path=model_path)

        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.input_dtype = self.input_details[0]['dtype']
        self.input_zero = self.input_details[0]['quantization'][1]
        self.input_scale = self.input_details[0]['quantization'][0]
        self.output_zero = self.output_details[0]['quantization'][1]
        self.output_scale = self.output_details[0]['quantization'][0]
        self.input_shape = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])

        self.is_initialized = True

    def infer(self, img: np.ndarray) -> sv.Detections:
        """
        :param img: numpy image
        :return: detections in the form of supervision
        """
        full_image_shape, net_image, pad = self.get_image_tensor(img, self.input_shape[0])
        net_image /= 255
        output = self.forward(net_image)

        output = np.asarray(output)
        predictions = np.squeeze(output).T

        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_thresh, :]
        confidence = scores[scores > self.conf_thresh]

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        boxes = self.xywh2xyxy(boxes)
        boxes = self.get_scaled_coords(boxes, img.shape, pad)

        detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidence)
        detections = detections.with_nms(threshold=self.iou_thresh)

        return detections

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Predict function using the EdgeTPU
        Inputs:
            x: (C, H, W) image tensor
            with_nms: apply NMS on output
        Returns:
            prediction array (with or without NMS applied)
        """
        # Transpose if C, H, W
        if x.shape[0] == 3:
            x = x.transpose((1, 2, 0))

        if self.input_zero or self.input_scale:
            # Scale input, conversion is: real = (int_8 - zero)*scale
            x = (x / self.input_scale) + self.input_zero

        x = x[np.newaxis].astype(self.input_dtype)

        self.model.set_tensor(self.input_details[0]['index'], x)
        self.model.invoke()
        x = self.model.get_tensor(self.output_details[0]['index'])

        result = x.astype(float)
        if self.output_zero or self.output_scale:
            # Scale output
            result = (result - self.output_zero) * self.output_scale

        return result

    def get_scaled_coords(self, xyxy, output_image_shape, pad):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.

        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        in_h, in_w = self.input_shape
        out_h, out_w, _ = output_image_shape

        ratio_w = out_w / (in_w - pad_w)
        ratio_h = out_h / (in_h - pad_h)

        xyxy[:, 0] *= ratio_w
        xyxy[:, 1] *= ratio_h
        xyxy[:, 2] *= ratio_w
        xyxy[:, 3] *= ratio_h

        return xyxy.astype(int)

    @staticmethod
    def _resize_and_pad(image, desired_size):
        old_size = image.shape[:2]
        ratio = float(desired_size / max(old_size))
        new_size = tuple([int(x * ratio) for x in old_size])
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]

        # new_size should be in (width, height) format
        image = cv2.resize(image, (new_size[1], new_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad = (delta_w, delta_h)

        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
        return new_im, pad

    def get_image_tensor(self, img, input_size: int):
        """
        Reshapes an input image into a square with sides max_size
        """
        new_im, pad = self._resize_and_pad(img, input_size)
        new_im = np.asarray(new_im, dtype=np.float32)
        return img.shape, new_im, pad

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


if __name__ == '__main__':

    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8s_float16.tflite', help='Input tflite model.')
    parser.add_argument('--img', type=str, default=str(ROOT / 'assets/bus.jpg'), help='Path to input image.')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()

    # # Check the requirements
    check_requirements('supervision')
    check_requirements('tflite-runtime')

    img = cv2.imread(args.img)

    # Create an instance of the Yolov8 class with the specified arguments
    detector = YOLOv8Tflite(conf_thresh=args.conf_thres, iou_thresh=args.iou_thres)

    # Load model
    detector.load_model(model_path=args.model)
    # Perform object detection and obtain the output image
    detections = detector.infer(img=img)

    box_annotator = sv.BoxAnnotator()
    label = [
        f'{detector.classes[class_id]}-{round(confidence, 2)}'
        for class_id, confidence in zip(detections.class_id, detections.confidence)]
    annotated_img = box_annotator.annotate(scene=img.copy(), detections=detections, labels=label)

    # Display the output image in a window
    cv2.imshow('Output', annotated_img)

    # Wait for a key press to exit
    cv2.waitKey(0)
