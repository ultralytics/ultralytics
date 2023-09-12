import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class Yolov8:

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, polygon, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        draw = ImageDraw.Draw(img, 'RGBA')
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        # get alpha color
        alpha_color_box = (round(color[0]), round(color[1]), round(color[2]), 255)
        alpha_color_mask = (round(color[0]), round(color[1]), round(color[2]), 127)

        # Draw the bounding box on the image
        draw.rectangle([(int(x1), int(y1)), (int(x1 + w), int(y1 + h))], outline=alpha_color_box)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        font = ImageFont.truetype('arial', 10)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left
        label_height = bottom - top

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        draw.rectangle([label_x, label_y - label_height, label_x + label_width, label_y + label_height],
                       fill=alpha_color_mask,
                       outline=alpha_color_box)

        # Draw the label text on the image
        draw.text((label_x, label_y), label, (0, 0, 0), font=font)

        # Draw the segmentation polygon on the image
        min_x = min(polygon, key=lambda x: x[0])
        min_y = min(polygon, key=lambda x: x[1])
        updated_polygon = [(round(coord[0] + x1 - min_x[0]), round(coord[1] + y1 - min_y[1])) for coord in polygon]
        draw.polygon(updated_polygon, fill=alpha_color_mask)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_mask(self, row, box, img_width, img_height):
        """
        Function extracts segmentation mask for object in a row
        :param row: Row with object
        :param box: Bounding box of the object [x1,y1,x2,y2]
        :param img_width: Width of original image
        :param img_height: Height of original image
        :return: Segmentation mask as NumPy array
        """
        mask = row.reshape(160, 160)
        mask = self.sigmoid(mask)
        mask = (mask > 0.5).astype('uint8') * 255
        x1, y1, x2, y2 = box
        mask_x1 = round(x1 / img_width * 160)
        mask_y1 = round(y1 / img_height * 160)
        mask_x2 = round(x2 / img_width * 160)
        mask_y2 = round(y2 / img_height * 160)
        mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
        img_mask = Image.fromarray(mask, 'L')
        img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
        mask = np.array(img_mask)
        return mask

    def get_polygon(self, mask):
        """
        Function calculates bounding polygon based on segmentation mask
        :param mask: Segmentation mask as Numpy Array
        :return:
        """
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polygon = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
        return polygon

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # convert image to Pillow format
        img = Image.fromarray(input_image)
        # Transpose and squeeze the output to match the expected shape
        len_classes = len(self.classes)
        output0 = output[0].astype('float')
        output1 = output[1].astype('float')
        output0 = output0[0].transpose()
        masks = output0[:, (4 + len_classes):]
        output1 = output1.reshape(32, 160 * 160)
        masks = masks @ output1
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        mask_arrays = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:(4 + len_classes)]
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                # Add the mask arrays to the respective lists
                mask_array = self.get_mask(masks[i], (left, top, left + width, top + height), self.img_width,
                                           self.img_height)
                mask_arrays.append(mask_array)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, polygon coordinates from the mask and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            mask = mask_arrays[i]
            polygon = self.get_polygon(mask)

            # Draw the detection on the input image
            self.draw_detections(img, box, polygon, score, class_id)

        # Return the modified input image
        return np.asarray(img)

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n-seg.onnx', help='Input your ONNX model.')
    parser.add_argument('--img', type=str, default=str(ASSETS / 'bus.jpg'), help='Path to input image.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')

    # Create an instance of the Yolov8 class with the specified arguments
    detection = Yolov8(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    output_image = detection.main()

    # Display the output image in a window
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow('Output', output_image)

    # Wait for a key press to exit
    cv2.waitKey(0)
