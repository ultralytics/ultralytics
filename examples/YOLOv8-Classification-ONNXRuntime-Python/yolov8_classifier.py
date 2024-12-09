import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8Classifier:
    """YOLOv8 model for image classification and visualization."""

    def __init__(self, model_path, image_path):
        """
        Initializes the YOLOv8Classifier instance for classification and visualization.

        Args:
            model_path (str): Path to the ONNX model file.
            image_path (str): Path to the input image file.
        """
        self.image_path = image_path
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        # Load class names from the ImageNet dataset
        self.class_names = yaml_load(check_yaml("ImageNet.yaml"))["names"]

    def annotate_class_label(self, class_index):
        """
        Annotates the class label on the image.

        Args:
            class_index (int): Index of the detected class to annotate.
        """
        # Draw the class label text on the image
        label = f"{self.class_names[class_index]}"
        cv2.putText(self.image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def preprocess_image(self):
        """
        Preprocesses the input image for model inference.

        Returns:
            np.array: Preprocessed image data ready for inference.
        """
        # Read the image from the given path
        self.image = cv2.imread(self.image_path)

        # Get the height and width of the image
        self.image_height, self.image_width = self.image.shape[:2]

        # Convert the image from BGR to RGB color space
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Resize the image to the input dimensions
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Normalize the image to the range [0, 1]
        image_normalized = np.array(image_resized) / 255.0

        # Transpose the image to channel-first format
        image_transposed = np.transpose(image_normalized, (2, 0, 1))

        # Expand dimensions to add batch size and convert to float32
        image_input = np.expand_dims(image_transposed, axis=0).astype(np.float32)

        # Return the preprocessed image input
        return image_input

    def process_output(self, model_output):
        """
        Processes the model's output to annotate the class label on the image.

        Args:
            model_output (np.array): Raw output from the model.

        Returns:
            np.array: Image with annotated class label.
        """
        # Generate binary output based on the model's score threshold
        out = [1 if score > 0.5 else 0 for score in model_output[0][0]]

        # Find the index of the detected class
        detected_class_index = out.index(1)

        # Annotate the image with the detected class label
        self.annotate_class_label(detected_class_index)

        # Return the annotated image
        return self.image

    def classify_image(self):
        """
        Executes the classification on the input image using the ONNX model.

        Returns:
            np.array: Output image with annotations.
        """
        # Preprocess the image for model input
        image_data = self.preprocess_image()

        # Run the model inference
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # Process and return the model output
        return self.process_output(model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo8n-cls.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--img", type=str, default="image.jpg", help="Path to the input image.")
    args = parser.parse_args()

    # Check for dependencies and set up ONNX runtime
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Create the classifier instance with specified parameters
    classifier = YOLOv8Classifier(args.model, args.img)

    # Perform classification and get the output image
    annotated_image = classifier.classify_image()

    # Display the annotated output image
    cv2.namedWindow("Annotated Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Annotated Output", annotated_image)
    cv2.waitKey(0)
