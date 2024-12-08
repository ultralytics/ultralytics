import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime_extensions

# Mapping from class ID to class name
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def apply_masks_and_draw(nms_output, individual_masks, image, fps):
    """
    Draws bounding boxes and applies masks to the image based on model outputs.

    Args:
        nms_output (np.ndarray): Array containing bounding box and class information.
        individual_masks (np.ndarray): Array containing mask data for each detection.
        image (np.ndarray): The original image/frame to annotate.
        fps (float): Frames per second to display on the image.

    Returns:
        None
    """
    original_height, original_width = image.shape[:2]

    for i, detection in enumerate(nms_output):
        # Extract bounding box coordinates and class information
        xc, yc, w, h = map(int, detection[:4])  # Center x, Center y, Width, Height
        cls = int(detection[5])  # Class ID
        conf = detection[4]  # Confidence score

        # Calculate top-left and bottom-right coordinates
        xmin = max(0, xc - w // 2)
        ymin = max(0, yc - h // 2)
        xmax = min(original_width, xc + w // 2)
        ymax = min(original_height, yc + h // 2)

        # Resize the individual mask to the original image size
        individual_mask_resized = cv2.resize(individual_masks[i], (original_width, original_height))

        # Apply threshold to create a binary mask
        _, mask_binary = cv2.threshold(individual_mask_resized, 0.5, 1, cv2.THRESH_BINARY)
        mask_binary = (mask_binary * 255).astype(np.uint8)  # Convert to uint8 for visualization
        mask_region = mask_binary[ymin:ymax, xmin:xmax]

        # Create an overlay for the mask
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_overlay[ymin:ymax, xmin:xmax, 1] = mask_region  # Apply mask to the green channel

        # Blend the original image with the mask overlay
        image = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)

        # Prepare label with class name and confidence score
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")  # Fallback to 'Class {cls}' if not found
        label = f"{class_name} - {conf:.2f}"

        # Put label above the bounding box
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display FPS on the image
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return image


def main():
    """Main function to perform webcam or video inference using the final YOLOv8 ONNX model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run webcam/video inference on the final YOLOv8 ONNX model.")
    parser.add_argument("--final-model", type=str, required=True, help="Path to the final ONNX model file.")
    parser.add_argument("--input-width", type=int, default=640, help="Input width for the model")
    parser.add_argument("--input-height", type=int, default=640, help="Input height for the model")
    parser.add_argument(
        "--video", type=str, default="", help="Path to a video file. If not provided, webcam will be used."
    )
    args = parser.parse_args()

    # Initialize ONNX Runtime session with necessary providers and options
    providers = ["CPUExecutionProvider"]  # Change to 'CUDAExecutionProvider' if GPU is available
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(args.final_model, providers=providers, sess_options=session_options)

    # Get input name for the model
    inname = [i.name for i in session.get_inputs()]

    # Initialize video capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {args.video} not found or cannot be opened.")
    else:
        cap = cv2.VideoCapture(0)  # Use default webcam
        if not cap.isOpened():
            raise RuntimeError("Webcam could not be opened.")

    print("Starting inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        start_time = time.time()

        # Prepare input frame
        # Note: Do NOT resize here if the model includes pre-processing steps
        # Let the model handle resizing to prevent dimension mismatches
        frame = cv2.resize(frame, (args.input_width, args.input_height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Create input dictionary for the model
        inp = {inname[0]: rgb_frame}

        # Run inference
        try:
            outputs = session.run(["nms_output_with_scaled_boxes_and_masks", "final_masks", "input_image_mask"], inp)
        except Exception as e:
            print(f"Error during inference: {e}")
            continue

        nms_output_with_scaled_boxes_and_masks = outputs[0]
        final_masks = outputs[1]

        # Apply masks and draw bounding boxes
        if nms_output_with_scaled_boxes_and_masks.shape[0] != 0:
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            annotated_frame = apply_masks_and_draw(nms_output_with_scaled_boxes_and_masks, final_masks, frame, fps)
        else:
            annotated_frame = frame

        # Display the annotated frame
        cv2.imshow("Result", annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting inference.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
