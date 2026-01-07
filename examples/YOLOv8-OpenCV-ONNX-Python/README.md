# Ultralytics YOLOv8 Object Detection with OpenCV and ONNX

This example demonstrates how to implement [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) object detection using [OpenCV](https://opencv.org/) in [Python](https://www.python.org/), leveraging the [ONNX (Open Neural Network Exchange)](https://onnx.ai/) model format for efficient inference.

## 🚀 Getting Started

Follow these simple steps to get the example running on your local machine.

1.  **Clone the Repository:**
    If you haven't already, clone the Ultralytics repository to access the example code:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-OpenCV-ONNX-Python/
    ```

2.  **Install Requirements:**
    Install the necessary Python packages listed in the `requirements.txt` file. We recommend using a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Detection Script:**
    Execute the main Python script, specifying the ONNX model path and the input image.
    ```bash
    python main.py --model yolov8n.onnx --img image.jpg
    ```
    The script will perform object detection on `image.jpg` using the `yolov8n.onnx` model and display the results.

## 🛠️ Exporting Your Model

If you want to use a different Ultralytics YOLOv8 model or one you've trained yourself, you need to export it to the ONNX format first.

1.  **Install Ultralytics:**
    If you don't have it installed, get the latest `ultralytics` package:

    ```bash
    pip install ultralytics
    ```

2.  **Export the Model:**
    Use the `yolo export` command to convert your desired model (e.g., `yolov8n.pt`) to ONNX. Ensure you specify `opset=12` or higher for compatibility with OpenCV's DNN module. You can find more details in the Ultralytics [Export documentation](https://docs.ultralytics.com/modes/export/).
    ```bash
    yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
    ```
    This command will generate a `yolov8n.onnx` file (or the corresponding name for your model) in your working directory. You can then use this `.onnx` file with the `main.py` script.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request to the main [Ultralytics repository](https://github.com/ultralytics/ultralytics). Thank you for helping us make Ultralytics YOLO even better!
