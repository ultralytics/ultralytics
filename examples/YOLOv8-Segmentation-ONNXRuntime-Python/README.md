# YOLOv8-Segmentation-ONNXRuntime-Python Demo

This repository provides a [Python](https://www.python.org/) demo for performing instance segmentation with [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) using [ONNX Runtime](https://onnxruntime.ai/). It highlights the interoperability of YOLOv8 models, allowing inference without requiring the full [PyTorch](https://pytorch.org/) stack. This approach is ideal for deployment scenarios where minimal dependencies are preferred. Learn more about the [segmentation task](https://docs.ultralytics.com/tasks/segment/) on our documentation.

## ✨ Features

- **Framework Agnostic**: Runs segmentation inference purely on ONNX Runtime without importing PyTorch.
- **Efficient Inference**: Supports both FP32 and [half-precision](https://www.ultralytics.com/glossary/half-precision) (FP16) for [ONNX](https://onnx.ai/) models, catering to different computational needs and optimizing [inference latency](https://www.ultralytics.com/glossary/inference-latency).
- **Ease of Use**: Utilizes simple command-line arguments for straightforward model execution.
- **Broad Compatibility**: Leverages [NumPy](https://numpy.org/) and [OpenCV](https://opencv.org/) for image processing, ensuring wide compatibility across various environments.

## 🛠️ Installation

Install the required packages using pip. You will need [`ultralytics`](https://github.com/ultralytics/ultralytics) for exporting the YOLOv8-seg ONNX model and using some utility functions, [`onnxruntime-gpu`](https://pypi.org/project/onnxruntime-gpu/) for GPU-accelerated inference, and [`opencv-python`](https://pypi.org/project/opencv-python/) for image processing.

```bash
pip install ultralytics
pip install onnxruntime-gpu # For GPU support
# pip install onnxruntime # For CPU-only support
pip install numpy opencv-python
```

## 🚀 Getting Started

### 1. Export the YOLOv8 ONNX Model

First, export your Ultralytics YOLOv8 segmentation model to the ONNX format using the `ultralytics` package. This step converts the PyTorch model into a standardized format suitable for ONNX Runtime. Check our [Export documentation](https://docs.ultralytics.com/modes/export/) for more details on export options and our [ONNX integration guide](https://docs.ultralytics.com/integrations/onnx/).

```bash
yolo export model=yolov8s-seg.pt imgsz=640 format=onnx opset=12 simplify
```

### 2. Run Inference

Perform inference with the exported ONNX model on your images or video sources. Specify the path to your ONNX model and the image source using the command-line arguments.

```bash
python main.py --model yolov8s-seg.onnx --source path/to/image.jpg
```

### Example Output

After running the command, the script will process the image, perform segmentation, and display the results with bounding boxes and masks overlaid.

<img src="https://user-images.githubusercontent.com/51357717/279988626-eb74823f-1563-4d58-a8e4-0494025b7c9a.jpg" alt="YOLOv8 Segmentation ONNX Demo Output" width="800">

## 💡 Advanced Usage

For more advanced usage scenarios, such as processing video streams or adjusting inference parameters, please refer to the command-line arguments available in the `main.py` script. You can explore options like confidence thresholds and input image sizes.

## 🤝 Contributing

We welcome contributions to improve this demo! If you encounter bugs, have feature requests, or want to submit enhancements (like a new algorithm or improved processing steps), please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics). See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get involved.

## 📄 License

This project is licensed under the AGPL-3.0 License. For detailed information, please see the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file or read the full [AGPL-3.0 license text](https://opensource.org/license/agpl-v3).

## 🙏 Acknowledgments

- This YOLOv8-Segmentation-ONNXRuntime-Python demo was contributed by GitHub user [jamjamjon](https://github.com/jamjamjon).
- Thanks to the [ONNX Runtime community](https://github.com/microsoft/onnxruntime) for providing a robust and efficient inference engine.

We hope you find this demo useful! Feel free to contribute and help make it even better.
