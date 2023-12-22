# YOLOv8-Segmentation-ONNXRuntime-Python Demo

This repository provides a Python demo for performing segmentation with YOLOv8 using ONNX Runtime, highlighting the interoperability of YOLOv8 models without the need for the full PyTorch stack.

## Features

- **Framework Agnostic**: Runs segmentation inference purely on ONNX Runtime without importing PyTorch.
- **Efficient Inference**: Supports both FP32 and FP16 precision for ONNX models, catering to different computational needs.
- **Ease of Use**: Utilizes simple command-line arguments for model execution.
- **Broad Compatibility**: Leverages Numpy and OpenCV for image processing, ensuring broad compatibility with various environments.

## Installation

Install the required packages using pip. You will need `ultralytics` for exporting YOLOv8-seg ONNX model and using some utility functions, `onnxruntime-gpu` for GPU-accelerated inference, and `opencv-python` for image processing.

```bash
pip install ultralytics
pip install onnxruntime-gpu  # For GPU support
# pip install onnxruntime    # Use this instead if you don't have an NVIDIA GPU
pip install numpy
pip install opencv-python
```

## Getting Started

### 1. Export the YOLOv8 ONNX Model

Export the YOLOv8 segmentation model to ONNX format using the provided `ultralytics` package.

```bash
yolo export model=yolov8s-seg.pt imgsz=640 format=onnx opset=12 simplify
```

### 2. Run Inference

Perform inference with the exported ONNX model on your images.

```bash
python main.py --model-path <MODEL_PATH> --source <IMAGE_PATH>
```

### Example Output

After running the command, you should see segmentation results similar to this:

<img src="https://user-images.githubusercontent.com/51357717/279988626-eb74823f-1563-4d58-a8e4-0494025b7c9a.jpg" alt="Segmentation Demo" width="800">

## Advanced Usage

For more advanced usage, including real-time video processing, please refer to the `main.py` script's command-line arguments.

## Contributing

We welcome contributions to improve this demo! Please submit issues and pull requests for bug reports, feature requests, or submitting a new algorithm enhancement.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.

## Acknowledgments

- The YOLOv8-Segmentation-ONNXRuntime-Python demo is contributed by GitHub user [jamjamjon](https://github.com/jamjamjon).
- Thanks to the ONNX Runtime community for providing a robust and efficient inference engine.
