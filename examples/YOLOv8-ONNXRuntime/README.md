# YOLOv8 - ONNX Runtime

This repository provides an example implementation for running [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models using the [ONNX Runtime](https://onnxruntime.ai/). This allows for efficient inference across various hardware platforms supporting the [ONNX format](https://onnx.ai/).

## ⚙️ Installation

To get started, you'll need [Python](https://www.python.org/) installed. Then, install the necessary dependencies.

### Installing Required Dependencies

Clone the repository and install the packages listed in the `requirements.txt` file using [pip](https://pip.pypa.io/en/stable/):

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/YOLOv8-ONNXRuntime
pip install -r requirements.txt
```

### Installing ONNX Runtime Backend

You need to choose the appropriate ONNX Runtime package based on your hardware.

**GPU Acceleration (NVIDIA)**

If you have an NVIDIA GPU and want to leverage CUDA for faster inference, install the `onnxruntime-gpu` package. Ensure you have the correct [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and CUDA toolkit installed. Refer to the official [ONNX Runtime GPU documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for compatibility details.

```bash
pip install onnxruntime-gpu
```

**CPU Only**

If you don't have a compatible NVIDIA GPU or prefer CPU-based inference, install the standard `onnxruntime` package. Check the [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/) for more options.

```bash
pip install onnxruntime
```

## 🚀 Usage

Once the dependencies and the appropriate ONNX Runtime backend are installed, you can perform inference using the provided Python script.

### Exporting Your Model

Before running inference, you need a YOLOv8 model in the ONNX format (`.onnx`). You can export your trained Ultralytics YOLOv8 models using the Ultralytics CLI or Python SDK. See the [Ultralytics export documentation](https://docs.ultralytics.com/modes/export/) for detailed instructions.

Example export command:

```bash
yolo export model=yolov8n.pt format=onnx # Export yolov8n model to ONNX
```

### Running Inference

Execute the `main.py` script with the path to your ONNX model and input image. You can also adjust the confidence and [Intersection over Union (IoU)](https://www.ultralytics.com/glossary/intersection-over-union-iou) thresholds for [object detection](https://docs.ultralytics.com/tasks/detect/).

```bash
python main.py --model yolov8n.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

- `--model`: Path to the YOLOv8 ONNX model file (e.g., `yolov8n.onnx`).
- `--img`: Path to the input image (e.g., `image.jpg`).
- `--conf-thres`: Confidence threshold for filtering detections. Only detections with a score higher than this value will be kept. Learn more about thresholds in the [performance metrics guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
- `--iou-thres`: IoU threshold for Non-Maximum Suppression (NMS). Boxes with IoU greater than this threshold will be suppressed. See the [NMS glossary entry](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) for details.

The script will process the image, perform object detection, draw bounding boxes on the detected objects, and save the output image as `output.jpg`.

## Contributing

Contributions to enhance this example or add new features are welcome! Please refer to the main [Ultralytics repository](https://github.com/ultralytics/ultralytics) for contribution guidelines. If you encounter issues or have suggestions, feel free to open an issue on the [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime) or the Ultralytics repository.
