# RT-DETR Object Detection with ONNX Runtime

This project demonstrates how to run Ultralytics [RT-DETR models](https://docs.ultralytics.com/models/rtdetr/) using the [ONNX Runtime](https://onnxruntime.ai/) inference engine in [Python](https://www.python.org/). It provides a straightforward example for performing [object detection](https://docs.ultralytics.com/tasks/detect/) with RT-DETR models that have been exported to the [ONNX format](https://onnx.ai/), a standard for representing [machine learning models](https://www.ultralytics.com/glossary/machine-learning-ml). RT-DETR, or Real-Time DEtection TRansformer, offers efficient and accurate object detection capabilities, detailed further in the [RT-DETR research paper](https://arxiv.org/abs/2304.08069).

## ⚙️ Installation

To get started, you'll need to install the necessary dependencies. Follow the steps below.

### Installing Required Dependencies

Install the core requirements using [pip](https://pip.pypa.io/en/stable/) and the provided `requirements.txt` file this will install CPU-based inference, install the standard **`onnxruntime`** package. This version utilizes CPU resources for model execution. See the [ONNX Runtime Execution Providers documentation](https://onnxruntime.ai/docs/execution-providers/) for more information on different execution options.

```bash
pip install -r requirements.txt
```

### Installing `onnxruntime-gpu` (Optional)

For accelerated inference using an NVIDIA GPU, install the **`onnxruntime-gpu`** package. Ensure you have the correct [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed first. Consult the official [ONNX Runtime GPU documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for detailed compatibility information and setup instructions.

```bash
pip install onnxruntime-gpu
```

## 🚀 Usage

Once the dependencies are installed, you can run inference using the `main.py` script.

Execute the script from your terminal, specifying the path to your ONNX model, the input image, and optional confidence and IoU thresholds:

```bash
python main.py --model rtdetr-l.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

**Arguments:**

- `--model`: Path to the RT-DETR [ONNX model file](https://docs.ultralytics.com/modes/export/) (e.g., `rtdetr-l.onnx`). You can easily [export Ultralytics models](https://docs.ultralytics.com/modes/export/) to ONNX format. Find more models on the [Ultralytics Models](https://docs.ultralytics.com/models/) page.
- `--img`: Path to the input image file (e.g., `image.jpg`).
- `--conf-thres`: Confidence threshold for filtering detections. Only detections with a score higher than this value will be kept. Learn more about thresholds in our guide on [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
- `--iou-thres`: [Intersection over Union (IoU)](https://www.ultralytics.com/glossary/intersection-over-union-iou) threshold used for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) to remove redundant [bounding boxes](https://www.ultralytics.com/glossary/bounding-box).

Adjust the `--conf-thres` and `--iou-thres` values based on your specific requirements for detection sensitivity and overlap removal.

## 🤝 Contributing

Contributions to enhance this example are welcome! Whether it's fixing bugs, adding new features, improving documentation, or suggesting optimizations, your input is valuable. Please refer to the Ultralytics [Contribution Guide](https://docs.ultralytics.com/help/contributing/) for detailed information on how to get started. You can also explore general guides on [contributing to open source projects](https://opensource.guide/how-to-contribute/). Thank you for helping improve the [Ultralytics](https://www.ultralytics.com/) ecosystem and its resources available on [GitHub](https://github.com/ultralytics/ultralytics)!
