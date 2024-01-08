# YOLOv8 - Int8-TFLite Runtime

Welcome to the YOLOv8 Int8 TFLite Runtime for efficient and optimized object detection project. This README provides comprehensive instructions for installing and using our YOLOv8 implementation.

## Installation

Ensure a smooth setup by following these steps to install necessary dependencies.

### Installing Required Dependencies

Install all required dependencies with this simple command:

```bash
pip install -r requirements.txt
```

### Installing `tflite-runtime`

To load TFLite models, install the `tflite-runtime` package using:

```bash
pip install tflite-runtime
```

### Installing `tensorflow-gpu` (For NVIDIA GPU Users)

Leverage GPU acceleration with NVIDIA GPUs by installing `tensorflow-gpu`:

```bash
pip install tensorflow-gpu
```

**Note:** Ensure you have compatible GPU drivers installed on your system.

### Installing `tensorflow` (CPU Version)

For CPU usage or non-NVIDIA GPUs, install TensorFlow with:

```bash
pip install tensorflow
```

## Usage

Follow these instructions to run YOLOv8 after successful installation.

Convert the YOLOv8 model to Int8 TFLite format:

```bash
yolo export model=yolov8n.pt imgsz=640 format=tflite int8
```

Locate the Int8 TFLite model in `yolov8n_saved_model`. Choose `best_full_integer_quant` or verify quantization at [Netron](https://netron.app/). Then, execute the following in your terminal:

```bash
python main.py --model yolov8n_full_integer_quant.tflite --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

Replace `best_full_integer_quant.tflite` with your model file's path, `image.jpg` with your input image, and adjust the confidence (conf-thres) and IoU thresholds (iou-thres) as necessary.

### Output

The output is displayed as annotated images, showcasing the model's detection capabilities:

![image](https://github.com/wamiqraza/Attribute-recognition-and-reidentification-Market1501-dataset/blob/main/img/bus.jpg)
