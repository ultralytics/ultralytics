# YOLOv8 - Tflite Runtime

This project implements YOLOv8 using tflite runtime for edge devices

## Installation

To run this project, you need to install the required dependencies. The following instructions will guide you through the installation process.

### Installing Required Dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Installing `tflite-runtime` 

```bash
pip install tflite-runtime
```

### Model Export:

```
yolo model=yolov8s.pt format=tflite imgsz=320
```
To export model in quantized int8 format:

```
yolo model=yolov8s.pt format=tflite imgsz=320 int8=True
```

### Usage

After successfully installing the required packages, you can run the YOLOv8 implementation using the following command:

```bash
python main.py --model yolov8s_float16.tflite --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```
