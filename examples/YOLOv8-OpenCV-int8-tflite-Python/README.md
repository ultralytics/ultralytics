# YOLOv8 - int8-tflite Runtime

This project implements YOLOv8 using int8 tflite Runtime.

## Installation

To run this project, you need to install the required dependencies. The following instructions will guide you through the installation process.

### Installing Required Dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Installing `tflite-runtime`

You can install the tflite-runtime package using the following command in order to load tflite model:

```bash
pip install tflite-runtime
```

### Installing `tensorflow`

If you have an NVIDIA GPU and want to leverage GPU acceleration, you can install the tensorflow-gpu package using the following command:

```bash
pip install tensorflow-gpu
```

Note: Make sure you have the appropriate GPU drivers installed on your system.

### Installing `tensorflow` (CPU version)

If you don't have an NVIDIA GPU or prefer to use the CPU version of onnxruntime, you can install the onnxruntime package using the following command:

```bash
pip install tensorflow
```

### Usage

After successfully installing the required packages, you can run the YOLOv8 implementation using the following command:

You will need to convert the model into int8 tflite using below command:

```bash
yolo export model=yolov8n.pt imgsz=640 format=tflite int8

```

You can find int8 tflite mode in the directory `yolov8n_saved_model` at today date there will be many int8 tflite file but you should select `best_full_integer_quant` or its advised to check on https://netron.app/ to see if its quantized. Then run below command from terminal

```bash
python main.py --model best_full_integer_quant.tflite --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

Make sure to replace best_full_integer_quant.tflite with the path to your YOLOv8 tflite model file, image.jpg with the path to your input image, and adjust the confidence threshold (conf-thres) and IoU threshold (iou-thres) values as needed.

### OutPut

![image](https://github.com/wamiqraza/Attribute-recognition-and-reidentification-Market1501-dataset/blob/main/img/output.png)
