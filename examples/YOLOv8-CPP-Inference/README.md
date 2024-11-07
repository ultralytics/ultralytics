# YOLOv8/YOLOv5 Inference C++

This example demonstrates how to perform inference using YOLOv8 and YOLOv5 models in C++ with OpenCV's DNN API.

On the basis of OpenCV DNN, the optional extension is performed using TensorRT to run YOLO models.I will explain with two examples: OpenCV-Onnx and TensorRT.

## OpenCV-Onnx Example(default)

### Usage

first,clone ultralytics repo

```bash
git clone ultralytics
cd ultralytics
pip install .
cd examples/YOLOv8-CPP-Inference
```

second,change the CMakeLists.txt at 14 row

```cmake
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
```

change it as your own CUDA path , linux system like "/usr/local/cuda".And then

```bash
# Add a **yolov8\_.onnx** and/or **yolov5\_.onnx** model(s) to the ultralytics folder.
# Edit the **main.cpp** to change the **projectBasePath** to match your user.

# Note that by default the CMake file will try to import the CUDA library to be used with the OpenCVs dnn (cuDNN) GPU Inference.
# If your OpenCV build does not use CUDA/cuDNN you can remove that import call and run the example on CPU.

mkdir build
cd build
cmake ..
make

# run
./build/OnnxExample
```

### Exporting YOLOv8 and YOLOv5 Models

To export YOLOv8 models:

```bash
yolo export model=yolov8s.pt imgsz=480,640 format=onnx opset=12
```

To export YOLOv5 models:

```bash
python3 export.py --weights yolov5s.pt --img 480 640 --include onnx --opset 12
```

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

This repository utilizes OpenCV's DNN API to run ONNX exported models of YOLOv5 and YOLOv8. In theory, it should work for YOLOv6 and YOLOv7 as well, but they have not been tested. Note that the example networks are exported with rectangular (640x480) resolutions, but any exported resolution will work. You may want to use the letterbox approach for square images, depending on your use case.

The **main** branch version uses Qt as a GUI wrapper. The primary focus here is the **Inference** class file, which demonstrates how to transpose YOLOv8 models to work as YOLOv5 models.

## TensorRT Example(Optional)

Environment

- TensorRT 10.0.1.6
- OpenCV 4.5.5
- Cuda 11.8

After installing the above dependencies, please modify the following fields in the CMakeLists.txt file at 12 and 14 row

```cmake
if(WITH_TENSORRT)
    message("Build TensorRT example!")
    set(TensorRT_DIR "/usr/local/tensorrt-10.0.1.6")
endif()
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
```

change them to TensorRT and Cuda project path,so that add dependencies.

### Build

```bash
# if you want to build tensorrt example ,it is necessary to add -DWITH_TENSORRT=ON.
cmake -S . -B build -DWITH_TENSORRT=ON
cmake --build build --config release
```

### YOLOv8

To export the Yolov8 onnx model files

```bash
yolo export model=yolov8n.pt imgsz=640,480 format=onnx opset=16
```

Then,convert the onnx file to engine file like

```bash
trtexec \
--onnx=./pretrain/yolov8n.onnx \
--saveEngine=./pretrain/yolov8n.engine
```

run command

```bash
./build/Release/example.exe ./yolov8n.engine ../../assets/bug.jpg 640 480
```

- ./yolov8n.engine the path of yolov8 model
- ../../assets/bug.jpg the path of input image
- 640 the height of input image
- 480 the width of input image

and the result image
!["result"](https://github.com/user-attachments/assets/bc0736a9-238a-4420-a02f-b522979140b1)
