# yolov8/yolov5 Inference C++

Usage:

```
# git clone ultralytics
pip install .
cd examples/cpp_

Add a **yolov8\_.onnx** and/or **yolov5\_.onnx** model(s) to the ultralytics folder.
Edit the **main.cpp** to change the **projectBasePath** to match your user.

Note that by default the CMake file will try and import the CUDA library to be used with the OpenCVs dnn (cuDNN) GPU Inference.
If your OpenCV build does not use CUDA/cuDNN you can remove that import call and run the example on CPU.

mkdir build
cd build
cmake ..
make
./Yolov8CPPInference
```

To export yolov8 models:

```
yolo export \
model=yolov8s.pt \
imgsz=[480,640] \
format=onnx \
opset=12
```

To export yolov5 models:

```
python3 export.py \
--weights yolov5s.pt \
--img 480 640 \
--include onnx \
--opset 12
```

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

This repository is based on OpenCVs dnn API to run an ONNX exported model of either yolov5/yolov8 (In theory should work for yolov6 and yolov7 but not tested). Note that for this example the networks are exported as rectangular (640x480) resolutions, but it would work for any resolution that you export as although you might want to use the letterBox approach for square images depending on your use-case.

The **main** branch version is based on using Qt as a GUI wrapper the main interest here is the **Inference** class file which shows how to transpose yolov8 models to work as yolov5 models.
