# yolov8/yolov5 Inference C++

Usage:
```
git clone https://github.com/JustasBart/yolov8_CPP_Inference_OpenCV_ONNX
cd yolov8_CPP_Inference_OpenCV_ONNX
```
Edit the **main.cpp** to change the **projectBasePath** to match your user.
```
mkdir build
cd build
cmake ..
mak
./Yolov8CPPInference
```

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

This repository is based on OpenCVs dnn API to run an ONNX exported model of either yolov5/yolov8 (In theory should work for yolov6 and yolov7 but not tested). Note that for this example the networks are exported as rectangular (640x480) resolutions, but it would work for any resolution that you export as although you might want to use the letterBox approach for square images depending on your use-case.

The **main** branch version is based on using Qt as a GUI wrapper the main interest here is the **Inference** class file which shows how to transpose yolov8 models to work as yolov5 models.
