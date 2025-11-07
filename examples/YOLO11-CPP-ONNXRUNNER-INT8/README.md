# Android ONNX INT8 (NNAPI/ORT)

## export way

### first use export_tflite.py to export best int8 tflite model

1、in root of ultralytics dir create "datasets/YOLODataset" dir and set train in this dir

2、set trained best.pt in root of ultralytics dir

3、python export_tflite.py

### next cov best int8 tflite to onnx

python -m tf2onnx.convert --tflite best_int8.tflite --output best_int8.onnx --opset 18

## add info

use change_onnx_mate.py add onnx mate

## use model

this dir is an android project if you want run this demo, please do some prepare and use android studio open it

### prepare onnxruntime

you need down onnxruntime for android from https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar and unzip it copy "jni" and "headers" into "onnxruntime_sdk"

### prepare opencv

you need down opencv for android from https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-android-sdk.zip and unzip it copy all from ”OpenCV-android-sdk“ into OpenCV-android-sdk

### copyright

app/src/main/assets/test_2400_1080.jpg is a screenshot from the king game, and its copyright belongs to Tencent. It is used here only to demonstrate the effectiveness of the model.
