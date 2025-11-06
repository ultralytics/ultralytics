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

look ott_check_for_int8.cpp and ott_check_for_int8.h

you can get some build help from CMakeLists.txt

## add

this code pass by android studio with cmake
