<h1 align="center">Yolov8seg-NCNN-Android-inference</h1>

<p align="center">
  <img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B">
  <img alt="Android Studio" src="https://img.shields.io/badge/Android-Studio-02550?logo=Android">
  <img alt="Tencent ncnn" src="https://img.shields.io/badge/Tencent-ncnn-red?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAAmASURBVHhe7d0hblBdFwVQ%2BGfSoEhQjAA6BmRVg8PhCSGMAUeKqWQKtAyCBEXq8Wg%2B%2FjFscbOzF%2F68d846l51X0dvHf%2F78%2Bfvo4L9nz55Fb%2F%2Fx40dUr5hAItB%2Bfv%2BXDK%2BWAIFuAQHQvT%2FdE4gEBEDEp5hAt4AA6N6f7glEAgIg4lNMoFtAAHTvT%2FcEIgEBEPEpJtAtIAC696d7ApGAAIj4FBPoFhAA3fvTPYFIQABEfIoJdAsIgO796Z5AJCAAIj7FBLoFBED3%2FnRPIBIQABGfYgLdAo%2FT%2BwDS34c%2Bzec%2BgdMbOPv%2B9fPrC%2BDs%2BfN2AkcFBMBRfi8ncFZAAJz193YCRwUEwFF%2BLydwVkAAnPX3dgJHBQTAUX4vJ3BWQACc9fd2AkcFBMBRfi8ncFZAAJz193YCRwUEwFF%2BLydwVkAAnPX3dgJHBQTAUX4vJ3BWQACc9fd2AkcFBMBRfi8ncFZAAJz193YCRwWO3wfw7t27owAfP36M3u8%2BgYgvLk5%2Fn3%2F9%2FPkCiI%2BgBxDoFRAAvbvTOYFYQADEhB5AoFdAAPTuTucEYgEBEBN6AIFeAQHQuzudE4gFBEBM6AEEegUEQO%2FudE4gFhAAMaEHEOgVEAC9u9M5gVhAAMSEHkCgV0AA9O5O5wRiAQEQE3oAgV4BAdC7O50TiAUEQEzoAQR6BR5fXFz8Tdp%2F9epVUv7o6dOnUf3l5WVUf3d3F9W7TyDie3T69%2FlPn5%2BfP39GgF%2B%2Ffo3qfQFEfIoJdAsIgO796Z5AJCAAIj7FBLoFBED3%2FnRPIBIQABGfYgLdAgKge3%2B6JxAJCICITzGBbgEB0L0%2F3ROIBARAxKeYQLeAAOjen%2B4JRAICIOJTTKBbQAB070%2F3BCIBARDxKSbQLSAAuvenewKRgACI%2BBQT6BY4fh%2FAmzdvqgWvr6%2Bj%2Fh8eHqL69uJ%2F91FEI9zc3ET1p4s%2FffoUteA%2BgIhPMYFtAT8CbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FW0AAbO%2Ff9OMCAmD8ABh%2FWyC%2BD%2BDz58%2BR4JMnT6J6xQSaBX79%2BhW1%2F%2Fr166jeF0DEp5hAt4AA6N6f7glEAgIg4lNMoFtAAHTvT%2FcEIgEBEPEpJtAtIAC696d7ApGAAIj4FBPoFhAA3fvTPYFIQABEfIoJdAsIgO796Z5AJCAAIj7FBLoFBED3%2FnRPIBIQABGfYgLdAgKge3%2B6JxAJCICITzGBboH4PoD7%2B%2FtuAd0TKBZ4%2BfJl1L0vgIhPMYFuAQHQvT%2FdE4gEBEDEp5hAt4AA6N6f7glEAgIg4lNMoFtAAHTvT%2FcEIgEBEPEpJtAtIAC696d7ApGAAIj4FBPoFhAA3fvTPYFIQABEfIoJdAsIgO796Z5AJCAAIj7FBLoFBED3%2FnRPIBIQABGfYgLdAvX3AXz48KF7A2H379%2B%2FD5%2BQlfM%2F6%2B8%2BgOz8qiYwLeBHgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwsIgOn1G35dQACsnwDzTwvM3wfw%2Ffv36AA8PDxE9d%2B%2BfYvqnzx5EtWnxel9AF%2B%2BfIlauLi4iOpfvHgR1Z%2B%2Bj8F9ANH6FBPYFvAjwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUEwPb%2BTT8uIADGD4DxtwUe%2F%2Fv76n8Tgvv7%2B6Q8rk3%2FPn3cwOEHnP779Kn%2F1dXVUcHb29vo%2Faf9X758GfXvCyDiU0ygW0AAdO9P9wQiAQEQ8Skm0C0gALr3p3sCkYAAiPgUE%2BgWEADd%2B9M9gUhAAER8igl0CwiA7v3pnkAkIAAiPsUEugUEQPf%2BdE8gEhAAEZ9iAt0CAqB7f7onEAkIgIhPMYFuAQHQvT%2FdE4gEBEDEp5hAt0B8H8C%2F%2BwSqBW5ubqr713wmcH19nT3gcPXDw0PUgS%2BAiE8xgW4BAdC9P90TiAQEQMSnmEC3gADo3p%2FuCUQCAiDiU0ygW0AAdO9P9wQiAQEQ8Skm0C0gALr3p3sCkYAAiPgUE%2BgWEADd%2B9M9gUhAAER8igl0CwiA7v3pnkAkIAAiPsUEugUEQPf%2BdE8gEhAAEZ9iAt0Cjy8vL%2F8mI7T%2Fffdk9v%2FXuk8gFczq09%2FnXz%2B%2FvgCy86eaQLWAAKhen%2BYJZAICIPNTTaBaQABUr0%2FzBDIBAZD5qSZQLSAAqteneQKZgADI%2FFQTqBYQANXr0zyBTEAAZH6qCVQLCIDq9WmeQCYgADI%2F1QSqBQRA9fo0TyATEACZn2oC1QICoHp9mieQCQiAzE81gWqB%2BvsAnj9%2FHi3g7du3Ub37ACK%2BuLj9PoDT59cXQHwEPYBAr4AA6N2dzgnEAgIgJvQAAr0CAqB3dzonEAsIgJjQAwj0CgiA3t3pnEAsIABiQg8g0CsgAHp3p3MCsYAAiAk9gECvgADo3Z3OCcQCAiAm9AACvQICoHd3OicQCwiAmNADCPQKCIDe3emcQCwgAGJCDyDQKxDfB5D%2BPvzv378jvR8%2FfkT1t7e3UX06f%2FRyxY%2FS%2BwDS%2FbWfX18A%2FhMRGBYQAMPLNzoBAeAMEBgWEADDyzc6AQHgDBAYFhAAw8s3OgEB4AwQGBYQAMPLNzoBAeAMEBgWEADDyzc6AQHgDBAYFhAAw8s3OgEB4AwQGBYQAMPLNzoBAeAMEBgWiO8DuLq6ivguLy%2Bj%2Bru7u6g%2BvQ%2Fg4eEher%2FiTODi4iJ6wPr59QUQHR%2FFBLoFBED3%2FnRPIBIQABGfYgLdAgKge3%2B6JxAJCICITzGBbgEB0L0%2F3ROIBARAxKeYQLeAAOjen%2B4JRAICIOJTTKBbQAB070%2F3BCIBARDxKSbQLSAAuvenewKRgACI%2BBQT6BYQAN370z2BSEAARHyKCXQL%2FAcvxcGwUVQqIgAAAABJRU5ErkJggg%3D%3D"></img>
</p>

This is a sample object segmentation project, it depends on yolov8, ncnn library and opencv

## Prerequisites

https://github.com/ultralytics/ultralytics

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

## How to build and run

### Step 1

https://github.com/ultralytics/ultralytics

Install ultralytics yolov8.

```bash
pip install ultralytics
```

Convert your model.

```bash
yolo export model=yolov8n-seg.pt format=ncnn
```

Copy **_yolov8n-seg.param_** and **_yolov8n-seg.bin_** to **app/src/main/jni/assets**

_For your own model :_ Change the name of the model in **_yolov8ncnn.cpp_** line 184. Change the classes names in **_yolo.h_**

### Step 2

https://github.com/Tencent/ncnn/releases

- Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
- Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### Step 3

https://github.com/nihui/opencv-mobile

- Download opencv-mobile-XYZ-android.zip
- Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### Step 4

- Open this project with Android Studio, build it and enjoy!

## Some notes

- Android ndk camera is used for best efficiency
- Crash may happen on very old devices for lacking HAL3 camera interface
- All models are manually modified to accept dynamic input shape
- Most small models run slower on GPU than on CPU, this is common
- FPS may be lower in dark environment because of longer camera exposure time

## Screenshot

<img src="screenshot.png" width=50%>

## Reference：

https://github.com/nihui/ncnn-android-nanodet https://github.com/Tencent/ncnn https://github.com/ultralytics/ultralytics/pull/3529
