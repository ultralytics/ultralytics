# YOLOV8-ONNXRuntime

Implementation YOLOv8 inference using ONNXRuntime.

## Requirements

- ONNXRuntime
- OpenCV
- Numpy

## Demo

Yolov8-n inference with image

```shell
python yolov8_onnx_inference.py image /path/to/yolov8n_coco_640.onnx /path/to/ultralytics/assets/bus.jpg --input_size 640

# output
input info:  NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
output info:  NodeArg(name='output0', type='tensor(float)', shape=[1, 84, 8400])
person: 0.875    [670, 376, 809, 878]
person: 0.869    [48, 399, 244, 902]
bus: 0.863    [21, 229, 798, 754]
person: 0.82    [221, 405, 344, 857]
stop sign: 0.346    [0, 254, 32, 325]
person: 0.301    [0, 551, 67, 873]
```

Yolov8-n inference with camera

```shell
python yolov8_onnx_inference.py stream /path/to/yolov8n_coco_640.onnx 0 --input_size 640 --show
```

Make sure to include **'opset=12'** when exporting onnx model using ultralytics command tool.
