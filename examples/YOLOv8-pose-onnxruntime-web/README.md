# YOLOv8-pose with onnxruntime-web

Pose Detection application right in your browser. Serving YOLOv8 in browser using onnxruntime-web with `wasm` backend.

## Usage

```bash
git clone ultralytics
cd ultralytics/examples/yolov8-pose-onnxruntime-web
yarn install # Install dependencies
```

Copy `yolov8*.onnx` to `./public/model` Update `modelName` in `App.jsx` to your model name

```jsx
...
// configs
const modelName = "yolov8*.onnx"; // change to your model name
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.4;
const scoreThreshold = 0.2;
...
```

```bash
yarn start # Start dev server
yarn build # Build for productions
```

## Models

**Exporting YOLOv8 and YOLOv5 Models**

> :warning: **Size Overload** : used YOLOv8n model in this repo is the smallest with size of 13 MB, so other models is definitely bigger than this which can cause memory problems on browser.

```python
  from ultralytics import YOLO

  # Load a model
  model = YOLO("yolov8n.pt")  # load an official model

  # Export the model
  model.export(format="onnx")
```

**NMS**

ONNX model to perform NMS operator \[CUSTOM\].
