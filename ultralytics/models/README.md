## Models HUB

Here are the models that are supported out-of-the-box with Ultralytics. For a detailed view and navigation, visit [model hub](<>) section of the docs.

### Usage

You can simply set the `model` parameter to any available yaml config or pretained weights

```bash
yolo task=... mode=... model=yolov5n.yaml
```

| Model              | Version/ | size (pixels) | mAPval 50-95 | Speed CPU b1 (ms) | params (M) | FLOPs @640 (B) | model file    | Pretrained Weights |
| ------------------ | -------- | ------------- | ------------ | ----------------- | ---------- | -------------- | ------------- | ------------------ |
| YOLOv5n            | v6.3     | 640           | 28.0         | 45                | 1.9        | 4.5            | yolov5n.yaml  | -                  |
| YOLOv5s            | -        | 640           | 37.4         | 98                | 7.2        | 16.5           | yolov5s.yaml  | -                  |
| YOLOv5m            | -        | 640           | 45.4         | 224               | 21.2       | 49.0           | yolov5m.yaml  | -                  |
| YOLOv5l            | -        | 640           | 49.0         | 430               | 46.5       | 109.1          | yolov5l.yaml  | -                  |
| YOLOv5x            | -        | 640           | 50.7         | 766               | 86.7       | 205.7          | yolov5x.yaml  | -                  |
| YOLOv5n6           | -        | 1280          | 36.0         | 153               | 3.2        | 4.6            | yolov5n6.yaml | -                  |
| YOLOv5s6           | -        | 1280          | 44.8         | 385               | 12.6       | 16.8           | yolov5s6.yaml | -                  |
| YOLOv5m6           | -        | 1280          | 51.3         | 887               | 35.7       | 50.0           | yolov5m6.yaml | -                  |
| YOLOv5l6           | -        | 1280          | 53.7         | 1784              | 76.8       | 111.4          | yolov5l6.yaml | -                  |
| YOLOv5x6 + \[TTA\] | -        | 1280 1536     | 55.0 55.8    | 3136 -            | 140.7 -    | 209.8 -        | yolov5x6.yaml | -                  |
