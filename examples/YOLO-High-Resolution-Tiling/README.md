# High Resolution Tiling

Use the `tile` argument to run Ultralytics YOLO on high-resolution images and videos. The existing prediction pipeline loads frames, batches tiles, merges predictions, and saves annotated output.

```bash
pip install -U ultralytics

# Run tiled video inference and save the annotated video.
yolo predict model=yolo26n.pt source=video.mp4 tile=640 save=True

# Choose tile size independently of model input size.
yolo predict model=yolo26n.pt source=large-image.tif tile=1024 imgsz=640 batch=8
```

For Python, stream video results to keep memory bounded:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
for result in model.predict("video.mp4", tile=640, stream=True, save=True):
    print(result.boxes)
```

`tile=True` uses `imgsz` for the tile size; a positive integer selects the tile size in source pixels. Existing prediction arguments such as `device`, `show`, and `save` apply. See [High Resolution Tiling](https://docs.ultralytics.com/guides/high-resolution-tiling) for training, validation, supported tasks, and performance guidance.
