---
comments: true
---

# Vision Transformers

Vit models currently support Python environment:

```python
from ultralytics.vit import SAM

# from ultralytics.vit import MODEL_TYPe

model = SAM("sam_b.pt")
model.info()  # display model information
model.predict(...)  # train the model
```

# Segment Anything

## About

## Supported Tasks

| Model Type | Pre-trained Weights | Tasks Supported       |
|------------|---------------------|-----------------------|
| sam base   | `sam_b.pt`          | Instance Segmentation |
| sam large  | `sam_l.pt`          | Instance Segmentation |

## Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :x:                |
| Training   | :x:                |
