---
comments: true
description: Learn about the Vision Transformer (ViT) and segment anything with SAM models. Train and use pre-trained models with Python API.
---

# Vision Transformers

Vit models currently support Python environment:

```python
from ultralytics.vit import SAM

# from ultralytics.vit import MODEL_TYPE

model = SAM("sam_b.pt")
model.info()  # display model information
model.predict(...)  # predict
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