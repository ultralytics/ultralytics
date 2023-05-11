---
comments: true
description: Learn about the Vision Transformer (ViT) and segment anything with SAM models. Train and use pre-trained models with Python API.
---

# Vision Transformers

Vit models currently support Python environment:

```python
from ultralytics.vit import RTDETR

# from ultralytics.vit import MODEL_TYPE

model = RTDETR("rtdetr_l.pt")
model.info()  # display model information
model.predict(...)  # predict
```

# Realtime Detection Transformers

## About

## Supported Tasks

| Model Type          | Pre-trained Weights | Tasks Supported       |
|---------------------|---------------------|-----------------------|
| rtdetr large        | `sam_b.pt`          | Object Detection |
| rtdetr extra-large  | `sam_l.pt`          | Object Detection |

## Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :heavy_check_mark: |
| Training   | :x: (Coming soon)  