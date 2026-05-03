# Extended Backbones

This repository can expose non-standard backbones to Ultralytics YAMLs as native modules. The `draxnet-yolo26` example added here shows the pattern for adapting an external backbone into a YOLO detection model without changing the rest of the training and inference stack.

## What Was Added

- `DraxBlock` was copied from the sibling MLX project and added to [ultralytics/nn/modules/block.py](/home/ralampay/workspace/ultralytics/ultralytics/nn/modules/block.py).
- `DraxResidualBlock` and `DraxNet` were added beside it so the Drax mixer can be used as a detection backbone.
- `DraxNet` was exported through [ultralytics/nn/modules/__init__.py](/home/ralampay/workspace/ultralytics/ultralytics/nn/modules/__init__.py) and registered in [ultralytics/nn/tasks.py](/home/ralampay/workspace/ultralytics/ultralytics/nn/tasks.py) as a multi-output backbone module.
- A new model config lives at [ultralytics/cfg/models/ext/draxnet-yolo26.yaml](/home/ralampay/workspace/ultralytics/ultralytics/cfg/models/ext/draxnet-yolo26.yaml).

## Source Backbone

The reference implementation came from `/home/ralampay/workspace/mlx`:

- `mlx/modes/image_classification/models/blocks.py`
- `mlx/modes/image_classification/models/draxnet.py`

In MLX, `DraxNet` is a ResNet-18 style classifier with a configurable stage layout. Its default pattern is:

```text
basic, basic, basic, drax
```

For detection, the classifier head is removed and the backbone returns feature maps instead of logits.

## How The Ultralytics Integration Works

Ultralytics model YAMLs are parsed by `parse_model()` in `ultralytics/nn/tasks.py`. For standard single-output modules, the parser only needs to know the output channel count. Multi-output backbones are handled by returning a list and selecting the needed pyramid levels with `Index`.

The Drax integration follows that pattern:

1. `DraxNet` returns three feature maps corresponding to P3, P4, and P5.
2. Internal `1x1` projections align those outputs to `256`, `512`, and `1024` channels so the existing YOLO26 head can be reused.
3. The YAML indexes those three maps with `Index`.
4. The rest of the head remains a normal YOLO26 detection head.

Conceptually:

```text
image
  -> DraxNet
  -> [P3, P4, P5]
  -> Index
  -> YOLO26 neck/head
```

## YAML Pattern

The backbone portion of [ultralytics/cfg/models/ext/draxnet-yolo26.yaml](/home/ralampay/workspace/ultralytics/ultralytics/cfg/models/ext/draxnet-yolo26.yaml) is:

```yaml
backbone:
  - [-1, 1, DraxNet, [1024, [2, 2, 2, 2], [basic, basic, basic, drax], True, True, [256, 512, 1024]]]
  - [0, 1, Index, [256, 0]]
  - [0, 1, Index, [512, 1]]
  - [0, 1, Index, [1024, 2]]
```

Argument meaning:

- `1024`: parser-facing top-level output width used consistently with YOLO conventions
- `[2, 2, 2, 2]`: ResNet-18 stage repeat layout
- `[basic, basic, basic, drax]`: per-stage block selection
- `True, True`: enable attention and efficient attention inside `DraxBlock`
- `[256, 512, 1024]`: projected P3/P4/P5 channel sizes consumed by the YOLO head

## Extending Another Backbone

To add another custom backbone, use the same flow:

1. Implement the backbone under `ultralytics/nn/modules/`.
2. Export it in `ultralytics/nn/modules/__init__.py`.
3. Register any parser-specific behavior in `ultralytics/nn/tasks.py`.
4. If the backbone returns multiple features, make it return a list and use `Index` in YAML.
5. Match the feature-map channels and strides expected by the downstream head.
6. Add a small construction and forward-pass test before using it for training.

The key requirement is not the exact class shape. The key requirement is that the module can be instantiated from YAML and emits features with the channel counts and strides the head expects.
