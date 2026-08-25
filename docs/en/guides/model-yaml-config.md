---
title: "YOLO Model YAML: Architecture Reference"
comments: true
description: Reference for the Ultralytics YOLO model YAML - nc and scales, backbone and head sections, the [from, repeats, module, args] layer format, and custom modules.
keywords: Ultralytics, YOLO, model architecture, YAML configuration, parse_model, backbone, head, scales, max_channels, custom modules, C3k2, C2PSA, TorchVision backbone
---

# Model YAML Configuration Guide

An Ultralytics model YAML is the architectural blueprint for a YOLO network: it declares the class count and scaling factors, then lists every backbone and head layer as `[from, repeats, module, args]`. Ultralytics YOLO26 builds the network straight from that file — `YOLO("my_model.yaml")` needs no Python model code.

This reference covers each section of the file, the modules you can reference by name, how [`parse_model`](../reference/nn/tasks.md) resolves those names, and how to register a custom module of your own.

<img width="1024" src="https://cdn.ul.run/i/a563b72f7404e71fb4c61b7710d28acf.avif" alt="Model YAML configuration workflow.">

## Configuration Structure

Model YAML files are organized into three main sections that work together to define the architecture.

### Parameters Section

The **parameters** section specifies the model's global characteristics and scaling behavior:

```yaml
# Parameters
nc: 80 # number of classes
scales: # compound scaling constants [depth, width, max_channels]
    n: [0.50, 0.25, 1024] # nano: half depth, quarter width
    s: [0.50, 0.50, 1024] # small: half depth, half width
    m: [0.50, 1.00, 512] # medium: half depth, full width, channels capped at 512
    l: [1.00, 1.00, 512] # large: full depth and width
    x: [1.00, 1.50, 512] # extra-large: full depth, 1.5x width
kpt_shape: [17, 3] # pose models only
```

- `nc` sets the number of classes the model predicts.
- `scales` define compound scaling factors as `[depth, width, max_channels]`, producing different size variants (nano through extra-large) from one file.
- `kpt_shape` applies to pose models. It can be `[N, 2]` for `(x, y)` keypoints or `[N, 3]` for `(x, y, visibility)`.

Shipped YAMLs also use four further top-level keys:

| Key          | Used by             | Purpose                                                                                     |
| ------------ | ------------------- | ------------------------------------------------------------------------------------------- |
| `end2end`    | YOLO26 detection family | Enables the NMS-free one-to-one head, in the detect, segment, pose and OBB configs. Absent from `yolo26-cls`, `-depth` and `-sem`. See [End-to-End Detection](end2end-detection.md). |
| `reg_max`    | YOLO26 detection family | Number of DFL bins. Those same configs ship `reg_max: 1`, which is what makes them DFL-free. |
| `channels`   | classification      | Input channel count, i.e. `1` for grayscale. Only classification reads it here — see below. |
| `activation` | `yolov6.yaml`       | Overrides the default activation for every `Conv` in the model, i.e. `torch.nn.ReLU()`.     |

!!! warning "`channels` in a detection model YAML is ignored"

    `ClassificationModel` reads the key (`self.yaml.get("channels", ch)`), but detection, segmentation, pose and OBB models are built through `_initialize_yolo_model`, which does `model.yaml["channels"] = ch` unconditionally — so the constructor argument always wins and a `channels: 1` line in such a YAML has no effect. Set the channel count on the **dataset** YAML instead: the trainer passes `ch=self.data["channels"]` when it builds the model, which is how a grayscale dataset produces a single-channel first convolution.

### How Width Scaling Actually Computes Channels

Channel counts are **not** the number written in `args`. For `parse_model`'s base-module set — the convolution and block modules such as `Conv`, `C3k2`, `C2PSA`, `C2f` and `SPPF` — each layer's output channels are clamped to `max_channels`, multiplied by `width`, then rounded up to the nearest multiple of 8:

```python
c2 = make_divisible(min(c2, max_channels) * width, 8)
```

| YAML `out_ch` | `width` | `max_channels` | Actual channels |
| ------------- | ------- | -------------- | --------------- |
| 1024          | 1.00    | 512            | 512             |
| 1024          | 1.50    | 512            | 768             |
| 1024          | 0.25    | 1024           | 256             |
| 100           | 0.50    | 1024           | 56              |

The last row is the one that surprises people: `100 x 0.5` is `50`, which rounds up to `56`. This is why `m`, `l` and `x` carry `max_channels: 512` while `n` and `s` carry `1024` — the larger scales would otherwise produce very wide layers.

The formula does not reach every layer. `Classify` is exempt so its output stays at `nc`; `Concat` sums the channels of its sources; `TorchVision` and `Index` use the bookkeeping value you wrote; and heads take `nc` rather than a channel count. Compute expected channels this way only for the base modules.

!!! tip "Reduce redundancy with `scales`"

    The `scales` parameter lets you generate multiple model sizes from a single base YAML. For instance, when you load `yolo26n.yaml`, Ultralytics reads the base `yolo26.yaml` and applies the `n` scaling factors (`depth=0.50`, `width=0.25`) to build the nano variant.

!!! note "`nc` and `kpt_shape` are dataset-dependent"

    If your dataset specifies a different `nc` or `kpt_shape`, Ultralytics will automatically override the model config at runtime to match the dataset YAML.

### Backbone and Head Architecture

The model architecture consists of backbone (feature extraction) and head (task-specific) sections:

```yaml
nc: 80

backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0: Initial convolution
    - [-1, 1, Conv, [128, 3, 2]] # 1: Downsample
    - [-1, 3, C3k2, [128, True]] # 2: Feature processing

head:
    - [-1, 1, nn.Upsample, [None, 2, nearest]] # 3: Upsample
    - [[-1, 0], 1, Concat, [1]] # 4: Skip connection to layer 0
    - [-1, 3, C3k2, [256, False]] # 5: Process features
    - [[5], 1, Detect, [nc]] # 6: Detection layer
```

Layer indices run continuously across both sections — the head does not restart at 0 — and a `from` index must name a layer that already exists. A skip connection also has to be spatially compatible: layer 0 is the only source the upsampled layer 3 can concatenate with here.

## Layer Specification Format

Every layer follows the consistent pattern: **`[from, repeats, module, args]`**

| Component   | Purpose               | Examples                                                  |
| ----------- | --------------------- | --------------------------------------------------------- |
| **from**    | Input connections     | `-1` (previous), `6` (layer 6), `[4, 6, 8]` (multi-input) |
| **repeats** | Number of repetitions | `1` (single), `3` (repeat 3 times)                        |
| **module**  | Module type           | `Conv`, `C2f`, `TorchVision`, `Detect`                    |
| **args**    | Module arguments      | `[64, 3, 2]` (channels, kernel, stride)                   |

### Connection Patterns

The `from` field creates flexible data flow patterns throughout your network:

=== "Sequential Flow"

    ```yaml
    - [-1, 1, Conv, [64, 3, 2]]    # Takes input from previous layer
    ```

=== "Skip Connections"

    ```yaml
    - [[-1, 6], 1, Concat, [1]]    # Combines current layer with layer 6
    ```

=== "Multi-Input Fusion"

    ```yaml
    - [[4, 6, 8], 1, Detect, [nc]] # Detection head using 3 feature scales
    ```

!!! note "Layer Indexing"

    Layers are indexed starting from 0. Negative indices reference previous layers (`-1` = previous layer), while positive indices reference specific layers by their position.

### Module Repetition

The `repeats` parameter creates deeper network sections:

```yaml
- [-1, 3, C2f, [128, True]] # Creates 3 consecutive C2f blocks
- [-1, 1, Conv, [64, 3, 2]] # Single convolution layer
```

The repetition count is multiplied by the `depth` scaling factor, rounded, and floored at 1 — but only when it is greater than 1, so `repeats: 1` is never scaled:

```python
n = max(round(n * depth), 1) if n > 1 else n
```

At `depth=0.33` a `repeats: 3` block collapses to a single block and `repeats: 6` becomes 2. Only modules that accept an internal repeat count consume `n` this way; any other module is simply stacked `n` times in an `nn.Sequential`.

## Available Modules

Modules are organized by functionality and defined in the [Ultralytics modules directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules). The following tables show commonly used modules by category, with many more available in the source code:

### Basic Operations

| Module        | Purpose                              | Source                                                                                         | Arguments                               |
| ------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------- |
| `Conv`        | Convolution + BatchNorm + Activation | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py) | `[out_ch, kernel, stride, pad, groups]` |
| `nn.Upsample` | Spatial upsampling                   | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html)               | `[size, scale_factor, mode]`            |
| `nn.Identity` | Pass-through operation               | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html)               | `[]`                                    |

### Composite Blocks

| Module   | Purpose                                                           | Source                                                                                           | Arguments                                          |
| -------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `C3k2`   | CSP block used by every YOLO11 and YOLO26 backbone                | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, c3k, expansion, attn, groups, shortcut]` |
| `C2PSA`  | Position-sensitive attention block, last backbone layer in YOLO26 | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, expansion]`                              |
| `C2f`    | CSP bottleneck with 2 convolutions, used by YOLOv8                | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, shortcut, groups, expansion]`            |
| `SPPF`   | Spatial Pyramid Pooling (fast)                                    | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, kernel_size]`                            |
| `Concat` | Channel-wise concatenation                                        | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)   | `[dimension]`                                      |

### Specialized Modules

| Module        | Purpose                           | Source                                                                                           | Arguments                                                |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `TorchVision` | Load any torchvision model        | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, model_name, weights, unwrap, truncate, split]` |
| `Index`       | Extract specific tensor from list | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)   | `[out_ch, index]`                                        |
| `Detect`      | YOLO detection head               | [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)   | `[nc]`                                                   |
| `Classify`    | Classification head, must be last | [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)   | `[nc]`                                                   |

!!! info "Complete Module List"

    This is a subset. For the full list of modules and their parameters, explore the [modules directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules).

!!! note "Which modules receive an injected input-channel argument"

    For the convolution and block modules — `Conv`, `C3k2`, `C2PSA`, `C2f`, `SPPF` and the rest of `parse_model`'s base-module set — input channels are injected from whatever layer `from` points at, so the args you write start at `c2`. That rule does **not** extend to the other modules in the tables above, each of which `parse_model` handles specially: `nn.Upsample` and `nn.MaxPool2d` take their `torch.nn` arguments unchanged, `Concat` takes a dimension, `Detect` and `Classify` take `nc`, and `Index` takes a leading `out_ch` that exists purely for channel bookkeeping while its constructor reads only the index. A module you register yourself gets no injection at all until you add it, which is what [step 5](#custom-module-integration) is for.

## Advanced Features

### TorchVision Integration

The TorchVision module enables seamless integration of any [TorchVision model](https://docs.pytorch.org/vision/stable/models.html) as a backbone:

=== "Python"

    ```python
    from ultralytics import YOLO

    # Model with ConvNeXt backbone
    model = YOLO("convnext_backbone.yaml")
    results = model.train(data="coco8.yaml", epochs=100)
    ```

=== "YAML Configuration"

    ```yaml
    backbone:
      - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, False]]
    head:
      - [-1, 1, Classify, [nc]]
    ```

    **Parameter Breakdown:**

    - `768`: Expected output channels
    - `convnext_tiny`: Model architecture ([available models](https://docs.pytorch.org/vision/stable/models.html))
    - `DEFAULT`: Use pretrained weights
    - `True`: Unwrap the model into a flat `nn.Sequential` of its child layers. This is what makes the next two arguments possible; setting it to `False` instead replaces the model's own classification head with `nn.Identity` and forces the split argument off.
    - `2`: Truncate the last 2 of those layers
    - `False`: Return a single tensor rather than a list of intermediate feature maps

!!! tip "Multi-Scale Features"

    Set the last parameter to `True` to get intermediate feature maps for multi-scale detection.

### Index Module for Feature Selection

When using models that output multiple feature maps, the Index module selects specific outputs:

```yaml
backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]] # Multi-output
head:
    - [0, 1, Index, [192, 4]] # Select 4th feature map (192 channels)
    - [0, 1, Index, [384, 6]] # Select 6th feature map (384 channels)
    - [0, 1, Index, [768, 8]] # Select 8th feature map (768 channels)
    - [[1, 2, 3], 1, Detect, [nc]] # Multi-scale detection
```

## Module Resolution System

Understanding how Ultralytics locates and imports modules is crucial for customization:

### Module Lookup Process

Ultralytics uses a three-tier system in [`parse_model`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):

```python
# Core resolution logic
m = (
    getattr(torch.nn, m[3:])
    if m.startswith("nn.")
    else getattr(__import__("torchvision").ops, m[16:])
    if m.startswith("torchvision.ops.")
    else globals()[m]
)  # get module
```

1. **PyTorch modules**: names starting with `nn.` → `torch.nn` namespace
2. **TorchVision operations**: names starting with `torchvision.ops.` → `torchvision.ops` namespace
3. **Ultralytics modules**: all other names → the `tasks.py` global namespace, populated by the imports below

A fourth gate applies under restricted loading: a name that resolves to something which is not an `nn.Module` subclass raises `TypeError` instead of being built.

### Module Import Chain

Standard modules become available through imports in [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):

```python

```

## Custom Module Integration

### Source Code Modification

Modifying the source code is the most versatile way to integrate your custom modules, but it can be tricky. To define and use a custom module, follow these steps:

1. **Install Ultralytics in development mode** using the Git clone method from the [Quickstart guide](../quickstart.md#how-do-i-clone-the-ultralytics-repository-for-development).

2. **Define your module** in [`ultralytics/nn/modules/block.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py):

    ```python
    class CustomBlock(nn.Module):
        """Custom block with Conv-BatchNorm-ReLU sequence."""

        def __init__(self, c1, c2):
            """Initialize CustomBlock with input and output channels."""
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU())

        def forward(self, x):
            """Forward pass through the block."""
            return self.layers(x)
    ```

3. **Expose your module at the package level** in [`ultralytics/nn/modules/__init__.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/__init__.py):

    ```python
    from .block import CustomBlock  # noqa makes CustomBlock available as ultralytics.nn.modules.CustomBlock
    ```

4. **Add to imports** in [`ultralytics/nn/tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):

    ```python
    from ultralytics.nn.modules import CustomBlock  # noqa
    ```

5. **Handle channel injection** inside [`parse_model()`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) in `ultralytics/nn/tasks.py`. This step is required, not optional: `parse_model` injects input channels only for modules it knows about, so an unregistered module receives its YAML args verbatim and fails with `TypeError: CustomBlock.__init__() missing 1 required positional argument: 'c2'`.

    ```python
    # Add this condition in the parse_model() function
    if m is CustomBlock:
        c1, c2 = ch[f], args[0]  # input channels, output channels
        args = [c1, c2, *args[1:]]
    ```

6. **Use the module** in your model YAML:

    ```yaml
    # custom_model.yaml
    nc: 1
    backbone:
        - [-1, 1, CustomBlock, [64]]
    head:
        - [-1, 1, Classify, [nc]]
    ```

7. **Check FLOPs** to ensure the forward pass works:

    ```python
    from ultralytics import YOLO

    model = YOLO("custom_model.yaml", task="classify")
    model.info()  # should print non-zero FLOPs if working
    ```

## Example Configurations

### Basic Detection Model

```yaml
# Simple YOLO detection model
nc: 80
scales:
    n: [0.50, 0.25, 1024]

backbone:
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]] # 2
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]] # 4
    - [-1, 1, SPPF, [256, 5]] # 5

head:
    - [-1, 1, Conv, [256, 3, 1]] # 6
    - [[6], 1, Detect, [nc]] # 7
```

!!! warning "A `scales` block needs a scale letter in the filename"

    Ultralytics reads the scale from the file name with the pattern `yolo` + digits + one of `nslmx`. Saved as `simple_detect.yaml` the file above logs `WARNING no model scale passed. Assuming scale='n'.` and silently uses the first entry. Worse, a name that merely looks scalable is not: `mynet26n.yaml` and `mynet26s.yaml` build the **identical** model, because neither stem contains `yolo`. Name a scalable config `myyolo26n.yaml` / `myyolo26s.yaml` and keep the base file unscaled (`myyolo26.yaml`).

### TorchVision Backbone Model

```yaml
# ConvNeXt backbone with YOLO head
nc: 80

backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]]

head:
    - [0, 1, Index, [192, 4]] # P3 features
    - [0, 1, Index, [384, 6]] # P4 features
    - [0, 1, Index, [768, 8]] # P5 features
    - [[1, 2, 3], 1, Detect, [nc]] # Multi-scale detection
```

### Classification Model

```yaml
# Simple classification model
nc: 1000

backbone:
    - [-1, 1, Conv, [64, 7, 2, 3]]
    - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
    - [-1, 4, C3k2, [64, True]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 8, C3k2, [128, True]]

head:
    - [-1, 1, Classify, [nc]]
```

`Classify` pools internally, so do not add an `nn.AdaptiveAvgPool2d` before it. Collapsing the feature map to 1x1 first starves the head's own convolution and makes training with `batch=1` fail with `ValueError: Expected more than 1 value per channel when training`.

## Best Practices

| Practice                         | What it means for your YAML                                                                                                                                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Start from a shipped config**  | Copy a YAML from [`ultralytics/cfg/models`](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) and change one thing at a time rather than writing a backbone from scratch.                                      |
| **Match channels across layers** | For a base module, actual output channels are `make_divisible(min(out_ch, max_channels) * width, 8)`, not the number you wrote. Compute the scaled value before assuming two layers line up, and note the other modules do not follow it. |
| **Reuse features with `Concat`** | `[[-1, N], 1, Concat, [1]]` merges an earlier feature map into the current one, the standard FPN pattern for multi-scale detection. Both sources must share spatial dimensions.                                                           |
| **Pick a scale for your target** | Use `n` for edge devices, `s` for a balanced tradeoff, `m`/`l`/`x` when accuracy matters more than latency.                                                                                                                               |
| **Verify after every change**    | `model.info()` must report non-zero FLOPs; see [Debugging Tips](#debugging-tips).                                                                                                                                                         |

For the reasoning behind depth, width and bottleneck choices, see [YOLO architecture explained](yolo-architecture.md).

## Troubleshooting

### Common Issues

| Problem                                         | Cause                          | Solution                                                                                                  |
| ----------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `KeyError: 'ModuleName'`                        | Module not imported            | Add to [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) imports |
| Channel dimension mismatch                      | Incorrect `args` specification | Verify input/output channel compatibility                                                                 |
| `AttributeError: 'int' object has no attribute` | Wrong argument type            | Check module documentation for correct argument types                                                     |
| Model fails to build                            | Invalid `from` reference       | Ensure referenced layers exist                                                                            |

### Debugging Tips

When developing custom architectures, systematic debugging helps identify issues early:

#### Use an Identity Head to Isolate the Backbone

Replace complex heads with `nn.Identity` to isolate backbone issues:

```yaml
# debug_model.yaml
nc: 1
backbone:
    - [-1, 1, Conv, [64, 3, 2]]
head:
    - [-1, 1, nn.Identity, []] # Pass-through for debugging
```

An identity head gives `parse_model` no head to infer the task from, so pass `task=` explicitly or model loading fails with `NotImplementedError` for task `None`:

```python
import torch

from ultralytics import YOLO

model = YOLO("debug_model.yaml", task="detect")
output = model.model(torch.randn(1, 3, 640, 640))
print(f"Output shape: {output.shape}")  # torch.Size([1, 64, 320, 320])
```

The identity head is what makes `output` a plain tensor. Swap in a real `Detect` head and the forward pass returns a dict of `boxes`, `scores` and `feats` instead, so `output.shape` raises `AttributeError`.

#### Inspect the Built Architecture

Checking the FLOPs count and printing out each layer can also help debug issues with your custom model config. FLOPs count should be non-zero for a valid model. If it's zero, then there's likely an issue with the forward pass. Running a simple forward pass should show the exact error being encountered.

```python
from ultralytics import YOLO

# Build model with verbose output to see layer details
model = YOLO("debug_model.yaml", task="detect", verbose=True)

# Check model FLOPs. Failed forward pass causes 0 FLOPs.
model.info()

# Inspect individual layers
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {layer}")
```

#### Validate Step by Step

1. **Start minimal**: Test with simplest possible architecture first
2. **Add incrementally**: Build complexity layer by layer
3. **Check dimensions**: Verify channel and spatial size compatibility
4. **Validate scaling**: Test with different model scales (`n`, `s`, `m`)

## FAQ

### How do I change the number of classes in my model?

Set the `nc` parameter at the top of your YAML file to match your dataset's number of classes.

```yaml
nc: 5 # 5 classes
```

### Can I use a custom backbone in my model YAML?

Yes. Use any supported module, including [TorchVision backbones](#torchvision-integration), or define your own and register it as described in [Custom Module Integration](#custom-module-integration).

### How do I see the YAML for a pretrained model like yolo26n.pt?

Load the checkpoint and read `model.model.yaml`, which holds the parsed architecture the weights were built from. The source files also ship with the package under [`ultralytics/cfg/models`](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) — copying one from there is the recommended starting point for a custom architecture.

### Why does my layer have fewer channels than the number in my YAML?

For a base module, `max_channels` caps the value before the width multiplier is applied, and the result is rounded up to a multiple of 8. With `m: [0.50, 1.00, 512]`, a layer declaring `[1024, 3, 2]` is clamped to 512 first, so it builds 512 channels rather than 1024. Other modules — `Concat`, `TorchVision`, `Index`, the heads — do not go through that formula. See [How Width Scaling Actually Computes Channels](#how-width-scaling-actually-computes-channels).

### Can I train a model YAML from scratch without pretrained weights?

Yes. Passing a YAML instead of a `.pt` file to `YOLO()` builds randomly initialized weights, so `YOLO("custom_model.yaml").train(data="coco8.yaml")` trains from scratch. Expect to need substantially more data and epochs than fine-tuning a pretrained checkpoint — see the [training mode guide](../modes/train.md).

### How do I scale my model for different sizes (nano, small, medium, etc.)?

Use the [`scales` section](#parameters-section) in your YAML to define scaling factors for depth, width, and max channels. The model will automatically apply these when you load the base YAML file with the scale appended to the filename (e.g., `yolo26n.yaml`).

### How do I troubleshoot channel mismatch errors?

Check that the output channels of one layer match the expected input channels of the next. Use `print(model.model.model)` to inspect your model's architecture.

### Can I use pretrained weights with a custom YAML?

Yes, you can use `model.load("path/to/weights")` to load weights from a pretrained checkpoint. However, only weights for layers that match would load successfully.
