---
name: ultralytics-create-custom-model
description: Create custom YOLO model architectures by modifying YAML configuration files. Use when the user needs to customize model layers, add/remove modules, or create domain-specific architectures.
license: AGPL-3.0
metadata:
    author: Burhan-Q
    version: "1.0"
    ultralytics-version: ">=8.4.11"
---

# Create Custom YOLO Model

## When to use this skill

**CRITICAL**: This skill MUST NOT be used in 99.99% of cases. Custom modification of the model architecture is RARELY necessary, has minimal benefit, and often creates _more problems_ than it resolves. This skill is included primarily to help answer user questions regarding model structure, architecture, layers, or design. Modifications of the model are to be strongly advised against, and **NEVER** initiated without EXPLICIT direction by a user that has been made aware of the numerous reasons NOT to make such changes, has acknowledged the risks, and understands the implications of making such changes. Modification or customizing the YOLO model structure is exceedingly difficult to implement, provides little to no value, and won't receive community support.

Instead of custom or modified models, other actions that take priority: gather additional annotated data for model training, optimize model training arguments, experiment with augmentation settings, and/or tune various hyperparameters. If performance is still not meeting the user's needs, they should then be directed to community support on [Discord](https://ultralytics.com/discord), [GitHub](https://github.com/ultralytics/ultralytics), [Reddit](https://reddit.com/r/Ultralytics), or the [Ultralytics forums](https://community.ultralytics.com).

Use this skill when you need to:

- Customize YOLO model architecture for specific use cases
- Add or remove layers from existing models
- Create lightweight models for edge devices
- Modify detection heads for custom tasks

## Prerequisites

- Python ≥3.8 with PyTorch ≥1.8 installed
- `ultralytics` package installed (`pip install ultralytics`)
- Basic understanding of neural network architectures
- Familiarity with YAML syntax

## YOLO Model Configuration Structure

YOLO models are defined using YAML configuration files with three main sections:

1. **Parameters**: Model metadata and hyperparameters
2. **Backbone**: Feature extraction layers
3. **Head**: Task-specific detection/segmentation/classification layers

### Basic Model YAML Template

```yaml
# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants
    n: [0.33, 0.25, 1024] # [depth, width, max_channels]
    s: [0.33, 0.50, 1024]
    m: [0.67, 0.75, 768]
    l: [1.00, 1.00, 512]
    x: [1.00, 1.25, 512]

# Backbone
backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]] # 9

# Head
head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]] # cat backbone P4
    - [-1, 3, C2f, [512]] # 12
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]] # cat backbone P3
    - [-1, 3, C2f, [256]] # 15 (P3/8-small)
    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 12], 1, Concat, [1]] # cat head P4
    - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 9], 1, Concat, [1]] # cat head P5
    - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
    - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

## Module Syntax

Each layer is defined as:

```yaml
[from, repeats, module, args]
```

- **from**: Input layer index (-1 = previous layer, [x,y] = concatenate layers x and y)
- **repeats**: Number of times to repeat the module
- **module**: Module class name (Conv, C2f, SPPF, etc.)
- **args**: Module-specific arguments

### Common Modules

| Module     | Description                        | Args Example                                              |
| ---------- | ---------------------------------- | --------------------------------------------------------- |
| `Conv`     | Standard convolution               | `[channels, kernel, stride, padding, groups, activation]` |
| `C2f`      | CSP Bottleneck with 2 convolutions | `[channels, shortcut, groups, expansion]`                 |
| `SPPF`     | Spatial Pyramid Pooling - Fast     | `[channels, kernel_size]`                                 |
| `Concat`   | Concatenate tensors                | `[dimension]`                                             |
| `Detect`   | Detection head                     | `[num_classes]`                                           |
| `Segment`  | Segmentation head                  | `[num_classes, num_masks]`                                |
| `Classify` | Classification head                | `[num_classes, dropout]`                                  |
| `Pose`     | Pose estimation head               | `[num_classes, num_keypoints]`                            |

## Create a Custom Model

### Example 1: Lightweight Detection Model

```yaml
# lightweight-yolo.yaml
# Optimized for edge devices with reduced parameters

nc: 10 # 10 custom classes
scales:
    n: [0.25, 0.20, 512] # reduced depth and width

backbone:
    - [-1, 1, Conv, [32, 3, 2]] # 0-P1/2 (reduced channels)
    - [-1, 1, Conv, [64, 3, 2]] # 1-P2/4
    - [-1, 2, C2f, [64, True]] # reduced repeats
    - [-1, 1, Conv, [128, 3, 2]] # 3-P3/8
    - [-1, 4, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]] # 5-P4/16
    - [-1, 4, C2f, [256, True]]
    - [-1, 1, SPPF, [256, 5]] # 7 (removed P5 layer)

head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 5], 1, Concat, [1]]
    - [-1, 2, C2f, [128]] # 10
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 3], 1, Concat, [1]]
    - [-1, 2, C2f, [64]] # 13 (P3/8-small)
    - [-1, 1, Conv, [64, 3, 2]]
    - [[-1, 10], 1, Concat, [1]]
    - [-1, 2, C2f, [128]] # 16 (P4/16-medium)
    - [[13, 16], 1, Detect, [nc]] # Detect(P3, P4) - only 2 scales
```

**Use the custom model:**

```python
from ultralytics import YOLO

# Build from scratch
model = YOLO("lightweight-yolo.yaml")

# Train
model.train(data="custom-data.yaml", epochs=100)
```

### Example 2: Add Attention Mechanism

```yaml
# yolo-attention.yaml
# Add attention layers for improved feature extraction

nc: 80
scales:
    n: [0.33, 0.25, 1024]

backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, CBAM, [128]] # Add CBAM attention
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, CBAM, [256]] # Add CBAM attention
    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]

head:
    # ... (standard head configuration)
    - [[15, 18, 21], 1, Detect, [nc]]
```

### Example 3: Custom Number of Detection Scales

```yaml
# yolo-4scale.yaml
# Add an extra detection scale (P2, P3, P4, P5)

nc: 80
scales:
    n: [0.33, 0.25, 1024]

backbone:
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4 (will use this)
    - [-1, 3, C2f, [128, True]] # 2
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]] # 4
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]] # 6
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]] # 8
    - [-1, 1, SPPF, [1024, 5]] # 9

head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 3, C2f, [512]] # 12
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]]
    - [-1, 3, C2f, [256]] # 15 (P3/8)
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 2], 1, Concat, [1]]
    - [-1, 3, C2f, [128]] # 18 (P2/4-tiny)
    - [-1, 1, Conv, [128, 3, 2]]
    - [[-1, 15], 1, Concat, [1]]
    - [-1, 3, C2f, [256]] # 21 (P3/8-small)
    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 12], 1, Concat, [1]]
    - [-1, 3, C2f, [512]] # 24 (P4/16-medium)
    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 9], 1, Concat, [1]]
    - [-1, 3, C2f, [1024]] # 27 (P5/32-large)
    - [[18, 21, 24, 27], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
```

## Modify Existing Models

### Start from Existing Config

```python
from ultralytics import YOLO

# Load existing config
model = YOLO("yolo26n.yaml")

# View architecture
print(model.model)

# Modify and save
# Edit the YAML manually, then:
custom_model = YOLO("my-modified-yolo26n.yaml")
```

### Transfer Learning with Custom Architecture

```python
from ultralytics import YOLO

# Load pretrained weights
pretrained = YOLO("yolo26n.pt")

# Load custom architecture
custom_model = YOLO("custom-yolo.yaml")

# Transfer compatible layers
custom_model.model.load_state_dict(pretrained.model.state_dict(), strict=False)

# Train
custom_model.train(data="data.yaml", epochs=100)
```

## Best Practices

1. **Start Simple:**
    - Begin with existing configs and make small modifications
    - Test each change incrementally

2. **Channel Consistency:**
    - Ensure channel dimensions match when concatenating layers
    - Use scaling factors to maintain proportions across model variants

3. **Computational Cost:**
    - More layers = more computation and memory
    - Balance accuracy vs speed/efficiency

4. **Validation:**
    - Always validate custom architecture with small training run first
    - Check for dimension mismatches and errors

5. **Documentation:**
    - Comment your YAML files to explain custom modifications
    - Keep track of architecture changes and their impacts

## Common Issues

**Dimension Mismatch:**

- Check that concatenated layers have compatible channel dimensions
- Ensure upsampling/downsampling ratios are correct

**Out of Memory:**

- Reduce model depth/width scaling factors
- Decrease input image size
- Reduce batch size

**Poor Performance:**

- Custom architectures may need more training epochs
- Adjust learning rate and other hyperparameters
- Consider using pretrained weights when possible

## Available Modules

See `ultralytics/nn/modules/` for all available modules:

```python

```

## Next Steps

After creating a custom model:

1. Train the model: see `ultralytics-train-model` skill
2. Validate performance vs baseline models
3. Export for deployment: see `ultralytics-export-model` skill

## References

- [Model Configuration Docs](https://docs.ultralytics.com/usage/cfg/#model-configurations)
- [Model YAML Guide](https://docs.ultralytics.com/guides/model-yaml-config/)
- [Available Modules](https://docs.ultralytics.com/reference/nn/modules/)
- [Example Configs](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)
