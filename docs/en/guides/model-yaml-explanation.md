---
title: Model YAML Configuration Guide
description: Learn how to structure and customize model architectures using Ultralytics YAML configuration files. Master module definitions, connections, and scaling parameters.
keywords: Ultralytics, YOLO, model architecture, YAML configuration, neural networks, deep learning, backbone, head, modules, custom models
---

# Model YAML Configuration

The model YAML configuration file serves as the architectural blueprint for Ultralytics neural networks. It defines how layers connect, what parameters each module uses, and how the entire network scales across different model sizes. Think of it as a programming language specifically designed for describing deep learning architectures with maximum flexibility and clarity.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsOGtPlZSTs" title="Ultralytics YOLO Model Training" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Model Configuration and Training
</p>

## Configuration Structure

Model YAML files contain three main sections that work together to define your architecture:

### Parameters Section

The parameters section establishes global model characteristics and scaling behaviors:

```yaml
# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # nano: shallow layers, narrow channels
  s: [0.50, 0.50, 1024]  # small: standard width with shallow depth
  m: [0.50, 1.00, 512]   # medium: full width, moderate depth
  l: [1.00, 1.00, 512]   # large: full depth and width
  x: [1.00, 1.50, 512]   # extra-large: maximum performance
```

!!! tip "Model Scaling"
    Scales work like architectural templates. When you specify `yolo11n.yaml`, the framework automatically applies the 'n' scaling factors to create a lightweight model optimized for speed.

### Backbone and Head Architecture

The model architecture consists of backbone (feature extraction) and head (task-specific) sections:

```yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]     # 0: Initial convolution
  - [-1, 1, Conv, [128, 3, 2]]    # 1: Downsample
  - [-1, 3, C2f, [128, True]]     # 2: Feature processing
  
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 6: Upsample
  - [[-1, 4], 1, Concat, [1]]                   # 7: Skip connection
  - [-1, 3, C2f, [256]]                         # 8: Process features
  - [[8], 1, Detect, [nc]]                      # 9: Detection layer
```

## Layer Specification Format

Every layer follows the consistent pattern: **`[from, repeats, module, args]`**

| Component | Purpose | Examples |
|-----------|---------|----------|
| **from** | Input connections | `-1` (previous), `6` (layer 6), `[4, 6, 8]` (multi-input) |
| **repeats** | Number of repetitions | `1` (single), `3` (repeat 3 times) |
| **module** | Module type | `Conv`, `C2f`, `TorchVision`, `Detect` |
| **args** | Module arguments | `[64, 3, 2]` (channels, kernel, stride) |

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
- [-1, 3, C2f, [128, True]]  # Creates 3 consecutive C2f blocks
- [-1, 1, Conv, [64, 3, 2]]  # Single convolution layer
```

The actual repetition count gets multiplied by the depth scaling factor from your model size configuration.

## Available Modules

Modules are organized by functionality and defined in the [Ultralytics modules directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules):

### Basic Operations

| Module | Purpose | Source | Arguments |
|--------|---------|--------|-----------|
| `Conv` | Convolution + BatchNorm + Activation | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py) | `[out_ch, kernel, stride, pad, groups]` |
| `nn.Upsample` | Spatial upsampling | [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html) | `[size, scale_factor, mode]` |
| `nn.Identity` | Pass-through operation | [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html) | `[]` |

### Composite Blocks

| Module | Purpose | Source | Arguments |
|--------|---------|--------|-----------|
| `C2f` | CSP bottleneck with 2 convolutions | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, shortcut, expansion]` |
| `SPPF` | Spatial Pyramid Pooling (fast) | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, kernel_size]` |
| `Concat` | Channel-wise concatenation | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py) | `[dimension]` |

### Specialized Modules

| Module | Purpose | Source | Arguments |
|--------|---------|--------|-----------|
| `TorchVision` | Load any torchvision model | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, model_name, weights, unwrap, truncate, split]` |
| `Index` | Extract specific tensor from list | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, index]` |
| `Detect` | YOLO detection head | [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py) | `[nc, anchors, ch]` |

!!! info "Module Documentation"
    For complete module documentation, visit the [API Reference](https://docs.ultralytics.com/reference/nn/modules/).

## Advanced Features

### TorchVision Integration

The TorchVision module enables seamless integration of any [torchvision model](https://pytorch.org/vision/stable/models.html) as a backbone:

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
      - [-1, 1, TorchVision, [768, "convnext_tiny", "DEFAULT", True, 2, False]]
    head:
      - [-1, 1, Classify, [nc]]
    ```

**Parameter Breakdown:**
- `768`: Expected output channels
- `"convnext_tiny"`: Model architecture ([available models](https://pytorch.org/vision/stable/models.html))
- `"DEFAULT"`: Use pretrained weights
- `True`: Remove classification head
- `2`: Truncate last 2 layers
- `False`: Return single tensor (not list)

!!! tip "Multi-Scale Features"
    Set the last parameter to `True` to get intermediate feature maps for multi-scale detection.

### Index Module for Feature Selection

When using models that output multiple feature maps, the Index module selects specific outputs:

```yaml
backbone:
  - [-1, 1, TorchVision, [768, "convnext_tiny", "DEFAULT", True, 2, True]]  # Multi-output
head:
  - [0, 1, Index, [192, 4]]   # Select 4th feature map (192 channels)
  - [0, 1, Index, [384, 6]]   # Select 6th feature map (384 channels)  
  - [0, 1, Index, [768, 8]]   # Select 8th feature map (768 channels)
  - [[1, 2, 3], 1, Detect, [nc]]  # Multi-scale detection
```

## Module Resolution System

Understanding how Ultralytics locates and imports modules is crucial for customization:

### Module Lookup Process

The framework uses a two-tier system in [`parse_model`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):

```python
# Core resolution logic
m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
```

1. **PyTorch modules**: Names starting with `'nn.'` → `torch.nn` namespace
2. **Ultralytics modules**: All other names → global namespace via imports

### Module Import Chain

Standard modules become available through imports in [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):

```python
from ultralytics.nn.modules import (
    Conv, C2f, SPPF, TorchVision, Index, Detect,
    # ... many more modules
)
```

## Custom Module Integration

### Method 1: Source Code Modification

1. **Define your module** in [`block.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py):
   ```python
   class CustomBlock(nn.Module):
       def __init__(self, c1, c2, kernel=3):
           super().__init__()
           self.conv = Conv(c1, c2, kernel)
   ```

2. **Add to imports** in [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py):
   ```python
   from ultralytics.nn.modules import CustomBlock
   ```

3. **Use in YAML**:
   ```yaml
   backbone:
     - [-1, 1, CustomBlock, [128, 5]]
   ```

### Method 2: Dynamic Injection

=== "Python"
    ```python
    import torch.nn as nn
    from ultralytics import YOLO
    
    # Define custom module
    class MyBlock(nn.Module):
        def __init__(self, c1, c2):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(c1, c2, 3, 1, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU()
            )
    
        def forward(self, x):
            return self.layers(x)
    
    # Inject into Ultralytics namespace
    import ultralytics.nn.tasks
    ultralytics.nn.tasks.MyBlock = MyBlock
    
    # Use in model
    model = YOLO("custom_model.yaml")
    ```

=== "YAML"
    ```yaml
    backbone:
      - [-1, 1, MyBlock, [64]]  # Now available
    head:
      - [-1, 1, Classify, [nc]]
    ```

### Method 3: Future YAML-Embedded Modules

The upcoming [PR #19615](https://github.com/ultralytics/ultralytics/pull/19615) enables inline module definitions:

```yaml
# Define modules directly in YAML
init: |
  class CustomBackbone(nn.Module):
      def __init__(self, c1, c2, depth=3):
          super().__init__()
          self.layers = nn.Sequential(*[
              Conv(c1 if i == 0 else c2, c2, 3, 1) for i in range(depth)
          ])
      
      def forward(self, x):
          return self.layers(x)
  
  globals()['CustomBackbone'] = CustomBackbone

parse: |
  # Custom argument processing
  if m is CustomBackbone:
      args = [c1, c2, *args]

backbone:
  - [-1, 1, CustomBackbone, [64, 32, 4]]
```

!!! warning "Development Feature"
    The YAML-embedded modules feature is not yet merged. Track [PR #19615](https://github.com/ultralytics/ultralytics/pull/19615) for updates.

## Example Configurations

### Basic Detection Model

```yaml
# Simple YOLO detection model
nc: 80
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 6, 2, 2]]    # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4  
  - [-1, 3, C2f, [128, True]]       # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  - [-1, 6, C2f, [256, True]]       # 4
  - [-1, 1, SPPF, [256, 5]]         # 5

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]       # cat backbone P4
  - [-1, 3, C2f, [256]]             # 8 (P3/8-small)
  - [[8], 1, Detect, [nc]]          # Detect(P3)
```

### TorchVision Backbone Model

```yaml
# ConvNeXt backbone with YOLO head
nc: 80

backbone:
  - [-1, 1, TorchVision, [768, "convnext_tiny", "DEFAULT", True, 2, True]]

head:
  - [0, 1, Index, [192, 4]]         # P3 features
  - [0, 1, Index, [384, 6]]         # P4 features  
  - [0, 1, Index, [768, 8]]         # P5 features
  - [[1, 2, 3], 1, Detect, [nc]]   # Multi-scale detection
```

### Classification Model

```yaml
# Simple classification model
nc: 1000

backbone:
  - [-1, 1, Conv, [64, 7, 2, 3]]
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
  - [-1, 4, C2f, [64, True]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 8, C2f, [128, True]]
  - [-1, 1, nn.AdaptiveAvgPool2d, [1]]

head:
  - [-1, 1, Classify, [nc]]
```

## Best Practices

!!! tip "Architecture Design Tips"

    **Start Simple**: Begin with proven architectures before customizing
    
    **Test Incrementally**: Validate each modification step-by-step
    
    **Monitor Channels**: Ensure channel dimensions match between connected layers
    
    **Use Skip Connections**: Leverage feature reuse with `[[-1, N], 1, Concat, [1]]` patterns
    
    **Scale Appropriately**: Choose model scales based on your computational constraints

!!! note "Performance Considerations"

    - **Depth vs Width**: Deep networks capture complex features, wide networks process more information
    - **Skip Connections**: Improve gradient flow and enable feature reuse
    - **Bottleneck Blocks**: Reduce computational cost while maintaining expressiveness
    - **Multi-Scale Features**: Essential for detecting objects at different sizes

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `KeyError: 'ModuleName'` | Module not imported | Add to [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) imports |
| Channel dimension mismatch | Incorrect `args` specification | Verify input/output channel compatibility |
| `AttributeError: 'int' object has no attribute` | Wrong argument type | Check module documentation for correct argument types |
| Model fails to build | Invalid `from` reference | Ensure referenced layers exist |

### Debugging Tips

=== "Python"
    ```python
    from ultralytics import YOLO
    
    # Build model with verbose output
    model = YOLO("debug_model.yaml", verbose=True)
    
    # Inspect model architecture
    model.info()
    ```

=== "CLI"
    ```bash
    # Validate configuration
    yolo cfg=debug_model.yaml task=detect mode=train epochs=1 verbose=True
    ```

## Related Documentation

- **[Configuration Guide](https://docs.ultralytics.com/usage/cfg/)**: Complete configuration options
- **[Training Guide](https://docs.ultralytics.com/modes/train/)**: Model training workflows  
- **[Model Hub](https://docs.ultralytics.com/models/)**: Pre-trained model architectures
- **[API Reference](https://docs.ultralytics.com/reference/)**: Complete API documentation

## Community & Support

- **[GitHub Issues](https://github.com/ultralytics/ultralytics/issues)**: Bug reports and feature requests
- **[GitHub Discussions](https://github.com/orgs/ultralytics/discussions)**: Community discussions and Q&A
- **[Discord](https://discord.gg/ultralytics)**: Real-time community support
- **[Contributing Guide](https://docs.ultralytics.com/help/contributing/)**: Contribute new modules and features

---

For the latest updates and detailed examples, visit the [official Ultralytics documentation](https://docs.ultralytics.com/).
