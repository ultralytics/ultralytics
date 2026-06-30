---
title: YOLO Architecture Explained
comments: true
description: Understand how the YOLO architecture evolved from YOLOv3 to YOLO26 across the backbone, neck, and detection head, including anchor-free and NMS-free design.
keywords: YOLO architecture, YOLOv3, YOLOv5, YOLOv8, YOLO11, YOLO26, backbone, neck, detection head, anchor-free, NMS-free, C2f, C3k2, DFL, CSP, SPPF, object detection
---

# YOLO Architecture Explained: From YOLOv3 to YOLO26

Every Ultralytics YOLO model is built from three stages: a **backbone** that extracts features, a **neck** that fuses them across scales, and a **head** that predicts boxes and classes. This guide documents the modules that make up each stage and how they changed from [YOLOv3](../models/yolov3.md) to [YOLO26](../models/yolo26.md), tracing every component to its definition in the configuration files under `ultralytics/cfg/models/` and the module classes in [`ultralytics/nn/modules/`](../reference/nn/modules/block.md).

Each model is defined declaratively in a YAML file as an ordered list of layers, where every layer follows the `[from, repeats, module, args]` format: which layer(s) feed it, how many times the module repeats, the layer class (`Conv`, `C3k2`, `SPPF`, `Detect`, …), and its constructor arguments. The [Model YAML Configuration Guide](model-yaml-config.md) documents this format — including how `repeats` and `args` scale with the variant's depth and width multiples — along with the module-resolution system in full. This guide focuses on the modules themselves and how they changed from version to version.

## The Three Stages

Every Ultralytics YOLO model routes the image through three sequential stages, each with a distinct job:

| Stage        | Job                                                                      | Output                                             |
| ------------ | ------------------------------------------------------------------------ | -------------------------------------------------- |
| **Backbone** | Extract features from the input image at multiple resolutions            | Feature maps at strides 8, 16, and 32 (P3, P4, P5) |
| **Neck**     | Fuse features across scales so small and large objects both have context | Multi-scale fused feature maps                     |
| **Head**     | Predict bounding boxes and class scores from the fused features          | Detections per anchor point                        |

The fundamental unit is the **`Conv`** block (defined in [`conv.py`](../reference/nn/modules/conv.md)): a 2D convolution, [batch normalization](https://www.ultralytics.com/glossary/batch-normalization), and a [SiLU](https://www.ultralytics.com/glossary/activation-function) activation, applied in sequence. Every larger module below is built by composing `Conv` blocks.

## Architecture Diagrams

Each version keeps the same **backbone → neck → head** skeleton and changes specific stages. The tabs below show the per-version structure: the backbone and neck stages follow the configs in `ultralytics/cfg/models/`, while the YOLOv3 and YOLOv5 heads are drawn in their original anchor-based form rather than the anchor-free `u`-variant head their package configs actually ship. Stepping through the tabs shows what each generation added. In short, the progression is: YOLOv3 is an FPN-only, anchor-based detector; YOLOv5 adds the bottom-up PAN path and `SPPF`; YOLOv8 switches to the `C2f` block with an anchor-free, [DFL](#distribution-focal-loss-dfl) head; YOLO11 inserts `C2PSA` attention and the `C3k2` block; and YOLO26 adds an `SPPF` residual and makes the head NMS-free and DFL-free. Node colors follow the documentation diagram convention: green input, blue backbone, slate spatial pooling and attention, orange neck, purple head and output.

=== "YOLOv3"

    ```mermaid
    flowchart TD
        IN[Input 640x640]:::start --> ST[Conv stem<br/>5x stride-2 down to P1-P5]:::proc
        ST --> BB[Darknet-53 backbone<br/>stacked Bottleneck]:::proc
        BB --> FPN[Neck FPN only<br/>top-down Upsample + Concat]:::decide
        FPN --> HD[Detect head<br/>3 scales, anchor-based]:::out
        HD --> O[Predictions + NMS]:::out
        classDef start fill:#4CAF50,color:#fff
        classDef proc fill:#2196F3,color:#fff
        classDef decide fill:#FF9800,color:#fff
        classDef out fill:#9C27B0,color:#fff
    ```

=== "YOLOv5"

    ```mermaid
    flowchart TD
        IN[Input 640x640]:::start --> BB[CSP backbone<br/>C3 blocks]:::proc
        BB --> SP[SPPF]:::extern
        SP --> FPN[Neck FPN top-down<br/>Upsample + Concat]:::decide
        FPN --> PAN[Neck PAN bottom-up<br/>Conv + Concat]:::decide
        PAN --> HD[Detect head<br/>anchor-based]:::out
        HD --> O[Predictions + NMS]:::out
        classDef start fill:#4CAF50,color:#fff
        classDef proc fill:#2196F3,color:#fff
        classDef decide fill:#FF9800,color:#fff
        classDef out fill:#9C27B0,color:#fff
        classDef extern fill:#607D8B,color:#fff
    ```

=== "YOLOv8"

    ```mermaid
    flowchart TD
        IN[Input 640x640]:::start --> BB[CSP backbone<br/>C2f blocks]:::proc
        BB --> SP[SPPF]:::extern
        SP --> FPN[Neck FPN top-down<br/>Upsample + Concat]:::decide
        FPN --> PAN[Neck PAN bottom-up<br/>Conv + Concat]:::decide
        PAN --> HD[Detect head<br/>anchor-free, decoupled, DFL reg_max 16]:::out
        HD --> O[Predictions + NMS]:::out
        classDef start fill:#4CAF50,color:#fff
        classDef proc fill:#2196F3,color:#fff
        classDef decide fill:#FF9800,color:#fff
        classDef out fill:#9C27B0,color:#fff
        classDef extern fill:#607D8B,color:#fff
    ```

=== "YOLO11"

    ```mermaid
    flowchart TD
        IN[Input 640x640]:::start --> BB[CSP backbone<br/>C3k2 blocks]:::proc
        BB --> SP[SPPF]:::extern
        SP --> PSA[C2PSA attention]:::extern
        PSA --> FPN[Neck FPN top-down<br/>Upsample + Concat]:::decide
        FPN --> PAN[Neck PAN bottom-up<br/>Conv + Concat]:::decide
        PAN --> HD[Detect head<br/>anchor-free, DFL reg_max 16]:::out
        HD --> O[Predictions + NMS]:::out
        classDef start fill:#4CAF50,color:#fff
        classDef proc fill:#2196F3,color:#fff
        classDef decide fill:#FF9800,color:#fff
        classDef out fill:#9C27B0,color:#fff
        classDef extern fill:#607D8B,color:#fff
    ```

=== "YOLO26"

    ```mermaid
    flowchart TD
        IN[Input 640x640]:::start --> BB[CSP backbone<br/>C3k2 blocks]:::proc
        BB --> SP[SPPF + shortcut]:::extern
        SP --> PSA[C2PSA attention]:::extern
        PSA --> FPN[Neck FPN top-down<br/>Upsample + Concat]:::decide
        FPN --> PAN[Neck PAN bottom-up<br/>Conv + Concat]:::decide
        PAN --> HD[Detect head<br/>anchor-free, reg_max 1, end2end]:::out
        HD --> O[End-to-end predictions<br/>NMS-free]:::out
        classDef start fill:#4CAF50,color:#fff
        classDef proc fill:#2196F3,color:#fff
        classDef decide fill:#FF9800,color:#fff
        classDef out fill:#9C27B0,color:#fff
        classDef extern fill:#607D8B,color:#fff
    ```

The YOLOv3 and YOLOv5 diagrams show the original anchor-based head. The `ultralytics` package ships the anchor-free **YOLOv3u** and **YOLOv5u** configs — the same Darknet-53 and `C3` backbones with YOLOv8's `Detect` head — described under [Detection Head](#detection-head-anchor-based-anchor-free-nms-free).

## Backbone Blocks: Bottleneck → C3 → C2f → C3k2

The backbone stacks a repeating CSP (Cross-Stage Partial) block between stride-2 `Conv` downsampling layers. That repeating block is what changed most across versions. All blocks below live in [`block.py`](../reference/nn/modules/block.md); `c1`/`c2` are input/output channels and `c = 0.5 * c2` is the hidden width.

### Bottleneck (YOLOv3)

The base unit is `Bottleneck`: two `Conv` layers (default kernels `(3, 3)`) with an optional residual add when `shortcut=True` and `c1 == c2`. [YOLOv3](../models/yolov3.md)'s **Darknet-53** backbone stacks these directly, with no CSP split, and detects at three scales (strides 8, 16, 32).

### C3 (YOLOv5)

[YOLOv5](../models/yolov5.md)'s `C3` splits the input across two `1x1` convolutions: `cv1` feeds `n` sequential `Bottleneck` blocks (kernels `(1, 1)` then `(3, 3)`), `cv2` bypasses them. The two paths are concatenated and fused by a third `1x1` `Conv`:

```python
def forward(self, x):
    # C3: bottleneck path m(cv1(x)) concatenated with bypass cv2(x), then fused by cv3
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

Only the **final** bottleneck output reaches the fusion conv, so `cv3` sees 2 feature maps.

### C2f (YOLOv8)

[YOLOv8](../models/yolov8.md)'s `C2f` ("CSP Bottleneck with 2 convolutions, faster") changes which features reach the fusion conv:

1. `cv1 = Conv(c1, 2 * c, 1)`, then `chunk(2)` splits the output into two `c`-channel tensors.
2. `n` `Bottleneck(c, c)` blocks (kernels `(3, 3)`, `(3, 3)`) run sequentially, each fed the previous block's output.
3. All `n + 2` intermediate tensors are concatenated and fused by `cv2 = Conv((2 + n) * c, c2, 1)`.

Where `C3` passes 2 feature maps into its fusion conv, `C2f` passes `n + 2` — every intermediate bottleneck output is reused.

### C3k2 (YOLO11 and YOLO26)

[YOLO11](../models/yolo11.md) and [YOLO26](../models/yolo26.md) use `C3k2`, a subclass of `C2f` that swaps the repeating unit. Each of the `n` blocks becomes, depending on the constructor flags:

- a plain `Bottleneck` (default, `c3k=False`),
- a `C3k` block (`c3k=True`) — a `C3` variant with a configurable kernel size, or
- a `Bottleneck` + `PSABlock` pair (`attn=True`).

The second YAML arg sets `c3k`; for example `[-1, 2, C3k2, [512, True]]` builds one `C3k2` module at 512 output channels whose internal blocks are `C3k` (since `c3k=True`). For CSP modules, the `repeats` field — here 2, before it is scaled by the variant's depth multiple — becomes the block's internal repeat count rather than stacking separate modules.

## Spatial Pooling: SPP → SPPF

At the end of the backbone, a spatial-pyramid-pooling block widens the receptive field. YOLOv5 replaced the original multi-kernel `SPP` with **`SPPF`** (Spatial Pyramid Pooling - Fast): a single `MaxPool2d(kernel_size=5, stride=1, padding=2)` applied `n = 3` times in sequence, with the input and all three pooled outputs concatenated and fused by a `1x1` `Conv`. This is mathematically equivalent to `SPP(k=(5, 9, 13))` but cheaper, because the chained `5x5` pools cover the larger kernels' receptive fields.

[YOLO26](../models/yolo26.md) passes a shortcut flag (`SPPF, [1024, 5, 3, True]`); since `c1 == c2 == 1024` at the deepest layer, `SPPF` adds a residual connection (`return y + x`).

## Spatial Attention: C2PSA (YOLO11+)

[YOLO11](../models/yolo11.md) added **`C2PSA`** after `SPPF`. It is a CSP block whose active branch is a stack of `n` `PSABlock` (Position-Sensitive Attention) modules: `cv1 = Conv(c1, 2 * c, 1)` splits the features, one half passes through the `PSABlock` stack, and `cv2 = Conv(2 * c, c1, 1)` fuses the concatenation. Each `PSABlock` applies multi-head [attention](https://www.ultralytics.com/glossary/attention-mechanism) followed by a two-layer feed-forward network (`Conv(c, 2 * c, 1)` → `Conv(2 * c, c, 1)`), each with a residual connection. [YOLO26](../models/yolo26.md) keeps the same `C3k2` + `C2PSA` backbone.

## Neck: FPN + PAN

The neck fuses the backbone's P3/P4/P5 feature maps with a top-down Feature Pyramid Network (FPN) followed by a bottom-up Path Aggregation Network (PAN). In the YAML head section, FPN is `nn.Upsample` + `Concat` (carrying semantic information down to higher resolutions) and PAN is stride-2 `Conv` + `Concat` (carrying localization information back up):

```yaml
# YOLO11 head (FPN top-down, then PAN bottom-up)
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]
- [[-1, 6], 1, Concat, [1]] # cat backbone P4
- [-1, 2, C3k2, [512, False]] # 13
# ... second upsample + concat to P3 ...
- [-1, 1, Conv, [256, 3, 2]]
- [[-1, 13], 1, Concat, [1]] # cat head P4 (PAN)
- [-1, 2, C3k2, [512, False]] # 19
```

The neck reuses the backbone block of its generation — `C3` in YOLOv5, `C2f` in YOLOv8, `C3k2` in YOLO11 and YOLO26 — so each merge point runs the same module the backbone uses. The three fused outputs feed the head. YOLOv3 is the exception: its neck is top-down FPN only (its YAML head has no stride-2 downsampling), without the bottom-up PAN path that YOLOv5 introduced.

## Detection Head: Anchor-Based → Anchor-Free → NMS-Free

The head turns the three fused feature maps into predictions for the [detection task](../tasks/detect.md). Its design has changed across versions, from anchor-based to anchor-free to NMS-free.

### Anchor-free, decoupled `Detect`

The **original** YOLOv3 and YOLOv5 used an [anchor-based](https://www.ultralytics.com/glossary/anchor-based-detectors), coupled head: predefined anchor boxes and a shared branch for box and class predictions. The standalone [ultralytics/yolov3](https://github.com/ultralytics/yolov3) and [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repositories keep that anchor-based design. The main `ultralytics` package instead ships the anchor-free **YOLOv3u** and **YOLOv5u** variants — the same Darknet-53 and `C3` backbones with YOLOv8's anchor-free `Detect` head — and the `yolov3.yaml` and `yolov5.yaml` configs documented here are these `u` variants, not the historical design.

The `Detect` head ([`head.py`](../reference/nn/modules/head.md)) is anchor-free and decoupled: per pyramid level it runs two parallel branches, and predicts directly on grid points rather than against anchor boxes.

- **Box branch (`cv2`):** `Conv(x, c2, 3)` → `Conv(c2, c2, 3)` → `Conv2d(c2, 4 * reg_max, 1)`.
- **Class branch (`cv3`):** in YOLO11 and YOLO26, two depthwise-separable blocks (`DWConv` + `1x1 Conv`) → `Conv2d(c3, nc, 1)`; YOLOv8 uses the legacy variant, two `3x3 Conv` layers → `Conv2d(c3, nc, 1)`.

Each anchor point therefore emits `no = nc + 4 * reg_max` outputs. Removing predefined anchors drops the anchor box sizes and aspect ratios from the hyperparameters that must be tuned.

### Distribution Focal Loss (DFL)

YOLOv8 and YOLO11 regress each of the 4 box coordinates as a **distribution** over `reg_max = 16` bins rather than a single scalar (the integral form from [Generalized Focal Loss](https://arxiv.org/abs/2006.04388)). The `DFL` module reshapes the `4 * reg_max` box channels to `(4, reg_max)`, applies a softmax over the `reg_max` bins, and takes the expected bin index — each bin index weighted by its softmax probability, then summed — as the predicted coordinate. This is implemented as a fixed `1x1` convolution whose weights are the bin indices `arange(reg_max)`, so the weighted sum is a single dot product.

### YOLO26: NMS-free, DFL-free

[YOLO26](../models/yolo26.md) sets two YAML parameters that the head reads directly:

- **`end2end: True`** — `Detect` deep-copies its branches into a one-to-one head (`one2one_cv2`/`one2one_cv3`) that produces a single prediction per object, removing the [Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) (NMS) post-processing step. See the [End-to-End Detection guide](end2end-detection.md) for export and migration details.
- **`reg_max: 1`** — with one bin, `self.dfl` becomes `nn.Identity()` and `no = nc + 4`; the head regresses coordinates directly and no DFL operation appears in the exported [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange) graph.

Across its five model sizes (n/s/m/l/x), YOLO26 reaches 40.9-57.5 mAP on COCO at 1.7-11.8 ms T4 [TensorRT](https://www.ultralytics.com/glossary/tensorrt) latency, as reported in the [YOLO26 paper](https://arxiv.org/abs/2606.03748).

## Version-by-Version Summary

| Version    | Backbone block            | Spatial pooling     | Attention | Detection head                                   | DFL                       |
| ---------- | ------------------------- | ------------------- | --------- | ------------------------------------------------ | ------------------------- |
| **YOLOv3** | Darknet-53 (`Bottleneck`) | none in base config | none      | Original: anchor-based; `u` variant: anchor-free | no / yes (`u`)            |
| **YOLOv5** | `C3` (CSP)                | `SPPF`              | none      | Original: anchor-based; `u` variant: anchor-free | no / yes (`u`)            |
| **YOLOv8** | `C2f`                     | `SPPF`              | none      | Anchor-free, decoupled                           | yes (`reg_max=16`)        |
| **YOLO11** | `C3k2`                    | `SPPF`              | `C2PSA`   | Anchor-free, decoupled                           | yes (`reg_max=16`)        |
| **YOLO26** | `C3k2`                    | `SPPF` + shortcut   | `C2PSA`   | Anchor-free, **NMS-free** (`end2end`)            | **removed** (`reg_max=1`) |

For per-model details, performance tables, and usage examples, see the individual pages for [YOLOv3](../models/yolov3.md), [YOLOv5](../models/yolov5.md), [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md).

## Inspect the Architecture Yourself

The `model.info()` method prints a layer, parameter, and [FLOPs](https://www.ultralytics.com/glossary/flops) summary, and the parsed module list is available on `model.model.model`.

!!! example "Inspect a YOLO model's architecture"

    ```python
    from ultralytics import YOLO

    # Load a pretrained model
    model = YOLO("yolo11n.pt")

    # Fuse Conv + BatchNorm layers so counts match the published specs
    model.fuse()

    # Print a summary: layers, parameters, gradients, GFLOPs
    model.info()

    # Inspect the detection head (the last module in the network)
    head = model.model.model[-1]
    print(type(head).__name__, "| reg_max:", head.reg_max, "| end2end:", head.end2end)
    ```

Running the snippet across three generations shows the changes numerically. These are real fused-model outputs from the `ultralytics` package, matching the parameter and FLOPs counts published on each [model page](../models/index.md):

| Model   | Layers | Parameters | GFLOPs | `reg_max` | `end2end` | DFL layer  |
| ------- | ------ | ---------- | ------ | --------- | --------- | ---------- |
| YOLOv8n | 72     | 3,151,904  | 8.7    | 16        | `False`   | `DFL`      |
| YOLO11n | 100    | 2,616,248  | 6.5    | 16        | `False`   | `DFL`      |
| YOLO26n | 122    | 2,408,932  | 5.4    | 1         | `True`    | `Identity` |

YOLO26n reports `reg_max=1`, `end2end=True`, and an `Identity` DFL layer — the architectural signature of its NMS-free, DFL-free head.

!!! note "Fused vs unfused counts"

    Parameter and FLOPs values are reported for the **fused** model (`model.fuse()`), which merges each `Conv` and its [batch normalization](https://www.ultralytics.com/glossary/batch-normalization) layer. This matches the published specifications; a freshly loaded checkpoint reports slightly higher counts before fusing.

## Conclusion

Across versions, the YOLO architecture changed one stage at a time: the backbone moved from Darknet-53 to CSP-based `C3`, `C2f`, and `C3k2` blocks with `C2PSA` attention; the neck kept its FPN + PAN structure while `SPP` became `SPPF`; and the head moved from anchor-based to anchor-free, then to YOLO26's NMS-free, DFL-free end-to-end design.

To define custom architectures, see the [Model YAML Configuration Guide](model-yaml-config.md), or compare models on the [model pages](../models/index.md). For questions, reach out on [GitHub](https://github.com/ultralytics/ultralytics/issues/new/choose) or [Discord](https://discord.com/invite/ultralytics).

## FAQ

### What are the three stages of a YOLO architecture?

A YOLO model has a **backbone** that extracts features from the image at strides 8, 16, and 32, a **neck** that fuses those features across scales with FPN and PAN, and a **head** that predicts bounding boxes and class scores. Every Ultralytics YOLO model from YOLOv3 to YOLO26 follows this three-stage design.

### What is the difference between the C2f and C3k2 blocks?

`C2f` (YOLOv8) is a CSP block that concatenates the outputs of every internal `Bottleneck` — `n + 2` feature maps — before its fusion convolution, where the older `C3` passes only 2. `C3k2` (YOLO11 and YOLO26) is a subclass of `C2f` that can replace each `Bottleneck` with a `C3k` block (a `C3` variant with a configurable kernel size) when its `c3k` flag is set. Both are defined in [`block.py`](../reference/nn/modules/block.md).

### What changed in the architecture between YOLOv8 and YOLO11?

YOLO11 makes three structural changes to YOLOv8: it replaces the `C2f` backbone and neck block with `C3k2`, inserts a `C2PSA` self-attention block after `SPPF`, and switches the head's classification branch to lighter depthwise-separable convolutions. Both keep the same anchor-free, decoupled `Detect` head with `reg_max=16` DFL regression, so the changes lower parameter and FLOPs counts while raising accuracy rather than redesigning the detection interface.

### Is YOLO anchor-free?

Modern Ultralytics YOLO models are anchor-free. YOLOv8, YOLO11, and YOLO26 use an anchor-free, decoupled `Detect` head with separate branches for box regression and classification. The original YOLOv3 and YOLOv5 were anchor-based, but Ultralytics ships them as the **YOLOv3u** and **YOLOv5u** variants, whose configs use the same anchor-free head as YOLOv8.

### Did YOLO26 remove NMS?

Yes — [YOLO26](../models/yolo26.md) sets `end2end=True`, which gives `Detect` a one-to-one head that produces a single prediction per object and removes the Non-Maximum Suppression post-processing step required by earlier models. See the [End-to-End Detection guide](end2end-detection.md) for details.

### What is Distribution Focal Loss (DFL) and why did YOLO26 remove it?

DFL regresses each box coordinate as a softmax distribution over `reg_max` bins (16 by default in YOLOv8 and YOLO11) and takes the expected value as the coordinate, rather than predicting a single scalar. YOLO26 sets `reg_max=1`, so the DFL layer becomes an identity operation, the head regresses coordinates directly, and no DFL op appears in exported ONNX or TensorRT graphs.

### How can I see the architecture of a specific YOLO model?

Load the model in Python and call `model.info()` for a layer, parameter, and GFLOPs summary. The parsed layers are in `model.model.model` — for example, `model.model.model[-1]` is the `Detect` head, exposing attributes like `reg_max` and `end2end`. The full architecture is defined in the model's [YAML configuration file](model-yaml-config.md).
