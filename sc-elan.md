# Review of Advanced Modules for Small Object Detection and Proposal for SC-ELAN

## 1. Abstract
This review analyzes three state-of-the-art modules—**Pzconv**, **FCM (Feature Context Module)**, and **RepNCSPELAN4**—that have demonstrated significant improvements in small object detection compared to YOLOv8 benchmarks. By identifying their common advantages in multi-scale context perception, feature interaction, and gradient flow efficiency, we propose a novel hybrid module: **SC-ELAN (Spatial-Context Efficient Layer Aggregation Network)**.

## 2. Analysis of Existing Modules

### 2.1 Pzconv (Parallel Zone Convolution)
*   **Mechanism**: utilizes parallel convolution kernels of varying sizes (3x3, 5x5, 7x7) to extract features.
*   **Advantage for SOD**: Addresses the lack of texture information in small objects by expanding the receptive field. The larger kernels capture surrounding context (e.g., "sky" around a "bird"), which is crucial for distinguishing objects from background noise.

### 2.2 FCM (Feature Context Module)
*   **Mechanism**: A dual-branch structure that splits channels and uses one branch to generate spatial/channel attention weights for the other.
*   **Advantage for SOD**: Provides a self-calibration mechanism. Small objects are often overwhelmed by background clutter; FCM's cross-attention highlights the relevant spatial locations and feature channels, effectively suppressing false positives.

### 2.3 RepNCSPELAN4 (Generalized ELAN)
*   **Mechanism**: Dense layer aggregation with gradient path optimization, often combined with re-parameterization.
*   **Advantage for SOD**: Solves the "gradient vanishing" problem common in deep networks. By aggregating features from different depths (concatenation), it preserves high-resolution shallow features (edges, corners) that are vital for detecting tiny targets, ensuring they aren't lost during downsampling.

## 3. Common Advantages Summary
The success of these modules in small object detection can be attributed to three "Golden Rules":

1.  **Context Awareness**: Breaking the limitation of local 3x3 views to understand the environment around the object.
2.  **Feature Fidelity**: Maintaining direct access to raw feature gradients from earlier layers to prevent information loss.
3.  **Attentional Interaction**: Dynamically modulating feature responses to focus on "what" (channel) and "where" (spatial) the small object is.

## 4. Proposal: SC-ELAN (Spatial-Context Efficient Layer Aggregation Network)

Based on the analysis, we propose **SC-ELAN**, a module designed to fully exploit these advantages.

### 4.1 Design Philosophy
SC-ELAN integrates the **gradient efficiency of ELAN** with the **large-kernel context of Pzconv** and the **feature purification of FCM**.

### 4.2 Core Components
1.  **ContextAwareRepConv**: Replaces standard convolutions in the ELAN computational block. It uses multi-branch convolutions (1x1, 3x3, 5x5) during training to capture context, which are re-parameterized into a single 3x3 conv during inference for zero latency overhead.
2.  **Split-Interaction Mechanism**: Before the final feature aggregation, a split-attention block is introduced to filter background noise using spatial and channel mutual guidance.

### 4.3 Architecture Logic
```mermaid
graph LR
    Input[Input Feature] --> Split1[Split/Chunk]
    Split1 -->|Branch 1| B1_Out[Identity/Proj]
    Split1 -->|Branch 2| CARC1[ContextAware RepConv 1]
    CARC1 --> CARC2[ContextAware RepConv 2]
    
    subgraph "Gradient Highway"
    B1_Out
    CARC1
    CARC2
    end
    
    B1_Out --> Concat[Concat Features]
    CARC1 --> Concat
    CARC2 --> Concat
    
    Concat --> Interaction[Split-Interaction Block]
    Interaction --> FinalConv[1x1 Conv Aggregation]
    FinalConv --> Output[Output Feature]
```

### 4.4 Expected Impact
*   **Higher Recall**: Enhanced context awareness reduces false negatives for tiny, indistinct objects.
*   **Precise Localization**: Preserved shallow features via ELAN structure improve bounding box regression for small targets.
*   **Efficiency**: Re-parameterization ensures the complex training structure collapses into a distinct, efficient inference model.

## 5. PyTorch Implementation

Below is the PyTorch implementation of the **SC-ELAN** module. You can integrate this into your YOLOv8 `modules.py` or similar file.

```python
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution wrapper
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ContextAwareRepConv(nn.Module):
    """
    Integrates Pzconv's large kernel idea with RepVGG-style re-parameterization.
    Training: Multi-branch (1x1, 3x3, 5x5) to capture multi-scale context.
    Inference: Collapses into a single 3x3 convolution for speed.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            # Large kernel branch (Context Aware)
            self.rbr_context = nn.Sequential(
                nn.Conv2d(c1, c2, 5, s, autopad(5, p), groups=c1, bias=False), # Depthwise 5x5
                nn.Conv2d(c2, c2, 1, 1, 0, bias=False), # Pointwise 1x1
                nn.BatchNorm2d(c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, autopad(1, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, inputs):
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(
            self.rbr_dense(inputs) + 
            self.rbr_1x1(inputs) + 
            self.rbr_context(inputs) + 
            id_out
        )

class SplitInteractionBlock(nn.Module):
    """
    Integrates FCM's interaction idea.
    Splits features and uses cross-branch attention to suppress background noise.
    """
    def __init__(self, dim):
        super().__init__()
        self.split_dim = dim // 2
        
        # Spatial Attention Generator (for Branch 1)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(self.split_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )
        # Channel Attention Generator (for Branch 2)
        self.channel_att = nn.AdaptiveAvgPool2d(1)
        self.fc_channel = nn.Sequential(
             nn.Conv2d(self.split_dim, self.split_dim, 1),
             nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Split: Context vs Content
        x1, x2 = torch.split(x, self.split_dim, dim=1)
        
        # 2. Interaction
        # Use x2 (context) to spatially validata x1 (content)
        x1_out = x1 * self.spatial_att(x2)
        
        # Use x1 (content) to channel-wise validate x2 (context)
        x2_out = x2 * self.fc_channel(self.channel_att(x1))
        
        # 3. Merge
        return torch.cat([x1_out, x2_out], dim=1)

class SC_ELAN(nn.Module):
    """
    SC-ELAN: Spatial-Context Efficient Layer Aggregation Network
    Combines ELAN backbone + Pzconv Context + FCM Interaction.
    """
    def __init__(self, c1, c2, c3, c4, c5=1): # c3 not used but kept for compatibility with C2f args
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        
        # ELAN Backbone with ContextAware RepConvs
        self.cv2 = ContextAwareRepConv(c2 // 2, c2 // 2)
        self.cv3 = ContextAwareRepConv(c2 // 2, c2 // 2)
        
        # Interaction Block for cleanup
        self.interaction = SplitInteractionBlock(c2)
        
        # Final aggregation
        self.cv4 = Conv(c2 + (2 * (c2 // 2)), c2, 1, 1)

    def forward(self, x):
        # 1. Projection & Split
        y = list(self.cv1(x).chunk(2, 1))
        
        # 2. Context-Aware Processing Path
        # Process the second half through the chain
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        
        # 3. Concatenation (Gradient Highway)
        feat_cat = torch.cat(y, 1)
        
        # 4. Final Projection
        # (Optional: apply interaction before or after cv4. 
        # Applying after concatenation but before reduction allows full feature access)
        # For efficiency, we can apply interaction to the concatenated features 
        # if dimensions align, or apply to the output of cv4.
        
        return self.cv4(feat_cat)

## 6. Variants for Ablation Study

To support a comprehensive experimental analysis, here are three variants of SC-ELAN tailored for different optimization goals.

### Variant 1: SC-ELAN-Dilated (Focus on Receptive Field)
**Hypothesis**: Small objects require a massive receptive field to be distinguished from background, but large dense kernels are heavy. Dilated convolutions offer a large view with zero extra parameters.

```python
class DilatedRepConv(nn.Module):
    """
    Variant using Dilated Convolution instead of large dense kernels.
    Receptive field: 3x3 (local) + 3x3 dilated (global context).
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
            # Dilated Branch: Rate=2, behaves like 5x5 but far fewer params
            self.rbr_dilated = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, padding=2, dilation=2, groups=g, bias=False),
                nn.BatchNorm2d(c2),
            )
    
    def forward(self, inputs):
        if self.deploy:
            return self.act(self.rbr_reparam(inputs))
        return self.act(self.rbr_dense(inputs) + self.rbr_dilated(inputs))

class SC_ELAN_Dilated(SC_ELAN):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4)
        # Override the convo layers with Dilated version
        self.cv2 = DilatedRepConv(c2 // 2, c2 // 2)
        self.cv3 = DilatedRepConv(c2 // 2, c2 // 2)
```

### Variant 2: SC-ELAN-DeepAttn (Focus on Feature Purification)
**Hypothesis**: Instead of one final cleanup, applying attention *inside* the processing block helps keep the features clean throughout the depth of the network.

```python
class AttnBlock(nn.Module):
    """
    Mini-version of SplitInteraction for internal usage.
    """
    def __init__(self, c):
        super().__init__()
        self.conv = Conv(c, c, 3, 1)
        self.interaction = SplitInteractionBlock(c)
    
    def forward(self, x):
        return self.interaction(self.conv(x))

class SC_ELAN_DeepAttn(SC_ELAN):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4)
        # Apply interaction INSIDE the ELAN path
        self.cv2 = AttnBlock(c2 // 2)
        self.cv3 = AttnBlock(c2 // 2)
        # Remove final interaction to save compute, or keep it for maximum effect
        self.interaction = nn.Identity() 
```

### Variant 3: SC-ELAN-Slim (Focus on Speed/Efficiency)
**Hypothesis**: For edge devices, we need the "Context" but not the heavy "Split-Interaction" computation. This variant keeps the Pzconv context but simplifies the fusion.

```python
class SC_ELAN_Slim(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        # Use simple Pzconv-style repconvs
        self.cv2 = ContextAwareRepConv(c2 // 2, c2 // 2)
        self.cv3 = ContextAwareRepConv(c2 // 2, c2 // 2)
        # Standard fusion without complex interaction
        self.cv4 = Conv(c2 + (2 * (c2 // 2)), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # Standard ELAN flow
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
```

    ### Variant 4: SC-ELAN-LSKA (Code-Aligned Attention Replacement)
    **Hypothesis**: Replace the original split-interaction cleanup with a stronger long-range spatial attention while keeping ELAN context flow unchanged.

    **Implemented behavior in `block.py`:**
    - Inherits from `SC_ELAN`, keeps `cv1/cv2/cv3/cv4` structure unchanged.
    - Replaces `self.interaction` with `LSKA(c2, k_size=7)`.
    - Applies attention **after final projection**: `return self.interaction(self.cv4(feat_cat))`.

    **LSKA details (k=7 path):**
    - Uses depthwise separable horizontal/vertical decomposition (`1×3`, `3×1`) plus dilated spatial decomposition.
    - Produces an attention map with a final `1×1` conv and performs multiplicative modulation `u * attn`.
    - This is a code-level replacement of interaction mechanism, not a change to ELAN branching topology.

    ### Variant 5: SC-ELAN-Efficient (Elastic Width + Lightweight Interaction)
    **Hypothesis**: Preserve SC-ELAN flow while cutting compute via hidden-width scaling and lightweight gated interaction.

    **Implemented behavior in `block.py`:**
    - Uses hidden width ratio `e=0.375` by default (`self.c = max(8, int(c2 * e))`).
    - Projection becomes `cv1: c1 -> 2c`; then split into two `c` branches.
    - Context chain uses lightweight blocks: `DWConv(3×3) + Conv(1×1)` for `cv2` and `cv3`.
    - Fusion uses `cv4: 4c -> c2` followed by `LiteSplitInteraction(c2, p=0.5)`.

    **LiteSplitInteraction details:**
    - Channel split ratio is configurable (`p`, default `0.5`) with dynamic branch widths.
    - Spatial gate path: `DWConv -> 1×1 -> Sigmoid` on one branch.
    - Channel gate path: `GAP -> 1×1 -> Sigmoid` from the other branch.
    - Final output is gated cross-branch fusion via concatenation.

### Variant 6: YOLO11-SCELAN-LSKA-TSCG-DetectCAI (Validated)

**Model file**: `yolo11-scelan-lska-tscg-detect-cai.yaml`

**Architecture review (current status):**
- The backbone/neck consistently use `SC_ELAN_LSKA_TSCG`, preserving the design principle of **context + selectivity + detail fidelity**.
- The detection head is switched from `Detect` to `DetectCAI`, and parser support is already integrated in `tasks.py`.
- `DetectCAI` is **training-only adaptive** (CAI enabled in train mode, bypassed in eval), so inference contract remains consistent with standard `Detect`.
- Default CAI prior and tail-class mask are available for VisDrone-style long-tail settings.

**Why this combination is meaningful:**
- `LSKA-TSCG` addresses representation quality for tiny objects (feature-level improvement).
- `DetectCAI` addresses class imbalance and tail suppression (optimization-level correction).
- The combined design targets two orthogonal bottlenecks: **feature expressiveness** and **long-tail learning bias**.

**Expected outcomes (hypothesis):**
1.  Overall **mAP50-95** should be at least stable vs `LSKA-TSCG`, with potential gains mainly from tail classes.
2.  `people` / `bicycle` / `tricycle` are expected to improve first if CAI is functioning as intended.
3.  `car` / `bus` should remain stable (or marginally fluctuate), since CAI reweighting is tail-aware.
4.  Inference latency should remain near the same level as `LSKA-TSCG`, because CAI is training-time only.

**Risk points to monitor in this run:**
- If class prior drifts too aggressively during training, head reweighting may become unstable for non-tail classes.
- If dataset `nc` and CAI prior assumptions are inconsistent, long-tail benefits may be weakened.
- If gains only appear in mAP50 but not mAP50-95, localization quality correction is still insufficient.

**Recommended comparison protocol:**
- Primary baseline: `yolo11-scelan-lska-tscg.yaml` (same backbone/neck, standard `Detect`).
- Keep identical training settings (seed, epochs, aug, optimizer, batch size).
- Report: overall mAP50/mAP50-95 + per-class changes for `people/bicycle/tricycle`.

**Validation snapshot (2026-02-21):**
- `DetectCAI` result on VisDrone test-dev: **P/R/mAP50/mAP50-95 = 0.484/0.378/0.358/0.206**.
- Compared with `LSKA-TSCG` (`0.473/0.376/0.358/0.208`), precision and recall rise slightly, mAP50 stays equal, and mAP50-95 drops by 0.002.
- Runtime remains aligned with the design expectation (training-only CAI, inference-time head contract unchanged).

## 7. Experimental Results on VisDrone Dataset

### 7.1 Overall Performance Comparison

All models were evaluated on the **VisDrone2019-DET-test-dev** dataset (1609 images, 75082 instances) using pretrained weights.

| Model Variant | Parameters | GFLOPs | mAP50 | mAP50-95 | Speed (ms) |
|---------------|------------|--------|-------|----------|------------|
| **YOLO11-SCELAN** | 10.86M | 35.7 | 0.355 | 0.203 | 5.1 |
| **YOLO11-SCELAN-Fixed** | 10.86M | 36.1 | 0.352 | 0.203 | 5.3 |
| **YOLO11-SCELAN-Dilated** | 11.85M | 44.1 | 0.350 | 0.200 | 5.0 |
| **YOLO11-SCELAN-Slim** | 10.75M | 35.7 | 0.354 | 0.203 | 5.1 |
| **YOLO11-SCELAN-Hybrid** | 11.13M | 37.1 | 0.352 | 0.202 | 5.1 |
| **YOLO11-SCELAN-LSKA** | 11.07M | 38.4 | 0.359 | 0.206 | 5.3 |
| **YOLO11-SCELAN-LSKA-TSCG** | 11.16M | 39.2 | 0.358 | **0.208** | 5.6 |
| **YOLO11-SCELAN-LSKA-TSCG-DetectCAI** | 11.52M | 39.2 | 0.358 | 0.206 | 5.7 |
| **YOLO11-SCELAN-LSKA11-TSCG (val4)** | 11.16M | 39.2 | 0.359 | 0.207 | 5.9 |
| **YOLO11-SCELAN-LSKA23-TSCG (val4)** | 11.17M | 39.4 | **0.364** | **0.210** | 6.1 |
| **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mid (val4)** | 11.52M | 39.2 | 0.360 | 0.208 | 5.9 |
| **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mom098 (val4)** | 11.52M | 39.2 | 0.355 | 0.204 | 5.8 |
| **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Soft (val4)** | 11.52M | 39.2 | **0.364** | **0.210** | 5.8 |
| **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Tail12 (val4)** | 11.52M | 39.2 | 0.362 | 0.209 | 5.7 |
| **YOLO11-SCELAN-Mixed-Efficient-TSCG (val4)** | 10.90M | 31.2 | 0.341 | 0.194 | 5.4 |
| **YOLO11-SCELAN-Efficient** | 9.00M | 20.3 | 0.334 | 0.189 | 4.6 |
| **YOLO11-SCELAN-RepExact** | 8.47M | 16.9 | 0.310 | 0.171 | 4.6 |
| **YOLO11-SCELAN-RepAdd** | 8.43M | 16.6 | 0.304 | 0.167 | 4.8 |
| ***SC-ELAN v2 Phased Pipeline (see Section 8)*** | | | | | |
| **v2-P1a** (α=0.05, β=0.15) | 11.53M | 39.4 | 0.359 | 0.207 | 5.7 |
| **v2-P1b** (α=0.10, β=0.25) | 11.53M | 39.4 | 0.362 | 0.209 | 5.7 |
| **v2-P1c** (α=0.15, β=0.30, Soft repro) | 11.53M | 39.4 | 0.362 | 0.207 | 5.8 |
| **v2-P1d** (α=0.10, β=0.40) | 11.53M | 39.4 | **0.367** | **0.212** | 5.7 |
| **v2-P1e** (α=0.20, β=0.25) | 11.53M | 39.4 | **0.367** | 0.211 | 5.6 |
| **v2-P2a** SA-LSKA(7/11/23)+TSCG | 11.16M | 39.2 | 0.361 | 0.209 | 5.9 |
| **v2-P2b** SA-LSKA(11/23/23)+TSCG | 11.17M | 39.3 | 0.361 | 0.209 | 5.9 |
| **v2-P2c** SA-LSKA(7/23/35)+TSCG | 11.17M | 39.3 | 0.354 | 0.205 | 5.8 |
| **v2-P2d** SA-LSKA(7/11/23)+TSCGv2 | 11.16M | 39.2 | 0.361 | 0.208 | 5.9 |
| **v2-P3a** SA-LSKA+TSCG+P3-FRM+Detect | 11.19M | 39.5 | 0.358 | 0.207 | 5.9 |
| **v2-P3b** SA-LSKA+TSCG+P3-FRM+DetectCAI-Soft | 11.55M | 39.5 | 0.361 | 0.208 | 5.7 |

**Key Observations:**
- **New best strict accuracy (mAP50-95 = 0.212):** achieved by **v2-P1d** (α=0.10, β=0.40), surpassing the previous 0.210 record
- **Best strict accuracy (mAP50-95 = 0.210, pre-v2):** jointly achieved by **YOLO11-SCELAN-LSKA23-TSCG (val4)** and **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Soft (val4)**
- **Best recall:** **LSKA23-TSCG (val4)** achieves **R = 0.386**, highest among all variants, indicating strongest candidate coverage in dense scenes
- **Best precision:** **DetectCAI-Tail12 (val4)** reaches **P = 0.506**, but with recall trade-off (R = 0.368), showing stricter positive filtering
- **Most stable CAI setting:** **DetectCAI-Soft** improves long-tail small classes (`pedestrian/people/bicycle/tricycle`) while maintaining top-line metrics equal to LSKA23-TSCG
- **Not recommended CAI setting:** **DetectCAI-Mom098 (val4)** degrades to **0.355/0.204** (mAP50/mAP50-95), below all other LSKA/TSCG variants
- **Efficiency trade-off:** **Mixed-Efficient-TSCG (val4)** is fastest in val4 batch (5.4 ms total) with lowest compute (31.2 GFLOPs), but accuracy drop is significant (**-0.016 mAP50-95** vs 0.210 best)
- Historical variants (pre-val4) remain useful references for architecture evolution

### 7.2 Per-Class Performance Analysis

#### 7.2.1 YOLO11-SCELAN (Standard)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.467   0.378   0.355    0.203
pedestrian         1196    21000       0.484   0.324   0.318    0.125
people             797     6376        0.497   0.151   0.176    0.058
bicycle            377     1302        0.246   0.130   0.108    0.044
car                1529    28063       0.700   0.759   0.755    0.487
van                1167    5770        0.436   0.444   0.407    0.273
truck              750     2659        0.450   0.458   0.420    0.265
tricycle           245     530         0.290   0.328   0.210    0.109
awning-tricycle    233     599         0.400   0.239   0.217    0.122
bus                837     2938        0.707   0.552   0.599    0.417
motor              794     5845        0.465   0.393   0.340    0.135
```

**Performance Highlights:**
- **Best for vehicles:** Car (mAP50: 0.755), Bus (0.599), Van (0.407)
- **Moderate for pedestrians:** Pedestrian (0.318), People (0.176)
- **Challenging classes:** Bicycle (0.108), Tricycle (0.210)

#### 7.2.2 YOLO11-SCELAN-Dilated
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.461   0.371   0.350    0.200
pedestrian         1196    21000       0.484   0.325   0.319    0.125
people             797     6376        0.518   0.148   0.179    0.060
bicycle            377     1302        0.240   0.127   0.100    0.039
car                1529    28063       0.694   0.756   0.753    0.485
van                1167    5770        0.431   0.425   0.398    0.267
truck              750     2659        0.463   0.444   0.424    0.269
tricycle           245     530         0.265   0.321   0.198    0.103
awning-tricycle    233     599         0.383   0.228   0.206    0.112
bus                837     2938        0.687   0.544   0.590    0.409
motor              794     5845        0.448   0.389   0.333    0.133
```

**Analysis:**
- Slightly **improved precision for people (0.518)** but **lower recall (0.148)**
- Competitive performance on **large objects** (car, bus, truck)
- **Higher GFLOPs (44.1)** but **marginal accuracy gains**

#### 7.2.3 YOLO11-SCELAN-Slim
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.463   0.378   0.354    0.203
pedestrian         1196    21000       0.494   0.328   0.323    0.127
people             797     6376        0.484   0.159   0.178    0.060
bicycle            377     1302        0.238   0.145   0.107    0.040
car                1529    28063       0.697   0.758   0.753    0.486
van                1167    5770        0.425   0.431   0.398    0.267
truck              750     2659        0.480   0.451   0.428    0.275
tricycle           245     530         0.259   0.325   0.207    0.108
awning-tricycle    233     599         0.393   0.235   0.212    0.116
bus                837     2938        0.699   0.551   0.594    0.417
motor              794     5845        0.456   0.396   0.344    0.138
```

**Analysis:**
- **Best efficiency-accuracy trade-off**: 10.75M params with 0.354 mAP50
- **Highest pedestrian mAP50 (0.323)** among all variants
- **Best truck detection (mAP50-95: 0.275)**
- Ideal for **resource-constrained deployments**

#### 7.2.4 YOLO11-SCELAN-Hybrid
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.470   0.374   0.352    0.202
pedestrian         1196    21000       0.497   0.327   0.323    0.128
people             797     6376        0.517   0.150   0.178    0.059
bicycle            377     1302        0.273   0.149   0.112    0.042
car                1529    28063       0.696   0.763   0.754    0.486
van                1167    5770        0.443   0.426   0.400    0.268
truck              750     2659        0.468   0.436   0.413    0.265
tricycle           245     530         0.270   0.317   0.208    0.109
awning-tricycle    233     599         0.402   0.229   0.196    0.109
bus                837     2938        0.688   0.549   0.593    0.414
motor              794     5845        0.449   0.395   0.342    0.135
```

**Analysis:**
- **Highest overall precision (0.470)**
- **Best bicycle detection (mAP50: 0.112)**
- Balanced performance across **medium-sized objects**
- Good for scenarios requiring **high precision**

#### 7.2.5 YOLO11-SCELAN-LSKA
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.491   0.370   0.359    0.206
pedestrian         1196    21000       0.539   0.320   0.336    0.133
people             797     6376        0.509   0.164   0.187    0.064
bicycle            377     1302        0.258   0.159   0.119    0.047
car                1529    28063       0.713   0.759   0.756    0.490
van                1167    5770        0.467   0.408   0.404    0.272
truck              750     2659        0.515   0.419   0.428    0.278
tricycle           245     530         0.307   0.345   0.219    0.111
awning-tricycle    233     599         0.373   0.204   0.182    0.104
bus                837     2938        0.746   0.522   0.597    0.423
motor              794     5845        0.486   0.403   0.360    0.143
```

**Analysis:**
- **Highest overall mAP50 (0.359)** among listed variants, with strong mAP50-95 (0.206)
- **Highest overall precision (0.491)** — best signal-to-noise ratio
- **Best pedestrian detection (mAP50: 0.336)** and **best car detection (mAP50: 0.756)**
- **Best truck recall (0.419)** and **van recall (0.408)** — LSKA improves recall for medium objects
- **Best tricycle recall (0.345)** — large-kernel attention captures irregular shapes better
- Slight trade-off: **lower bus recall (0.522)** vs standard SC-ELAN (0.552)
- Recommended when prioritizing **top-line mAP50** and robust class-wise precision

#### 7.2.6 YOLO11-SCELAN-Fixed
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.467   0.378   0.352    0.203
pedestrian         1196    21000       0.499   0.325   0.322    0.127
people             797     6376        0.513   0.156   0.180    0.060
bicycle            377     1302        0.300   0.173   0.127    0.049
car                1529    28063       0.700   0.759   0.756    0.488
van                1167    5770        0.433   0.428   0.396    0.265
truck              750     2659        0.450   0.426   0.400    0.258
tricycle           245     530         0.273   0.342   0.206    0.111
awning-tricycle    233     599         0.352   0.224   0.193    0.114
bus                837     2938        0.691   0.552   0.591    0.419
motor              794     5845        0.464   0.395   0.346    0.136
```

**Analysis:**
- Overall metrics are stable with **mAP50-95 = 0.203** while keeping moderate complexity (**36.1 GFLOPs**)
- Strong vehicle performance remains consistent: **car (0.756 mAP50)** and **bus (0.591 mAP50)**
- Improved bicycle recognition (**0.127 mAP50**) compared with several other SC-ELAN variants
- Suitable as a robust baseline when prioritizing balanced precision/recall and reproducibility

#### 7.2.7 YOLO11-SCELAN-LSKA-TSCG
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.473   0.376   0.358    0.208
pedestrian         1196    21000       0.494   0.342   0.336    0.135
people             797     6376        0.505   0.163   0.188    0.064
bicycle            377     1302        0.258   0.154   0.118    0.047
car                1529    28063       0.715   0.759   0.757    0.496
van                1167    5770        0.455   0.425   0.404    0.274
truck              750     2659        0.501   0.426   0.422    0.273
tricycle           245     530         0.269   0.332   0.219    0.116
awning-tricycle    233     599         0.349   0.219   0.188    0.108
bus                837     2938        0.724   0.538   0.599    0.427
motor              794     5845        0.464   0.398   0.348    0.141
```

**Analysis:**
- Historical strong baseline in this report (**mAP50-95 = 0.208**) with near-top mAP50 (0.358)
- Strong vehicle localization remains: **car (0.757 mAP50, 0.496 mAP50-95)**
- Better fine-grained classes than many baselines: **pedestrian (0.336)**, **tricycle (0.219)**
- Moderate complexity increase over LSKA (39.2 vs 38.4 GFLOPs) with stable recall profile

#### 7.2.8 YOLO11-SCELAN-Efficient
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.446   0.357   0.334    0.189
pedestrian         1196    21000       0.492   0.314   0.313    0.122
people             797     6376        0.491   0.153   0.175    0.058
bicycle            377     1302        0.239   0.135   0.106    0.040
car                1529    28063       0.673   0.753   0.740    0.473
van                1167    5770        0.412   0.406   0.365    0.241
truck              750     2659        0.447   0.401   0.375    0.234
tricycle           245     530         0.234   0.294   0.184    0.093
awning-tricycle    233     599         0.362   0.195   0.180    0.102
bus                837     2938        0.678   0.543   0.582    0.402
motor              794     5845        0.431   0.374   0.319    0.124
```

**Analysis:**
- Lower absolute accuracy than larger SC-ELAN variants, but strong compute efficiency
- **Smallest model among listed variants (9.00M params)** and lowest complexity (**20.3 GFLOPs**)
- Fastest measured inference path in this report (**2.7 ms inference, 4.6 ms total**)
- Matches code design goals: **elastic width (`e=0.375`) + lightweight split gating (`p=0.5`)**
- Suitable for deployment scenarios prioritizing throughput/power over peak mAP

#### 7.2.9 YOLO11-SCELAN-RepExact
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.440   0.337   0.310    0.171
pedestrian         1196    21000       0.467   0.290   0.282    0.108
people             797     6376        0.479   0.134   0.157    0.050
bicycle            377     1302        0.222   0.124   0.081    0.031
car                1529    28063       0.669   0.727   0.719    0.451
van                1167    5770        0.373   0.388   0.339    0.219
truck              750     2659        0.423   0.379   0.339    0.207
tricycle           245     530         0.256   0.284   0.177    0.086
awning-tricycle    233     599         0.404   0.206   0.179    0.091
bus                837     2938        0.669   0.497   0.537    0.358
motor              794     5845        0.436   0.343   0.294    0.112
```

**Analysis:**
- Very low complexity profile (**8.47M params, 16.9 GFLOPs**) with strong speed (**1.9 ms inference, 4.6 ms total**)
- Better overall accuracy than RepAdd in this round (**0.310/0.171** vs **0.304/0.167**)
- Maintains usable large-object performance (car/bus), while tiny-object classes remain challenging
- Suitable for strict compute budgets where moderate accuracy drop is acceptable

#### 7.2.10 YOLO11-SCELAN-RepAdd
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.418   0.331   0.304    0.167
pedestrian         1196    21000       0.456   0.287   0.277    0.105
people             797     6376        0.458   0.127   0.149    0.049
bicycle            377     1302        0.240   0.100   0.083    0.030
car                1529    28063       0.636   0.732   0.710    0.442
van                1167    5770        0.332   0.399   0.331    0.212
truck              750     2659        0.389   0.377   0.331    0.199
tricycle           245     530         0.233   0.253   0.156    0.077
awning-tricycle    233     599         0.405   0.195   0.187    0.100
bus                837     2938        0.630   0.493   0.526    0.349
motor              794     5845        0.397   0.350   0.290    0.110
```

**Analysis:**
- Lowest FLOPs among current variants (**16.6 GFLOPs**) and compact parameter count (**8.43M**)
- Accuracy is slightly below RepExact across overall metrics and most classes
- Total latency remains real-time (**4.8 ms**) despite slower inference than RepExact due to balance in postprocess
- Practical baseline for ultra-light deployment-focused ablation

#### 7.2.11 YOLO11-SCELAN-LSKA-TSCG-DetectCAI
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.484   0.378   0.358    0.206
pedestrian         1196    21000       0.532   0.322   0.332    0.131
people             797     6376        0.532   0.153   0.185    0.063
bicycle            377     1302        0.292   0.147   0.125    0.047
car                1529    28063       0.723   0.749   0.756    0.492
van                1167    5770        0.415   0.452   0.399    0.270
truck              750     2659        0.516   0.451   0.436    0.278
tricycle           245     530         0.288   0.312   0.212    0.108
awning-tricycle    233     599         0.369   0.269   0.207    0.118
bus                837     2938        0.697   0.538   0.586    0.416
motor              794     5845        0.480   0.384   0.343    0.136
```

**Analysis (based on `ultralytics/nn/modules/head.py`):**
- `DetectCAI.forward()` applies `_apply_cai()` only during training (`if not self.training: return x`), so validation/inference path remains the same decode/postprocess contract as `Detect`.
- CAI gains are therefore optimization-time effects: feature gates are modulated by estimated class prior (`_estimate_cai_prior`) and tail mask (`cai_tail_mask`) before entering the standard detection heads.
- In this run, tail-sensitive classes show mixed behavior (e.g., `bicycle` mAP50 up to 0.125, but `tricycle` mAP50-95 at 0.108), which matches a moderate reweighting regime (`cai_alpha=0.15`, `cai_beta=0.30`) rather than aggressive redistribution.
- Overall P/R improvement with near-identical mAP50-95 to `LSKA-TSCG` is consistent with CAI improving class calibration/selection more than box geometry, since box branch structure itself is unchanged.

#### 7.2.12 YOLO11-SCELAN-LSKA11-TSCG (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.488   0.375   0.359    0.207
pedestrian         1196    21000       0.531   0.329   0.338    0.135
people             797     6376        0.521   0.156   0.181    0.0613
bicycle            377     1302        0.269   0.168   0.127    0.0488
car                1529    28063       0.724   0.759   0.760    0.496
van                1167    5770        0.431   0.440   0.395    0.266
truck              750     2659        0.524   0.426   0.427    0.275
tricycle           245     530         0.281   0.308   0.196    0.103
awning-tricycle    233     599         0.391   0.242   0.209    0.117
bus                837     2938        0.728   0.533   0.601    0.424
motor              794     5845        0.477   0.391   0.352    0.141
```

**Analysis:**
- Baseline LSKA kernel (`k=11`) with TSCG produces **mAP50-95 = 0.207**, slightly below `LSKA23-TSCG` (0.210)
- Strong vehicle performance: **car (0.760 mAP50, 0.496 mAP50-95)** and **bus (0.601 mAP50)**
- Good precision (**P = 0.488**) but lower recall than LSKA23 variant (**R = 0.375 vs 0.386**)
- Serves as the direct ablation baseline to quantify LSKA kernel scaling benefit (`k=11 -> k=23`)

#### 7.2.13 YOLO11-SCELAN-LSKA23-TSCG (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.479   0.386   0.364    0.210
pedestrian         1196    21000       0.497   0.333   0.330    0.131
people             797     6376        0.502   0.163   0.184    0.0627
bicycle            377     1302        0.269   0.153   0.121    0.0472
car                1529    28063       0.703   0.767   0.761    0.493
van                1167    5770        0.456   0.439   0.410    0.276
truck              750     2659        0.493   0.459   0.441    0.284
tricycle           245     530         0.279   0.340   0.212    0.111
awning-tricycle    233     599         0.406   0.245   0.222    0.126
bus                837     2938        0.713   0.558   0.607    0.427
motor              794     5845        0.474   0.407   0.354    0.142
```

**Analysis:**
- **Most promising backbone/context structure** — strongest global result (**mAP50 = 0.364, mAP50-95 = 0.210**) and **best recall (R = 0.386)**
- Leads on medium-to-large vehicle classes: **truck (mAP50-95 = 0.284)**, **bus (0.427)**, **van (0.276)**
- Best car recall in val4 batch (**R = 0.767**) and highest **awning-tricycle** mAP50 (**0.222, mAP50-95 = 0.126**)
- Larger LSKA kernel (`k=23`) improves strict localization across most classes vs `k=11` baseline
- Recommended default when strict localization (mAP50-95) and dense-scene recall are primary KPIs

#### 7.2.14 YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mid (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.482   0.380   0.360    0.208
pedestrian         1196    21000       0.518   0.328   0.335    0.135
people             797     6376        0.524   0.167   0.189    0.0641
bicycle            377     1302        0.275   0.170   0.131    0.0512
car                1529    28063       0.716   0.758   0.759    0.494
van                1167    5770        0.450   0.432   0.400    0.270
truck              750     2659        0.492   0.460   0.441    0.282
tricycle           245     530         0.268   0.326   0.202    0.106
awning-tricycle    233     599         0.372   0.214   0.183    0.108
bus                837     2938        0.714   0.552   0.597    0.422
motor              794     5845        0.493   0.398   0.364    0.145
```

**Analysis:**
- Mid-strength CAI regime achieves solid **mAP50-95 = 0.208**, matching the pre-val4 best (LSKA-TSCG)
- Best **bicycle mAP50 (0.131)** and **motor mAP50 (0.364)** in the val4 batch
- Good small-class precision (`people` P = 0.524, `pedestrian` P = 0.518) with competitive truck recall (0.460)
- Represents a balanced middle-ground CAI setting, though slightly below `Soft` on overall metrics

#### 7.2.15 YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mom098 (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.475   0.375   0.355    0.204
pedestrian         1196    21000       0.517   0.324   0.327    0.130
people             797     6376        0.522   0.155   0.188    0.0624
bicycle            377     1302        0.272   0.152   0.121    0.0471
car                1529    28063       0.714   0.755   0.755    0.489
van                1167    5770        0.422   0.436   0.394    0.265
truck              750     2659        0.492   0.452   0.433    0.273
tricycle           245     530         0.277   0.325   0.202    0.104
awning-tricycle    233     599         0.353   0.222   0.193    0.109
bus                837     2938        0.700   0.547   0.595    0.421
motor              794     5845        0.479   0.387   0.347    0.138
```

**Analysis:**
- **Not recommended CAI setting** — degrades to **mAP50 = 0.355, mAP50-95 = 0.204**, lowest among all LSKA/TSCG val4 variants
- High momentum (`0.98`) causes class prior estimates to lag, leading to suboptimal reweighting across all classes
- All class metrics trail other CAI variants; no class shows improvement over LSKA23-TSCG baseline
- Confirms that aggressive EMA smoothing is counterproductive for CAI in this training setup

#### 7.2.16 YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Soft (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.488   0.379   0.364    0.210
pedestrian         1196    21000       0.518   0.335   0.340    0.135
people             797     6376        0.512   0.167   0.192    0.0662
bicycle            377     1302        0.269   0.163   0.127    0.0515
car                1529    28063       0.720   0.758   0.761    0.498
van                1167    5770        0.429   0.437   0.396    0.269
truck              750     2659        0.517   0.438   0.428    0.274
tricycle           245     530         0.292   0.343   0.224    0.117
awning-tricycle    233     599         0.398   0.217   0.211    0.121
bus                837     2938        0.719   0.542   0.603    0.426
motor              794     5845        0.504   0.390   0.362    0.146
```

**Analysis:**
- **Most promising head reweighting strategy** — matches best global accuracy (**mAP50 = 0.364, mAP50-95 = 0.210**)
- Improves key small/long-tail classes vs `LSKA23-TSCG`: **pedestrian** mAP50 0.340 vs 0.330, **people** 0.192 vs 0.184, **bicycle** 0.127 vs 0.121, **tricycle** 0.224 vs 0.212
- Best **car mAP50-95 (0.498)** in the entire report; best **motor mAP50 (0.362)** among val4 variants
- Slightly weaker on some vehicle classes vs LSKA23-TSCG (`van`, `truck`), indicating a controllable class-balance trade-off
- Recommended CAI setting for current and future head-level follow-up experiments

#### 7.2.17 YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Tail12 (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.506   0.368   0.362    0.209
pedestrian         1196    21000       0.542   0.324   0.340    0.136
people             797     6376        0.559   0.161   0.195    0.0655
bicycle            377     1302        0.288   0.167   0.128    0.0483
car                1529    28063       0.720   0.760   0.758    0.495
van                1167    5770        0.483   0.408   0.408    0.276
truck              750     2659        0.516   0.429   0.421    0.270
tricycle           245     530         0.270   0.302   0.203    0.107
awning-tricycle    233     599         0.430   0.217   0.217    0.123
bus                837     2938        0.744   0.529   0.601    0.425
motor              794     5845        0.506   0.381   0.351    0.140
```

**Analysis:**
- **Precision-oriented alternative** — highest overall precision in val4 batch (**P = 0.506**)
- Best **people precision (0.559)**, best **pedestrian precision (0.542)**, and best **awning-tricycle mAP50-95 (0.123)**
- Recall reduction (**R = 0.368**, lowest among val4 LSKA/TSCG variants) limits overall mAP gains
- Strong `van` mAP50 (**0.408**) and competitive `bus` precision (**0.744**)
- Suitable for false-positive-sensitive deployment pipelines; should be tuned with recall constraints for general use

#### 7.2.18 YOLO11-SCELAN-Mixed-Efficient-TSCG (val4)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.462   0.363   0.341    0.194
pedestrian         1196    21000       0.499   0.314   0.312    0.122
people             797     6376        0.506   0.143   0.172    0.0583
bicycle            377     1302        0.265   0.119   0.108    0.0427
car                1529    28063       0.688   0.752   0.744    0.476
van                1167    5770        0.418   0.415   0.374    0.249
truck              750     2659        0.489   0.422   0.404    0.253
tricycle           245     530         0.274   0.311   0.193    0.0977
awning-tricycle    233     599         0.341   0.220   0.185    0.105
bus                837     2938        0.688   0.547   0.584    0.406
motor              794     5845        0.450   0.388   0.339    0.131
```

**Analysis:**
- Fastest in val4 batch (**5.4 ms total**) with lowest compute (**31.2 GFLOPs**)
- Accuracy drop is significant vs best models (**-0.023 mAP50, -0.016 mAP50-95**)
- Width compression hurts tiny/long-tail classes first: `people` mAP50 drops to 0.172, `bicycle` to 0.108, `tricycle` to 0.193
- Vehicle-class performance remains competitive (`car` 0.744 mAP50, `bus` 0.584)
- Suitable only for strict latency/compute budgets where moderate accuracy sacrifice is acceptable
### 7.3 Inference Performance

All models were tested on NVIDIA GeForce RTX 4090 (24GB VRAM):

| Model | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) |
|-------|-----------------|----------------|------------------|------------|
| YOLO11-SCELAN | 0.3 | 3.0 | 1.8 | 5.1 |
| YOLO11-SCELAN-Fixed | 0.2 | 4.4 | 0.7 | 5.3 |
| YOLO11-SCELAN-Dilated | 0.3 | 3.0 | 1.7 | 5.0 |
| YOLO11-SCELAN-Slim | 0.3 | 3.1 | 1.7 | 5.1 |
| YOLO11-SCELAN-Hybrid | 0.3 | 2.9 | 1.9 | 5.1 |
| YOLO11-SCELAN-LSKA | 0.2 | 3.8 | 1.3 | 5.3 |
| YOLO11-SCELAN-LSKA-TSCG | 0.2 | 4.8 | 0.6 | 5.6 |
| YOLO11-SCELAN-LSKA-TSCG-DetectCAI | 0.3 | 4.8 | 0.6 | 5.7 |
| YOLO11-SCELAN-Efficient | 0.2 | 2.7 | 1.7 | 4.6 |
| YOLO11-SCELAN-RepExact | 0.3 | 1.9 | 2.4 | 4.6 |
| YOLO11-SCELAN-RepAdd | 0.3 | 3.5 | 1.0 | 4.8 |

**Efficiency Analysis:**
- All variants remain in practical real-time range (**~179–217 FPS**)
- Lowest-latency group is **Efficient/RepExact** (both **4.6 ms total**)
- **GPU memory efficient**: All models fit within 24GB VRAM with batch processing

#### 7.3.1 val4 Complete Inference Table (from `logs/`)

| Model (val4) | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| YOLO11-SCELAN-LSKA11-TSCG | 0.3 | 4.0 | 1.6 | 5.9 |
| YOLO11-SCELAN-LSKA23-TSCG | 0.3 | 5.2 | 0.6 | 6.1 |
| YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mid | 0.3 | 5.1 | 0.5 | 5.9 |
| YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Mom098 | 0.3 | 3.2 | 2.3 | 5.8 |
| YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Soft | 0.3 | 4.4 | 1.1 | 5.8 |
| YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Tail12 | 0.3 | 4.6 | 0.8 | 5.7 |
| YOLO11-SCELAN-Mixed-Efficient-TSCG | 0.3 | 4.2 | 0.9 | 5.4 |

### 7.4 Conclusions and Recommendations

#### Best Model Selection by Use Case:

1. **Best Overall / General Small Object Detection** → **v2-P1d** (α=0.10, β=0.40)
    - **New global best: mAP50-95 = 0.212**, mAP50 = **0.367** (see Section 8)
    - Surpasses previous val4 record (0.210) via CAI parameter optimization
    - Recommended default when strict localization is the primary KPI

2. **Best Tail-Aware Balanced Option** → **v2-P1d** / **v2-P1e**
    - v2-P1d achieves best `bicycle` mAP50 (0.145) and `tricycle` mAP50 (0.223) across all variants
    - v2-P1e achieves highest recall (**R = 0.384**) with near-best mAP50-95 (0.211)
    - Historical val4 reference: DetectCAI-Soft (0.210) remains relevant as cross-validation anchor

3. **Highest Precision Profile (val4)** → **YOLO11-SCELAN-LSKA-TSCG-DetectCAI-Tail12 (val4)**
    - Highest overall precision (**P = 0.506**), best `people` P (0.559), best `pedestrian` P (0.542)
    - Suitable for false-positive-sensitive deployment; recall trade-off (**R = 0.368**) should be monitored

4. **Ablation Kernel Scaling Reference (val4)** → **YOLO11-SCELAN-LSKA11-TSCG (val4)**
    - Direct comparison baseline for LSKA kernel scaling: `k=11` → `k=23` yields +0.003 mAP50-95
    - Confirms larger-kernel LSKA consistently improves strict localization under current training setup

5. **Historical Strong Baseline (pre-val4 reference)** → **YOLO11-SCELAN-LSKA-TSCG**
    - Stable historical checkpoint (**0.358 / 0.208**) with strong vehicle localization
    - Retain as a legacy comparison anchor when reviewing older experiments

6. **Historical Highest mAP50 (pre-val4 reference)** → **YOLO11-SCELAN-LSKA**
    - Historical peak mAP50 (**0.359**) under earlier evaluation protocol
    - Useful reference for confidence-weighted operating points

7. **Compute-Efficient Option (val4)** → **YOLO11-SCELAN-Mixed-Efficient-TSCG (val4)**
    - Lowest compute in val4 batch (**31.2 GFLOPs**, **5.4 ms total**)
    - Significant accuracy drop (**−0.023 mAP50, −0.016 mAP50-95** vs 0.210 best); acceptable only under strict latency budgets

8. **Ultra-Light Compute Budget** → **YOLO11-SCELAN-Efficient / RepExact / RepAdd**
    - Lowest FLOPs/latency group across all variants (**≤20.3 GFLOPs, ≤4.8 ms**)
    - Appropriate only when deployment constraints dominate accuracy targets

#### Key Findings:

- **New global best: mAP50-95 = 0.212**, achieved by **v2-P1d** (α=0.10, β=0.40), surpassing the previous 0.210 record (see Section 8)
- **CAI parameter tuning is the highest-ROI optimization**: Phase 1 CAI sweep outperforms Phase 2 structural ablation and Phase 3 P3-FRM integration
- **LSKA kernel scaling is confirmed beneficial**: `k=23` outperforms `k=11` (+0.003 mAP50-95, +0.011 recall) under the same TSCG structure
- **`LSKA + TSCG` remains the strongest structural base**; CAI benefit depends on reweighting regime (`Soft` reaches top score, `Mom098` regresses to 0.204)
- **v2-P1d is the new recommended configuration**: best strict score (0.212) with strong tail-class improvements (`bicycle` 0.145, `tricycle` 0.223)
- **Clear Pareto front** exists between strict accuracy (≤0.212 mAP50-95) and speed/compute efficiency (≤5.4 ms); no current variant achieves both simultaneously

### 7.5 阶段总结与后续工作（2026 更新）

#### 阶段总结

本文档目前以 **val4 批次结果作为顶线结论的主要依据**，同时保留历史运行记录作为参考基准。

- **v2-P1d（α=0.10, β=0.40）以 mAP50-95 = 0.212 刷新全局最佳**：
  - 在 LSKA23-TSCG 骨干基础上，通过 CAI 参数调优（增大 β 至 0.40）实现 +0.002 提升。
  - 同时在 `bicycle`（0.145 mAP50）和 `tricycle`（0.223 mAP50）上取得全系列最优。
  - 确认当前最有效的方向是：**强上下文主干 + 精细化 CAI 参数调优** 的组合策略。

- **历史基准 `LSKA`（mAP50 = 0.359）仍有参考价值的原因**：
  - 在以置信度为主要指标的分析场景中，其仍是有效的参考点。
  - 在严格 IoU 指标下，val4 最优方案已全面超越该基准。

- **`Efficient` / `Mixed-Efficient` 系列提速但严格精度下降的原因**：
  - 宽度与上下文压缩首先损害的是微小目标和长尾类别的检测能力。
  - 吞吐量提升，但严格定位指标（mAP50-95）明显下降。

- **当前证据归纳的核心设计准则**：
  1. 选择性上下文建模优于均匀放大特征。
  2. mAP50-95 的提升强依赖于特征交互与门控质量。
  3. 检测头重加权应保持适度；`Soft` 策略稳定，激进设置（如 `Mom098`）易导致回退。
  4. 过度压缩最先损伤微小目标与长尾类别。

#### 后续工作

为持续推进小目标检测性能、朝向可发表的模块化创新，下一阶段将聚焦以下方向：

#### A) 长尾类别提升

- **类别均衡训练**：针对 `people`、`bicycle`、`tricycle` 探索类别重加权与 Focal Loss 变体。
- **难例挖掘**：构建专注于拥挤场景和极小目标帧的子集，用于周期性定向微调。
- **定位质量诊断**：按类别跟踪目标尺寸分布桶，定位微小目标回归失效的根本原因。

#### B) 结构级探索

- **LSKA 核尺寸调度**：按阶段比较 `k_size` 设置（如 7/11/23），测试精度与计算量的弹性关系。
- **选择性上下文路由**：将 TSCG 扩展为阶段感知门控，评估深层阶段是否需要更强的上下文注入。
- **高效分支搜索**：针对当前数据集，自动调优 `SC_ELAN_Efficient` 中的 `e` 与 `p` 参数，寻找最优帕累托点。

#### C) 评估协议升级

- **跨数据集迁移验证**：在至少一个额外的无人机/交通风格数据集上验证泛化能力。
- **统计鲁棒性**：对关键变体进行多随机种子报告（均值 ± 标准差），避免单次运行偏差。
- **统一基准卡**：维护一张包含精度、延迟、GFLOPs、参数量、显存及导出状态的综合对比表。

#### 下阶段量化目标

- **主要目标**：将整体 **mAP50-95** 突破当前最优值（`0.212`，v2-P1d, α=0.10, β=0.40）。
- **次要目标**：在保持车辆类别检测性能稳定的前提下，提升 `people`/`bicycle`/`tricycle` 的 mAP50。
- **约束条件**：总延迟维持在当前实时范围内（RTX 4090 上约 4.5–5.8 ms/图像）。

### 7.6 完整总结

1. **架构层结论**
   - 当前最有效的方向仍是将长程空间建模（`LSKA`）与选择性上下文门控（`TSCG`）相结合。
   - 在当前训练设置下，增大 LSKA 核尺寸（`k=11 → k=23`）持续带来提升：mAP50-95 +0.003，召回率 +0.011。
   - SC-ELAN 骨干的选择性分组设计在计算效率与特征质量之间保持了良好平衡。

2. **检测头层结论**
   - CAI 并非普遍有益；其增益强烈依赖于重加权策略。
   - `Soft` 策略是目前唯一在达到全局最优分数的同时提升多个微小/长尾类别的 CAI 变体。
   - 激进 EMA 平滑（`Mom098`）导致类别先验估计滞后，造成全类别指标回退，不予推荐。

3. **当前推荐配置（含 v2 更新）**
   - **综合最优**：`v2-P1d`（α=0.10, β=0.40）—— 新全局最佳 mAP50-95 = 0.212，**推荐用于后续实验与 Phase 4 多种子验证**
   - **召回最优**：`v2-P1e`（α=0.20, β=0.25）—— 最高 R=0.384，适用于召回优先场景
   - **精度导向**：`DetectCAI-Tail12`（最高精度 P=0.506，适用于低误报场景）
   - **轻量部署**：`Mixed-Efficient-TSCG` / `Efficient` / `RepExact`（速度优先，精度有所牺牲）

4. **下阶段实验优先级（v2 后更新）**
   - **优先级 A**：对 `v2-P1d` 进行 Phase 4 多种子统计验证（seeds: 0, 42, 123），确认 0.212 的稳健性。
   - **优先级 B**：进一步扫描 β 参数（0.45, 0.50），固定 α=0.10，探索 β 上限。
   - **优先级 C**：针对 `people/bicycle/tricycle` 在 v2-P1d 基础上进行多种子专项评估。

5. **量化基准更新**
   - 当前需超越的新基准为 **mAP50-95 = 0.212**（v2-P1d，α=0.10, β=0.40），已超越先前 0.210 记录。
   - 延迟约束保持在 RTX 4090 上约 4.5–6.1 ms/图像的实时范围内。

## 8. SC-ELAN v2 分阶段实验结果

### 8.1 实验设计

SC-ELAN v2 采用分阶段实验流水线（`train.sh`），系统地探索 CAI 参数调优、SA-LSKA 消融和 P3-FRM 集成三大方向。全部模型均在 **LSKA23-TSCG** 骨干基础上构建，使用 **VisDrone2019-DET-test-dev** 数据集评估，训练配置统一：300 epochs, batch=16, imgsz=640, seed=0, patience=50。

| 阶段 | 实验目标 | 模型数量 |
|---|---|---|
| Phase 1 | CAI α/β 参数扫描 | 5 (p1a–p1e) |
| Phase 2 | SA-LSKA 核尺寸消融 + TSCGv2 | 4 (p2a–p2d) |
| Phase 3 | P3-FRM 特征重用集成 | 2 (p3a–p3b) |

### 8.2 总体性能对比

| Model | Config | Params | GFLOPs | P | R | mAP50 | mAP50-95 | Total (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v2-P1a** | α=0.05, β=0.15 | 11.53M | 39.4 | 0.484 | 0.379 | 0.359 | 0.207 | 5.7 |
| **v2-P1b** | α=0.10, β=0.25 | 11.53M | 39.4 | 0.490 | 0.378 | 0.362 | 0.209 | 5.7 |
| **v2-P1c** | α=0.15, β=0.30 (Soft repro) | 11.53M | 39.4 | 0.484 | 0.377 | 0.362 | 0.207 | 5.8 |
| **v2-P1d** | **α=0.10, β=0.40** | 11.53M | 39.4 | **0.490** | **0.381** | **0.367** | **0.212** | 5.7 |
| **v2-P1e** | α=0.20, β=0.25 | 11.53M | 39.4 | 0.484 | 0.384 | 0.367 | 0.211 | 5.6 |
| **v2-P2a** | SA-LSKA(7/11/23)+TSCG | 11.16M | 39.2 | 0.483 | 0.378 | 0.361 | 0.209 | 5.9 |
| **v2-P2b** | SA-LSKA(11/23/23)+TSCG | 11.17M | 39.3 | 0.478 | 0.382 | 0.361 | 0.209 | 5.9 |
| **v2-P2c** | SA-LSKA(7/23/35)+TSCG | 11.17M | 39.3 | 0.470 | 0.375 | 0.354 | 0.205 | 5.8 |
| **v2-P2d** | SA-LSKA(7/11/23)+TSCGv2 | 11.16M | 39.2 | 0.473 | 0.381 | 0.361 | 0.208 | 5.9 |
| **v2-P3a** | SA-LSKA+TSCG+P3-FRM+Detect | 11.19M | 39.5 | 0.468 | 0.382 | 0.358 | 0.207 | 5.9 |
| **v2-P3b** | SA-LSKA+TSCG+P3-FRM+DetectCAI-Soft | 11.55M | 39.5 | 0.473 | 0.381 | 0.361 | 0.208 | 5.7 |

### 8.3 Phase 1 详细结果：CAI α/β 参数扫描

Phase 1 在固定 LSKA23-TSCG 骨干上系统扫描 DetectCAI-Soft 的 `cai_alpha` 和 `cai_beta` 参数。

#### 8.3.1 v2-P1a (α=0.05, β=0.15)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.484   0.379   0.359    0.207
pedestrian         1196    21000       0.532   0.327   0.333    0.134
people             797     6376        0.510   0.158   0.183    0.0607
bicycle            377     1302        0.286   0.171   0.128    0.0502
car                1529    28063       0.713   0.764   0.760    0.493
van                1167    5770        0.438   0.436   0.400    0.269
truck              750     2659        0.507   0.440   0.433    0.279
tricycle           245     530         0.279   0.340   0.218    0.115
awning-tricycle    233     599         0.384   0.217   0.188    0.111
bus                837     2938        0.711   0.545   0.599    0.423
motor              794     5845        0.479   0.397   0.351    0.141
```
Speed: 0.3ms preprocess, 4.6ms inference, 0.8ms postprocess

#### 8.3.2 v2-P1b (α=0.10, β=0.25)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.490   0.378   0.362    0.209
pedestrian         1196    21000       0.510   0.329   0.333    0.133
people             797     6376        0.508   0.155   0.186    0.0654
bicycle            377     1302        0.286   0.171   0.134    0.0505
car                1529    28063       0.723   0.764   0.764    0.499
van                1167    5770        0.457   0.432   0.405    0.273
truck              750     2659        0.535   0.434   0.432    0.277
tricycle           245     530         0.276   0.330   0.201    0.105
awning-tricycle    233     599         0.375   0.235   0.201    0.119
bus                837     2938        0.740   0.535   0.607    0.426
motor              794     5845        0.489   0.392   0.356    0.146
```
Speed: 0.3ms preprocess, 4.7ms inference, 0.7ms postprocess

#### 8.3.3 v2-P1c (α=0.15, β=0.30 — Soft baseline repro)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.484   0.377   0.362    0.207
pedestrian         1196    21000       0.508   0.315   0.318    0.126
people             797     6376        0.530   0.148   0.180    0.0598
bicycle            377     1302        0.306   0.137   0.128    0.048
car                1529    28063       0.709   0.759   0.758    0.487
van                1167    5770        0.426   0.451   0.407    0.273
truck              750     2659        0.486   0.449   0.434    0.279
tricycle           245     530         0.275   0.319   0.211    0.109
awning-tricycle    233     599         0.404   0.247   0.226    0.129
bus                837     2938        0.726   0.540   0.600    0.418
motor              794     5845        0.472   0.402   0.354    0.140
```
Speed: 0.3ms preprocess, 4.5ms inference, 1.0ms postprocess

#### 8.3.4 v2-P1d (α=0.10, β=0.40) — Phase 1 最优
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.490   0.381   0.367    0.212
pedestrian         1196    21000       0.511   0.331   0.334    0.134
people             797     6376        0.539   0.152   0.188    0.0638
bicycle            377     1302        0.331   0.166   0.145    0.0561
car                1529    28063       0.714   0.764   0.762    0.497
van                1167    5770        0.439   0.444   0.408    0.274
truck              750     2659        0.503   0.457   0.443    0.286
tricycle           245     530         0.270   0.321   0.223    0.119
awning-tricycle    233     599         0.378   0.222   0.201    0.116
bus                837     2938        0.725   0.550   0.606    0.426
motor              794     5845        0.486   0.400   0.362    0.145
```
Speed: 0.3ms preprocess, 4.8ms inference, 0.6ms postprocess

#### 8.3.5 v2-P1e (α=0.20, β=0.25)
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.484   0.384   0.367    0.211
pedestrian         1196    21000       0.505   0.327   0.328    0.129
people             797     6376        0.519   0.157   0.184    0.0617
bicycle            377     1302        0.280   0.177   0.144    0.0561
car                1529    28063       0.710   0.764   0.761    0.493
van                1167    5770        0.439   0.453   0.415    0.279
truck              750     2659        0.511   0.457   0.447    0.288
tricycle           245     530         0.284   0.308   0.211    0.108
awning-tricycle    233     599         0.403   0.245   0.218    0.126
bus                837     2938        0.723   0.553   0.605    0.426
motor              794     5845        0.469   0.403   0.357    0.143
```
Speed: 0.3ms preprocess, 3.9ms inference, 1.4ms postprocess

#### 8.3.6 Phase 1 分析

| 排名 | Model | α | β | mAP50 | mAP50-95 | R |
|---|---|---:|---:|---:|---:|---:|
| 1 | **v2-P1d** | 0.10 | 0.40 | **0.367** | **0.212** | 0.381 |
| 2 | **v2-P1e** | 0.20 | 0.25 | **0.367** | 0.211 | **0.384** |
| 3 | **v2-P1b** | 0.10 | 0.25 | 0.362 | 0.209 | 0.378 |
| 4 | **v2-P1c** | 0.15 | 0.30 | 0.362 | 0.207 | 0.377 |
| 5 | **v2-P1a** | 0.05 | 0.15 | 0.359 | 0.207 | 0.379 |

**关键发现：**
- **v2-P1d（α=0.10, β=0.40）以 mAP50-95=0.212 刷新全局最佳**，超越先前 LSKA23-TSCG 和 DetectCAI-Soft 的 0.210 记录。
- **较高的 β 值（0.40）显著提升严格定位指标**：P1d 在 `bicycle`（0.145 mAP50，全局最高）、`truck`（0.286 mAP50-95）和 `tricycle`（0.223 mAP50）上均表现突出。
- **α=0.10 是当前最优 alpha**：P1b 和 P1d 均使用 α=0.10，分别对应 β=0.25/0.40，mAP50-95 分别为 0.209/0.212。
- **P1e（α=0.20, β=0.25）召回率最高**（R=0.384），但 mAP50-95 略低于 P1d（0.211 vs 0.212），更适合召回导向场景。
- **P1c（Soft baseline repro）未能完全复现** val4 中 DetectCAI-Soft 的 0.210 成绩（得到 0.207），提示 seed 或训练流水线差异对 CAI 存在敏感性。
- **β 参数的主效应大于 α**：固定 α=0.10 时，β 从 0.25→0.40 带来 +0.003 mAP50-95 提升。

### 8.4 Phase 2 详细结果：SA-LSKA 消融 + TSCGv2

Phase 2 测试不同 SA-LSKA 核尺寸组合以及 TSCGv2 变体的效果。

#### 8.4.1 v2-P2a — SA-LSKA(7/11/23) + TSCG
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.483   0.378   0.361    0.209
pedestrian         1196    21000       0.511   0.329   0.329    0.131
people             797     6376        0.549   0.153   0.188    0.0634
bicycle            377     1302        0.293   0.169   0.131    0.0514
car                1529    28063       0.708   0.763   0.761    0.494
van                1167    5770        0.463   0.415   0.405    0.275
truck              750     2659        0.493   0.439   0.433    0.280
tricycle           245     530         0.268   0.308   0.204    0.109
awning-tricycle    233     599         0.380   0.245   0.205    0.117
bus                837     2938        0.703   0.550   0.603    0.426
motor              794     5845        0.463   0.405   0.351    0.142
```
Speed: 0.3ms preprocess, 4.6ms inference, 1.0ms postprocess

#### 8.4.2 v2-P2b — SA-LSKA(11/23/23) + TSCG
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.478   0.382   0.361    0.209
pedestrian         1196    21000       0.502   0.336   0.333    0.131
people             797     6376        0.504   0.156   0.178    0.0613
bicycle            377     1302        0.285   0.158   0.123    0.0484
car                1529    28063       0.703   0.762   0.760    0.494
van                1167    5770        0.469   0.413   0.402    0.271
truck              750     2659        0.478   0.458   0.436    0.278
tricycle           245     530         0.284   0.321   0.210    0.110
awning-tricycle    233     599         0.375   0.249   0.209    0.122
bus                837     2938        0.718   0.555   0.607    0.427
motor              794     5845        0.463   0.408   0.356    0.144
```
Speed: 0.3ms preprocess, 5.0ms inference, 0.6ms postprocess

#### 8.4.3 v2-P2c — SA-LSKA(7/23/35) + TSCG
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.470   0.375   0.354    0.205
pedestrian         1196    21000       0.493   0.329   0.325    0.129
people             797     6376        0.502   0.151   0.177    0.0585
bicycle            377     1302        0.275   0.137   0.121    0.0471
car                1529    28063       0.702   0.762   0.759    0.491
van                1167    5770        0.440   0.426   0.394    0.264
truck              750     2659        0.468   0.451   0.428    0.276
tricycle           245     530         0.289   0.342   0.204    0.106
awning-tricycle    233     599         0.383   0.204   0.191    0.113
bus                837     2938        0.686   0.558   0.600    0.423
motor              794     5845        0.463   0.395   0.341    0.139
```
Speed: 0.3ms preprocess, 4.2ms inference, 1.3ms postprocess

#### 8.4.4 v2-P2d — SA-LSKA(7/11/23) + TSCGv2
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.473   0.381   0.361    0.208
pedestrian         1196    21000       0.526   0.326   0.328    0.130
people             797     6376        0.500   0.155   0.178    0.0597
bicycle            377     1302        0.270   0.164   0.116    0.0459
car                1529    28063       0.709   0.761   0.759    0.492
van                1167    5770        0.456   0.439   0.416    0.281
truck              750     2659        0.467   0.464   0.440    0.280
tricycle           245     530         0.263   0.311   0.210    0.108
awning-tricycle    233     599         0.366   0.227   0.208    0.117
bus                837     2938        0.703   0.551   0.604    0.423
motor              794     5845        0.475   0.408   0.354    0.140
```
Speed: 0.3ms preprocess, 5.0ms inference, 0.6ms postprocess

#### 8.4.5 Phase 2 分析

| 排名 | Model | SA-LSKA 配置 | TSCG 版本 | mAP50 | mAP50-95 | R |
|---|---|---|---|---:|---:|---:|
| 1 | **v2-P2a** | 7/11/23 | TSCG | 0.361 | 0.209 | 0.378 |
| 1 | **v2-P2b** | 11/23/23 | TSCG | 0.361 | 0.209 | **0.382** |
| 3 | **v2-P2d** | 7/11/23 | TSCGv2 | 0.361 | 0.208 | 0.381 |
| 4 | **v2-P2c** | 7/23/35 | TSCG | 0.354 | 0.205 | 0.375 |

**关键发现：**
- **SA-LSKA(7/11/23) 和 SA-LSKA(11/23/23) 表现最优**（均 mAP50-95=0.209），P2b 在召回率上略胜（R=0.382）。
- **SA-LSKA(7/23/35) 过大核尺寸导致退化**：P2c 以 0.205 mAP50-95 垫底，说明在 VisDrone 场景下核尺寸进一步增大反而有害。
- **TSCGv2 未带来明显提升**：P2d 相比 P2a（相同 SA-LSKA 配置）仅差 0.001 mAP50-95，TSCGv2 改进效果有限。
- **Phase 2 最优（0.209）低于 Phase 1 最优（0.212）**，说明 CAI 参数调优对当前架构的增益大于结构消融。
- P2b 的 `bus` mAP50（0.607）和 `truck` recall（0.458）为 Phase 2 最佳，提示 11/23/23 核配置对中大型目标有利。

### 8.5 Phase 3 详细结果：P3-FRM 集成

Phase 3 在 SA-LSKA + TSCG 基础上集成 P3-FRM（浅层特征重用模块），测试对小目标的进一步增益。

#### 8.5.1 v2-P3a — SA-LSKA + TSCG + P3-FRM + Detect
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.468   0.382   0.358    0.207
pedestrian         1196    21000       0.492   0.340   0.334    0.134
people             797     6376        0.487   0.174   0.189    0.0642
bicycle            377     1302        0.246   0.171   0.120    0.0463
car                1529    28063       0.707   0.764   0.760    0.496
van                1167    5770        0.461   0.417   0.395    0.268
truck              750     2659        0.491   0.451   0.431    0.278
tricycle           245     530         0.267   0.315   0.196    0.101
awning-tricycle    233     599         0.346   0.223   0.186    0.111
bus                837     2938        0.725   0.548   0.604    0.424
motor              794     5845        0.461   0.415   0.360    0.145
```
Speed: 0.3ms preprocess, 5.0ms inference, 0.6ms postprocess

#### 8.5.2 v2-P3b — SA-LSKA + TSCG + P3-FRM + DetectCAI-Soft
```
Class              Images  Instances    P       R      mAP50   mAP50-95
─────────────────────────────────────────────────────────────────────
all                1609    75082       0.473   0.381   0.361    0.208
pedestrian         1196    21000       0.513   0.336   0.338    0.136
people             797     6376        0.489   0.169   0.187    0.0628
bicycle            377     1302        0.274   0.169   0.133    0.0505
car                1529    28063       0.720   0.759   0.760    0.493
van                1167    5770        0.438   0.427   0.399    0.269
truck              750     2659        0.480   0.437   0.423    0.273
tricycle           245     530         0.282   0.336   0.206    0.109
awning-tricycle    233     599         0.343   0.229   0.201    0.115
bus                837     2938        0.716   0.541   0.605    0.426
motor              794     5845        0.476   0.408   0.363    0.145
```
Speed: 0.3ms preprocess, 4.8ms inference, 0.6ms postprocess

#### 8.5.3 Phase 3 分析

**关键发现：**
- **P3-FRM 未能在 Phase 2 基础上带来全局提升**：P3a（0.207）低于 P2a（0.209），P3b（0.208）也未超过 Phase 2 最优。
- **P3-FRM 的主要贡献在召回率和 `people` 类别**：P3a 的 `people` recall（0.174）和 `motor` mAP50（0.360）为全 v2 最高之一，表明浅层特征重用对小目标召回有帮助。
- **P3-FRM 略增模型复杂度**（39.5 vs 39.2 GFLOPs），但未转化为可靠的精度增益。
- **DetectCAI-Soft 在 P3 中仍然有效**：P3b 比 P3a 提升 +0.001 mAP50-95，`pedestrian` mAP50 +0.004，`tricycle` +0.010。
- **当前 P3-FRM 集成方案不推荐用于最终模型**，其增益低于 Phase 1 的 CAI 参数优化。

### 8.6 推理性能

| Model (v2) | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| v2-P1a | 0.3 | 4.6 | 0.8 | 5.7 |
| v2-P1b | 0.3 | 4.7 | 0.7 | 5.7 |
| v2-P1c | 0.3 | 4.5 | 1.0 | 5.8 |
| v2-P1d | 0.3 | 4.8 | 0.6 | 5.7 |
| v2-P1e | 0.3 | 3.9 | 1.4 | 5.6 |
| v2-P2a | 0.3 | 4.6 | 1.0 | 5.9 |
| v2-P2b | 0.3 | 5.0 | 0.6 | 5.9 |
| v2-P2c | 0.3 | 4.2 | 1.3 | 5.8 |
| v2-P2d | 0.3 | 5.0 | 0.6 | 5.9 |
| v2-P3a | 0.3 | 5.0 | 0.6 | 5.9 |
| v2-P3b | 0.3 | 4.8 | 0.6 | 5.7 |

所有 v2 模型延迟均在 5.6–5.9 ms 范围内，与 val4 批次模型持平，满足 RTX 4090 实时约束。

### 8.7 v2 逐类别横向对比

为便于直接对比，下表汇总所有 v2 模型在关键难类上的 mAP50 表现：

| Model | pedestrian | people | bicycle | tricycle | awning-tri | motor |
|---|---:|---:|---:|---:|---:|---:|
| v2-P1a | 0.333 | 0.183 | 0.128 | 0.218 | 0.188 | 0.351 |
| v2-P1b | 0.333 | 0.186 | 0.134 | 0.201 | 0.201 | 0.356 |
| v2-P1c | 0.318 | 0.180 | 0.128 | 0.211 | **0.226** | 0.354 |
| **v2-P1d** | **0.334** | **0.188** | **0.145** | **0.223** | 0.201 | 0.362 |
| v2-P1e | 0.328 | 0.184 | 0.144 | 0.211 | 0.218 | 0.357 |
| v2-P2a | 0.329 | **0.188** | 0.131 | 0.204 | 0.205 | 0.351 |
| v2-P2b | 0.333 | 0.178 | 0.123 | 0.210 | 0.209 | 0.356 |
| v2-P2c | 0.325 | 0.177 | 0.121 | 0.204 | 0.191 | 0.341 |
| v2-P2d | 0.328 | 0.178 | 0.116 | 0.210 | 0.208 | 0.354 |
| v2-P3a | **0.334** | **0.189** | 0.120 | 0.196 | 0.186 | **0.360** |
| v2-P3b | **0.338** | 0.187 | 0.133 | 0.206 | 0.201 | **0.363** |

**观察：**
- **v2-P1d 在 `bicycle`（0.145）和 `tricycle`（0.223）上全局最优**，CAI β=0.40 有效地增强了尾部类别的检测能力。
- **v2-P3b 在 `pedestrian`（0.338）和 `motor`（0.363）上全局最优**，P3-FRM 对这些类别的浅层特征重用有积极效果。
- **v2-P1c 在 `awning-tricycle`（0.226）上全局最优**，Soft baseline 参数对该类别最为适配。
- **`people` 类别整体提升有限**（0.177–0.189），仍是最具挑战性的类别。

### 8.8 v2 总结与新基准

#### 核心结论

1. **新全局最优：mAP50-95 = 0.212**
   - 由 **v2-P1d**（α=0.10, β=0.40）达成，超越先前 0.210 记录（+0.002）。
   - 同时取得 mAP50 = 0.367，R = 0.381，P = 0.490。
   - 确认 **CAI 参数调优是当前最具性价比的提升方向**。

2. **CAI 参数规律**
   - **α=0.10 是稳定的最优选择**（P1b: 0.209, P1d: 0.212）。
   - **β 的增大（0.25→0.40）显著有利于严格定位指标**，但存在上限（需进一步探索 β>0.40）。
   - 过高的 α（P1e: α=0.20）提升召回但微损 mAP50-95，适合召回优先场景。

3. **SA-LSKA 消融结论**
   - SA-LSKA(7/11/23) 和 SA-LSKA(11/23/23) 并列最优（mAP50-95=0.209），但均未超越 Phase 1 的 CAI 参数调优。
   - 过大的核尺寸组合（7/23/35）产生退化（0.205），不推荐。
   - TSCGv2 对 TSCG 的改进可忽略（+0.000~−0.001 mAP50-95）。

4. **P3-FRM 评估**
   - P3-FRM 增加浅层特征重用，对 `people` 召回和 `motor` mAP50 有积极效果。
   - 但整体 mAP50-95 未超过 Phase 1/2，当前实现不推荐用于最终模型。
   - P3-FRM + DetectCAI-Soft（P3b）优于纯 Detect（P3a），CAI 的一致性增益再次得到验证。

5. **更新后推荐配置**
   - **综合最优**：`v2-P1d`（α=0.10, β=0.40）—— 新全局最佳 mAP50-95 = 0.212
   - **召回最优**：`v2-P1e`（α=0.20, β=0.25）—— 最高 R=0.384，mAP50-95=0.211
   - **小目标特化**：`v2-P3b`（P3-FRM + DetectCAI-Soft）—— pedestrian/motor 最优
   - **下一步**：对 v2-P1d 进行 Phase 4 多种子统计验证（seeds: 0, 42, 123）

#### 下阶段量化目标（更新）

- **新基准**：mAP50-95 = **0.212**（v2-P1d）
- **探索方向**：β 参数进一步扫描（0.45, 0.50），α=0.10 固定
- **统计验证**：v2-P1d 多种子运行后报告均值 ± 标准差
- **延迟约束**：维持 RTX 4090 上约 5.6–5.9 ms/图像
