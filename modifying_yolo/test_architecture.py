import torch
import torch.nn as nn
from typing import Tuple, List

# =================================================================================
# SELF-CONTAINED MODULE DEFINITIONS
# All the necessary building blocks from your files are included here.
# =================================================================================

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with Batch Normalization and SiLU activation."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedBottleneck(nn.Module):
    """An enhanced bottleneck block for the EnhancedC2f module."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, use_se: bool = True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.se = SEModule(c2) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        out = self.se(out)
        return x + out if self.add else out

class EnhancedC2f(nn.Module):
    """The corrected EnhancedC2f module."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(EnhancedBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Concat(nn.Module):
    """Concatenate a list of tensors along a specified dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, self.d)

class Detect(nn.Module):
    """A dummy Detect head to complete the model structure."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.m = nn.ModuleList(nn.Conv2d(x, self.no, 1) for x in ch) # Simplified for debugging

    def forward(self, x):
        return [self.m[i](x[i]) for i in range(self.nl)]


# =================================================================================
# STANDALONE MODEL ARCHITECTURE
# This class hardcodes the architecture from your YAML file with explicit connections.
# =================================================================================
class EnhancedYOLOv8(nn.Module):
    def __init__(self, nc=80):
        super().__init__()

        width_mult = 0.25 # For 'n' model from YAML
        def scale(c): return int(c * width_mult)

        # ----------------- Backbone -----------------
        self.layer_0 = Conv(3, scale(64), 3, 2)
        self.layer_1 = Conv(scale(64), scale(128), 3, 2)
        self.layer_2 = EnhancedC2f(scale(128), scale(128), n=3, shortcut=True)
        self.layer_3 = Conv(scale(128), scale(256), 3, 2)
        self.layer_4 = EnhancedC2f(scale(256), scale(256), n=6, shortcut=True)
        self.layer_5 = Conv(scale(256), scale(512), 3, 2)
        self.layer_6 = EnhancedC2f(scale(512), scale(512), n=6, shortcut=True)
        self.layer_7 = Conv(scale(512), scale(1024), 3, 2)
        self.layer_8 = EnhancedC2f(scale(1024), scale(1024), n=3, shortcut=True)
        self.layer_9 = SPPF(scale(1024), scale(1024), 5)

        # ----------------- Head -----------------
        self.layer_10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_11 = Concat(1)
        self.layer_12 = EnhancedC2f(scale(1024) + scale(512), scale(512), n=3)
        self.layer_13 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_14 = Concat(1)
        self.layer_15 = EnhancedC2f(scale(512) + scale(256), scale(256), n=3)
        self.layer_16 = Conv(scale(256), scale(256), 3, 2)
        self.layer_17 = Concat(1)
        self.layer_18 = EnhancedC2f(scale(256) + scale(512), scale(512), n=3)
        self.layer_19 = Conv(scale(512), scale(512), 3, 2)
        self.layer_20 = Concat(1)
        self.layer_21 = EnhancedC2f(scale(512) + scale(1024), scale(1024), n=3)
        self.layer_22 = Detect(nc, ch=(scale(256), scale(512), scale(1024)))

    def forward(self, x):
        """Forward pass with debug prints at each layer."""
        outputs = {}
        print(f"Input shape: {x.shape}\n")

        # --- Backbone ---
        outputs[0] = self.layer_0(x)
        print(f"Layer  0 (Conv): {outputs[0].shape}")
        outputs[1] = self.layer_1(outputs[0])
        print(f"Layer  1 (Conv): {outputs[1].shape}")
        outputs[2] = self.layer_2(outputs[1])
        print(f"Layer  2 (EnhancedC2f): {outputs[2].shape}")
        outputs[3] = self.layer_3(outputs[2])
        print(f"Layer  3 (Conv): {outputs[3].shape}")
        outputs[4] = self.layer_4(outputs[3])
        print(f"Layer  4 (EnhancedC2f): {outputs[4].shape}")
        outputs[5] = self.layer_5(outputs[4])
        print(f"Layer  5 (Conv): {outputs[5].shape}")
        outputs[6] = self.layer_6(outputs[5])
        print(f"Layer  6 (EnhancedC2f): {outputs[6].shape}")
        outputs[7] = self.layer_7(outputs[6])
        print(f"Layer  7 (Conv): {outputs[7].shape}")
        outputs[8] = self.layer_8(outputs[7])
        print(f"Layer  8 (EnhancedC2f): {outputs[8].shape}")
        outputs[9] = self.layer_9(outputs[8])
        print(f"Layer  9 (SPPF): {outputs[9].shape}\n" + "-"*30)

        # --- Head ---
        y = self.layer_10(outputs[9])
        print(f"Layer 10 (Upsample): {y.shape}")
        y = self.layer_11([y, outputs[6]])
        print(f"Layer 11 (Concat): {y.shape} <- from Layer 10 ({outputs[10].shape if 10 in outputs else 'N/A'}) & 6 ({outputs[6].shape})")
        y = self.layer_12(y)
        outputs[12] = y
        print(f"Layer 12 (EnhancedC2f): {y.shape}")

        y2 = self.layer_13(y)
        print(f"Layer 13 (Upsample): {y2.shape}")
        y2 = self.layer_14([y2, outputs[4]])
        print(f"Layer 14 (Concat): {y2.shape} <- from Layer 13 & 4 ({outputs[4].shape})")
        outputs[15] = self.layer_15(y2)
        print(f"Layer 15 (EnhancedC2f): {outputs[15].shape} -> P3 out")

        y3 = self.layer_16(outputs[15])
        print(f"Layer 16 (Conv): {y3.shape}")
        y3 = self.layer_17([y3, outputs[12]])
        print(f"Layer 17 (Concat): {y3.shape} <- from Layer 16 & 12 ({outputs[12].shape})")
        outputs[18] = self.layer_18(y3)
        print(f"Layer 18 (EnhancedC2f): {outputs[18].shape} -> P4 out")

        y4 = self.layer_19(outputs[18])
        print(f"Layer 19 (Conv): {y4.shape}")
        y4 = self.layer_20([y4, outputs[9]])
        print(f"Layer 20 (Concat): {y4.shape} <- from Layer 19 & 9 ({outputs[9].shape})")
        outputs[21] = self.layer_21(y4)
        print(f"Layer 21 (EnhancedC2f): {outputs[21].shape} -> P5 out\n" + "-"*30)

        # --- Detection Head ---
        detect_inputs = [outputs[15], outputs[18], outputs[21]]
        final_output = self.layer_22(detect_inputs)
        print("Layer 22 (Detect):")
        for i, out_tensor in enumerate(final_output):
            print(f"  - Branch {i+1} output shape: {out_tensor.shape}")
        return final_output


# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == "__main__":
    # Create a dummy input tensor that matches a typical image size
    dummy_input = torch.randn(1, 3, 640, 640)

    print("Building model...\n")
    model = EnhancedYOLOv8(nc=80)

    print("Starting forward pass with debug prints...\n")
    try:
        model(dummy_input)
        print("\n✅ Forward pass completed successfully!")
    except Exception as e:
        print(f"\n❌ ERROR CAUGHT!")
        import traceback
        traceback.print_exc()
        print("\nCheck the last printed layer shape before this error to find the mismatch.")

