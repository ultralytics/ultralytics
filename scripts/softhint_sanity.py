"""Softhint fusion sanity: verify beta=0 forward equals vanilla, beta!=0 differs.

Run: python scripts/softhint_sanity.py
Exits non-zero on any failure; prints a one-line PASS otherwise.
"""

from __future__ import annotations

import sys
import torch

from ultralytics.nn.tasks import YOLOAnomalyV2Model


def to_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    cfg = "ultralytics/cfg/models/v2/yolo26-anomaly-v2-softhint.yaml"
    m = YOLOAnomalyV2Model(cfg, ch=3, nc=1, verbose=False)
    m.eval()

    n_bias_params = sum(p.numel() for p in m.heatmap_bias_fusion.parameters())
    print(f"HeatmapBiasFusion params: {n_bias_params}")
    print(f"beta init: {m.heatmap_bias_fusion.beta.detach().tolist()}")

    torch.manual_seed(0)
    x = torch.randn(2, 3, 640, 640)

    # 1) Mask-off forward.
    out_off = to_tensor(m(x))

    # 2) Mask-on with beta=0 -> must equal mask-off.
    bboxes = torch.tensor([[0.30, 0.40, 0.20, 0.15], [0.65, 0.55, 0.30, 0.20]])
    batch_idx = torch.tensor([0, 1], dtype=torch.long)
    m.set_mask_input(bboxes, batch_idx)
    out_on_beta0 = to_tensor(m(x))
    diff0 = (out_off - out_on_beta0).abs().max().item()
    print(f"beta=0 max-abs-diff vs mask-off: {diff0:.3e}")
    if diff0 > 1e-5:
        print("FAIL: beta=0 forward should equal vanilla mask-off forward.")
        return 1

    # 3) Mask-on with beta=1.0 -> output must change.
    with torch.no_grad():
        m.heatmap_bias_fusion.beta.fill_(1.0)
    m.set_mask_input(bboxes, batch_idx)
    out_on_beta1 = to_tensor(m(x))
    diff1 = (out_off - out_on_beta1).abs().max().item()
    print(f"beta=1 max-abs-diff vs mask-off: {diff1:.3e}")
    if diff1 < 1e-3:
        print("FAIL: beta=1 forward should differ measurably from vanilla.")
        return 1

    print("PASS softhint sanity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
