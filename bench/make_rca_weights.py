"""Save zero-bias and nonzero-bias random-init checkpoints for the three ratio architectures.

The seed is reset before each build so the two arms of one architecture share identical non-bias weights and differ
only in the biases that random init leaves at exactly zero. That is what makes the latency difference attributable.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/autodl-tmp/code/ultravit-lane-b")

from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_num_params  # noqa: E402

ARCHS = {
    "dinov3splus": "deim_dinov3splus_sta_l6_xl.yaml",
    "ffnattn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml",
    "attn2": "yolo26x-ultravit-attn2-deim_mal_deimv2Neck.yaml",
}
OUT = Path("/root/autodl-tmp/data/rca-weights")
OUT.mkdir(parents=True, exist_ok=True)

for tag, yaml_name in ARCHS.items():
    for arm, fill in (("zerobias", None), ("nonzerobias", 1e-3)):
        path = OUT / f"{tag}_{arm}.pt"
        if path.exists():
            print(f"  skip {path.name}", flush=True)
            continue
        torch.manual_seed(0)
        model = RTDETRDetectionModel(yaml_name, nc=80, verbose=False)
        touched = 0
        if fill is not None:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name.endswith(".bias") and torch.count_nonzero(p) == 0:
                        p.add_(torch.full_like(p, fill))
                        touched += 1
        state = model.state_dict()
        zero = sum(1 for k, v in state.items() if k.endswith(".bias") and v.numel() and not torch.count_nonzero(v))
        model.yaml["yaml_file"] = yaml_name
        model.args = {"model": yaml_name}
        torch.save({"model": model.half().eval(), "date": "2026-07-27", "version": "rca", "epoch": -1}, path)
        print(
            f"  {path.name}: params={get_num_params(model) / 1e6:.2f}M tensors={len(state)} "
            f"made_nonzero={touched} remaining_zero_bias={zero}",
            flush=True,
        )
print("=== weights done", flush=True)
