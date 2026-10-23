"""Build H11 warm-start checkpoints: yolo26n-pose3d with only layers 0-22 copied (by name and shape) from a source.

The head (layer 23) always stays at its fresh initialisation, so every non-pose arm has the same transfer footprint
and none inherits a detect or depth head that happens to shape-match. Sources are plain state dicts or checkpoints.
"""
import sys
import torch
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics.nn.tasks import Pose3DModel

CFG = "/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml"
WD = "/data/shared-datasets/depth_anything/weights_all/"
SRC = {
    "D27_depthinit": "/home/rick/h10/yolo26n-sni-bbp3e5-depth-CONTROL-scratch.sd.pt",  # shipped YOLO27n-depth
    "D26_depthinit": WD + "baseline-yolo26n-depth.pt",
    "O_detinit": "/home/rick/weights/yolo26n.pt",
}
for name, path in SRC.items():
    torch.manual_seed(0)
    m = Pose3DModel(CFG, verbose=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else (ck.get("ema") or ck["model"]).float().state_dict()
    tgt = m.state_dict()
    take = {k: v for k, v in sd.items() if k in tgt and v.shape == tgt[k].shape and int(k.split(".")[1]) <= 22}
    m.load_state_dict(take, strict=False)
    n = sum(1 for k in tgt if int(k.split(".")[1]) <= 22)
    torch.save({"model": m, "train_args": {}}, f"/home/rick/h11/init_{name}.pt")
    print(f"{name}: copied {len(take)}/{n} tensors of layers 0-22, {sum(v.numel() for v in take.values()) / 1e6:.2f}M params")
