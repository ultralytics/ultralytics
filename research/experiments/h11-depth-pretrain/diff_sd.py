import sys, torch
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics.nn.tasks import Pose3DModel
tgt = Pose3DModel("/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml", verbose=False)
print("pose3d layers:", [(i, type(l).__name__) for i, l in enumerate(tgt.model)][:12])
tgt = tgt.state_dict()
L = lambda k: int(k.split(".")[1])
for f in sys.argv[1:]:
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    print("==", f.split("/")[-1])
    for g, r in {"backbone 0-10": range(0, 11), "neck 11-22": range(11, 23)}.items():
        tk = [k for k in tgt if L(k) in r]
        ok = [k for k in tk if k in sd and sd[k].shape == tgt[k].shape]
        pn = sum(tgt[k].numel() for k in ok) / sum(tgt[k].numel() for k in tk)
        badl = sorted({L(k) for k in tk if k not in ok})
        print(f"   {g}: {len(ok)}/{len(tk)} tensors, {pn:.0%} params; layers not transferring: {badl}")
