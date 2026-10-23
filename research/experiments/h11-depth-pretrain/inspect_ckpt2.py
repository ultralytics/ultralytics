import sys, torch, pickle
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics.nn.tasks import Pose3DModel
tgt = Pose3DModel("/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml", verbose=False).state_dict()
layer = lambda k: int(k.split(".")[1])
groups = {"backbone 0-10": range(0, 11), "neck 11-22": range(11, 23), "head 23": range(23, 24)}
for p in sys.argv[1:]:
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print("==", p.split("/")[-1], "LOAD FAIL", type(e).__name__, str(e)[:150]); continue
    m = ck.get("ema") or ck.get("model")
    sd = m.float().state_dict()
    a = ck.get("train_args", {}) or {}
    print("==", p.split("/")[-1], "| yaml", getattr(m, "yaml", {}).get("yaml_file"), "| imgsz", a.get("imgsz"), "| params %.2fM" % (sum(v.numel() for v in sd.values()) / 1e6))
    for g, r in groups.items():
        tk = [k for k in tgt if layer(k) in r]
        ok = [k for k in tk if k in sd and sd[k].shape == tgt[k].shape]
        pn = sum(tgt[k].numel() for k in ok) / max(1, sum(tgt[k].numel() for k in tk))
        bad = [k for k in tk if k in sd and sd[k].shape != tgt[k].shape][:2]
        print(f"   {g}: {len(ok)}/{len(tk)} tensors, {pn:.0%} of params" + (f"  e.g. mismatch {bad[0]} {tuple(sd[bad[0]].shape)} vs {tuple(tgt[bad[0]].shape)}" if bad else ""))
