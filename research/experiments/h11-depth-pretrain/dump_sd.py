import sys, torch
sys.path.insert(0, "/home/rick/ultralytics_sni")
W = "/data/shared-datasets/depth_anything/weights_all/"
for n in ("yolo26n-sni-bbp3e5-depth-CONTROL-scratch", "yolo26n-sni-bbp3e5-depth"):
    ck = torch.load(W + n + ".pt", map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"]).float()
    torch.save({"state_dict": m.state_dict(), "yaml": m.yaml}, f"/home/rick/h10/{n}.sd.pt")
    print(n, "saved", len(m.state_dict()), "tensors;", [ (i, type(l).__name__) for i, l in enumerate(m.model)][:12])
