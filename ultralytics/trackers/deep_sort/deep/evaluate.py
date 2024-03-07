# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default Deepsort tracker settings available at https://github.com/nwojke/deep_sort

import torch

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
res = scores.topk(5, dim=1)[1][:, 0]
top1correct = gl[res].eq(ql).sum().item()

print("Acc top1:{:.3f}".format(top1correct / ql.size(0)))
