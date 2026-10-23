# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Linear probe: how much person depth do a checkpoint's frozen neck features already encode?

Before spending GPU on a warm-start A/B, ask the cheaper question. For each labelled person, the three neck outputs
that feed the head (P3/P4/P5) are bilinearly sampled at the root pixel and at each depth-labelled joint, and a ridge
regression is fit from those features to (a) log root depth and (b) each joint's root-relative depth. Persons are
split by image (COCO) or by sequence (3DPW) so the probe cannot memorise a scene. A checkpoint whose features give
a better linear read-out of depth is a better starting point for the depth half of pose3d; one that does not is
unlikely to win on warm start alone.

Usage:
    python research/src/feature_depth_probe.py --data 3dpw-pose3d-r12.yaml --weights a.pt b.pt [--random]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import check_det_dataset

TAPS = {"neck": (16, 19, 22), "backbone": (4, 6, 10)}  # yolo26 P3/P4/P5: head inputs, or backbone outputs


def decode_z(z_rel, z_root):
    """Inline copy of ultralytics.utils.pose3d.decode_z, so the probe also runs from checkouts without pose3d."""
    return z_rel * 4.0 - 2.0, z_root * 50.0


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--weights", nargs="*", default=[])
ap.add_argument("--random", action="store_true", help="also probe a randomly initialised yolo26n-pose3d")
ap.add_argument("--max-images", type=int, default=3000)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--taps", choices=TAPS, default="neck")
a = ap.parse_args()

data = check_det_dataset(a.data)
src = data["val"]
root = Path(data["path"])
files = [Path(f) if Path(f).is_absolute() else root / f for f in (Path(src).read_text().split() if str(src).endswith(".txt") else [])] or sorted(
    p for p in Path(src).rglob("*") if p.suffix.lower() in (".jpg", ".png")
)
files = files[:: max(1, len(files) // a.max_images)][: a.max_images]


def label_path(img):
    """Image path -> label path, the standard images/ -> labels/ swap."""
    s = str(img)
    return Path(s[: s.rfind("/images/")] + "/labels/" + s[s.rfind("/images/") + 8 :]).with_suffix(".txt")


def group_of(img):
    """Split key: the 3DPW sequence, or the image itself."""
    m = re.match(r"(.+)_\d+$", img.stem)
    return m.group(1) if "3dpw" in str(img) and m else img.stem


def models():
    """Yield (name, model) for every checkpoint to probe."""
    if a.random:
        from ultralytics.nn.tasks import Pose3DModel

        torch.manual_seed(0)
        cfg = Path(__file__).parents[2] / "ultralytics/cfg/models/26/yolo26n-pose3d.yaml"
        yield "random-init", Pose3DModel(str(cfg), verbose=False)
    for w in a.weights:
        ck = torch.load(w, map_location="cpu", weights_only=False)
        yield Path(w).stem, (ck.get("ema") or ck["model"]).float()


@torch.no_grad()
def features(model):
    """Return per-joint feature rows, root-depth and relative-depth targets, and split groups."""
    model = model.cuda().eval()
    feats = {}
    NECK = TAPS[a.taps]
    hooks = [model.model[i].register_forward_hook(lambda m, i, o, k=i: feats.__setitem__(k, o)) for i in NECK]
    lb = LetterBox((a.imgsz, a.imgsz), auto=False)
    import cv2

    R, J, Q = ({"x": [], "y": [], "g": []} for _ in range(3))
    for f in files:
        lp = label_path(f)
        if not lp.exists() or lp.stat().st_size == 0:
            continue
        im = cv2.imread(str(f))
        h0, w0 = im.shape[:2]
        x = lb(image=im)
        r = min(a.imgsz / h0, a.imgsz / w0)
        pw, ph = (a.imgsz - w0 * r) / 2, (a.imgsz - h0 * r) / 2
        t = torch.from_numpy(x[..., ::-1].transpose(2, 0, 1).copy()).float().cuda()[None] / 255
        try:
            model(t)
        except Exception:
            pass  # the head may not run outside its own branch; the neck hooks have already fired
        k = np.loadtxt(lp, ndmin=2)[:, 5:].reshape(-1, 18, 4)
        px = torch.tensor(np.stack([k[..., 0] * w0 * r + pw, k[..., 1] * h0 * r + ph], -1)).float().cuda()
        grid = (px / a.imgsz * 2 - 1).view(1, -1, 1, 2)  # (1, P*18, 1, 2), normalised for grid_sample
        v = torch.cat([F.grid_sample(feats[i].float(), grid, align_corners=False)[0, :, :, 0].T for i in NECK], 1)
        v = v.view(len(k), 18, -1).cpu().numpy()
        z_rel, z_root = decode_z(k[:, :-1, 3], k[:, -1, 3])
        for p in range(len(k)):
            if k[p, -1, 2] == 2 and z_root[p] > 0.3:
                R["x"].append(v[p, -1]), R["y"].append(np.log(z_root[p])), R["g"].append(group_of(f))
            for j in np.nonzero(k[p, :-1, 2] == 2)[0]:
                J["x"].append(v[p, j] - v[p, -1]), J["y"].append(z_rel[p, j]), J["g"].append(group_of(f))
                if z_root[p] > 0.3 and z_root[p] + z_rel[p, j] > 0.1:  # scale-free: log depth ratio to the root
                    Q["x"].append(v[p, j] - v[p, -1]), Q["g"].append(group_of(f))
                    Q["y"].append(np.log(z_root[p] + z_rel[p, j]) - np.log(z_root[p]))
    for h in hooks:
        h.remove()
    model.cpu()
    return R, J, Q


def ridge_eval(d, log_target):
    """Fit ridge on half the groups, score on the other half; alpha chosen on a split of the training half."""
    x, y, g = np.asarray(d["x"], np.float64), np.asarray(d["y"]), np.asarray(d["g"])
    ug = np.unique(g)
    rng = np.random.default_rng(0)
    test_g = set(rng.permutation(ug)[: len(ug) // 2])
    te = np.array([gg in test_g for gg in g])
    mu, sd = x[~te].mean(0), x[~te].std(0) + 1e-6
    x = (x - mu) / sd
    xtr, ytr, xte, yte = x[~te], y[~te], x[te], y[te]

    def fit(xa, ya, lam):
        xa1 = np.c_[xa, np.ones(len(xa))]
        reg = lam * np.eye(xa1.shape[1])
        reg[-1, -1] = 0
        return np.linalg.solve(xa1.T @ xa1 + reg, xa1.T @ ya)

    n = len(xtr)
    cut = int(n * 0.8)
    best = min((np.mean((np.c_[xtr[cut:], np.ones(n - cut)] @ fit(xtr[:cut], ytr[:cut], lam) - ytr[cut:]) ** 2), lam)
               for lam in (1, 10, 100, 1000, 10000))[1]
    pred = np.c_[xte, np.ones(len(xte))] @ fit(xtr, ytr, best)
    if log_target:
        zt, zp = np.exp(yte), np.exp(pred)
        return dict(absrel=round(float(np.mean(np.abs(zp - zt) / zt)), 4), n_test=int(len(yte)))
    base = float(np.mean(np.abs(yte - np.median(ytr))))  # predicting the training median
    mae = float(np.mean(np.abs(pred - yte)))
    return dict(mae_mm=round(mae * 1000, 1), median_baseline_mm=round(base * 1000, 1),
                d1=round(float(np.mean(np.abs(pred - yte) < 0.1)), 4), n_test=int(len(yte)))


for name, m in models():
    R, J, Q = features(m)
    q = ridge_eval(Q, False)
    tag = f"{Path(a.data).stem} {a.taps}@{a.imgsz} {name}"
    print(f"PROBE {tag}: root {ridge_eval(R, True)}  joint-dz {ridge_eval(J, False)}  "
          f"ratio-dz mae {q['mae_mm'] / 1000:.4f} (median {q['median_baseline_mm'] / 1000:.4f})", flush=True)
