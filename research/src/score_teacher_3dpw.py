# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Score SAM 3D Body itself against 3DPW ground truth, on the validator's lifting and metrics.

The teacher was run from the GT boxes, so persons pair by box. Its depths are in its own diagonal-focal
convention (H6: f = image diagonal, exactly); they are re-keyed onto the GT's `focal = 1.2 * max(w, h)` before
anything is lifted, the same rescale `refocal_labels.py` applies, or the teacher is scored in the wrong unit.

Writes an npz in the `dump_matched_pairs.py` layout, so `body_prior_probe.py` runs on the teacher unchanged.

Usage:
    python research/src/score_teacher_3dpw.py --gt /home/rick/3dpw-pose3d-r12 --teacher teacher3dpw/labels --out t.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils.pose3d import decode_z, encode_z, keypoints_to_camera, procrustes_align

ap = argparse.ArgumentParser()
ap.add_argument("--gt", type=Path, required=True)
ap.add_argument("--teacher", type=Path, required=True)
ap.add_argument("--ratio", type=float, default=1.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

G, P, F, CX, CY = [], [], [], [], []
missing = 0
for tf in sorted(a.teacher.glob("*.txt")):
    gf = a.gt / "labels" / tf.name
    w, h = Image.open(a.gt / "images" / f"{tf.stem}.jpg").size
    g = np.loadtxt(gf, ndmin=2)
    t = np.loadtxt(tf, ndmin=2) if tf.stat().st_size else np.zeros((0, g.shape[1]))
    for row in g:
        if len(t) == 0:
            missing += 1
            continue
        j = np.abs(t[:, 1:5] - row[1:5]).sum(1).argmin()
        if np.abs(t[j, 1:5] - row[1:5]).sum() > 1e-3:
            missing += 1
            continue
        tk = t[j, 5:].reshape(18, 4).copy()
        z_rel, z_root = decode_z(tk[:-1, 3], tk[-1, 3])
        k = a.ratio * max(w, h) / np.hypot(w, h)  # teacher focal (diagonal) -> declared focal
        tk[:-1, 3], tk[-1, 3] = encode_z(z_rel * k, np.float64(z_root * k))
        gk = row[5:].reshape(18, 4).copy()
        for kk in (tk, gk):
            kk[:, 0] *= w
            kk[:, 1] *= h
        G.append(gk), P.append(tk), F.append(a.ratio * max(w, h)), CX.append(w / 2), CY.append(h / 2)

g, p = torch.tensor(np.array(G)), torch.tensor(np.array(P))
f = torch.tensor(F)
vis = (g[:, :-1, 2] == 2).double()
g3 = torch.stack([keypoints_to_camera(g[i], F[i], CX[i], CY[i]) for i in range(len(g))])
p3 = torch.stack([keypoints_to_camera(p[i], F[i], CX[i], CY[i]) for i in range(len(p))])
gr, pr = g3[:, :-1] - g3[:, -1:], p3[:, :-1] - p3[:, -1:]
mp = (((pr - gr).norm(dim=-1) * 1000 * vis).sum(1) / vis.sum(1).clamp_min(1)).mean()
pa = procrustes_align(pr, gr, vis)
pam = (((pa - gr).norm(dim=-1) * 1000 * vis).sum(1) / vis.sum(1).clamp_min(1)).mean()
_, gzr = decode_z(g[:, :-1, 3], g[:, -1, 3])
_, pzr = decode_z(p[:, :-1, 3], p[:, -1, 3])
absrel = ((pzr - gzr).abs() / gzr).mean()
print(f"TEACHER 3DPW persons={len(g)} missing={missing} MPJPE={mp:.2f} PA-MPJPE={pam:.2f} AbsRel(Z)={absrel:.5f}")
np.savez(a.out, g=g.float().numpy(), p=p.float().numpy(), focal=np.array(F), cx=np.array(CX), cy=np.array(CY))
