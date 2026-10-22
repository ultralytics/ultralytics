# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""H10 — how much of the student's 3D error would a parametric body model remove?

Reads the matched pairs written by `dump_matched_pairs.py` and reports, on the validator's own lifting:

1. Decomposition: MPJPE, scale-fit MPJPE, PA-MPJPE, and the residual with GT depth or GT 2D swapped in.
2. Plausibility: per-bone length error and left/right asymmetry, predicted vs GT.
3. Refit under bone-length constraints — the part of a body model that does not need pixels: oracle (GT
   lengths), mean (population lengths), prop (population proportions at the predicted size), sym (own lengths,
   L/R averaged).
4. Anthropometric root depth: rescale the predicted root depth so the lifted skeleton has a known metric size.

Usage:
    python research/src/body_prior_probe.py F_clip_3dpw.npz [--mean-from F_clip_coco.npz] [--device cuda]
"""

import argparse
import json

import numpy as np
import torch

from ultralytics.utils.pose3d import decode_z, keypoints_to_camera, procrustes_align

ROOT = 17
# COCO-17 bones that live entirely inside the 12 body joints 3DPW annotates, plus the root spokes.
BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # upper arm, forearm  L, R
    (11, 13), (13, 15), (12, 14), (14, 16),  # thigh, shin  L, R
    (5, 6), (11, 12), (5, 11), (6, 12),  # shoulder width, hip width, torso sides
    (11, ROOT), (12, ROOT),  # hip-root spokes
]
NAMES = ["uarm_L", "farm_L", "uarm_R", "farm_R", "thigh_L", "shin_L", "thigh_R", "shin_R",
         "shoulders", "hips", "torso_L", "torso_R", "hipL_root", "hipR_root"]
PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (10, 11)]  # L/R bone index pairs


def lift(k, d):
    """Lift (N,18,4) keypoints to camera space (N,18,3) with each pair's own focal and principal point."""
    out = []
    for i in range(0, len(k), 4096):
        sl = slice(i, i + 4096)
        f, cx, cy = d["focal"][sl], d["cx"][sl], d["cy"][sl]
        z_rel, z_root = decode_z(k[sl, :-1, 3], k[sl, -1:, 3])
        z = torch.cat([z_rel + z_root, z_root], -1)
        x = (k[sl, :, 0] - cx[:, None]) * z / f[:, None]
        y = (k[sl, :, 1] - cy[:, None]) * z / f[:, None]
        out.append(torch.stack([x, y, z], -1))
    return torch.cat(out)


def rr(x):
    """Root-relative 17 body joints."""
    return x[:, :ROOT] - x[:, ROOT:]


def mpjpe(p, g, vis):
    """Per-person MPJPE in mm over visible joints."""
    e = (p - g).norm(dim=-1) * 1000
    return (e * vis).sum(1) / vis.sum(1).clamp_min(1)


def bone_len(x):
    """(N, nbones) bone lengths in metres from (N,18,3)."""
    return torch.stack([(x[:, a] - x[:, b]).norm(dim=-1) for a, b in BONES], 1)


def refit(p3d, d, target_len, bone_ok, s_px, s_z, iters=300):
    """Move the predicted joints the least they must to take the target bone lengths.

    Objective per person: 2D reprojection residual against the prediction (sigma s_px pixels), depth residual
    against the prediction (sigma s_z metres), and bone-length residual against `target_len` (sigma 5 mm).
    The root is a free joint like the others; only bones flagged in `bone_ok` are constrained.
    """
    f, cx, cy = d["focal"][:, None], d["cx"][:, None], d["cy"][:, None]
    u0 = p3d[..., 0] / p3d[..., 2] * f + cx
    v0 = p3d[..., 1] / p3d[..., 2] * f + cy
    z0 = p3d[..., 2]
    x = p3d.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.005)
    for it in range(iters):
        if it == 200:
            for gr in opt.param_groups:
                gr["lr"] = 0.001
        opt.zero_grad()
        z = x[..., 2].clamp_min(0.1)
        u = x[..., 0] / z * f + cx
        v = x[..., 1] / z * f + cy
        l2d = (((u - u0) ** 2 + (v - v0) ** 2) / s_px**2).sum(1)
        lz = ((z - z0) ** 2 / s_z**2).sum(1)
        lb = (((bone_len(x) - target_len) / 0.005) ** 2 * bone_ok).sum(1)
        (l2d + lz + lb).sum().backward()
        opt.step()
    return x.detach()


def summarize(name, vals):
    """Mean of a per-person tensor, as a float."""
    return name, round(float(vals.mean()), 2)


def main():
    """Run the probe and print one JSON line per measurement block."""
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--mean-from", default=None, help="npz whose GT supplies population bone lengths (else self)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device

    def load(path):
        z = np.load(path)
        d = {k: torch.as_tensor(z[k], dtype=torch.float64 if k != "g" and k != "p" else torch.float32, device=dev)
             for k in ("g", "p", "focal", "cx", "cy")}
        d["g"], d["p"] = d["g"].double(), d["p"].double()
        return d

    d = load(a.npz)
    has = (d["g"][:, :ROOT, 2] == 2).any(1)  # persons with at least one depth-labelled joint, as MPJPE averages
    d = {k: v[has] for k, v in d.items()}
    g, p = d["g"], d["p"]
    vis = (g[:, :ROOT, 2] == 2).double()
    g3, p3 = lift(g, d), lift(p, d)
    G, P = rr(g3), rr(p3)
    res = {"n": len(g)}

    # 1. decomposition
    m = mpjpe(P, G, vis)
    s = ((P * G).sum(-1) * vis).sum(1) / ((P * P).sum(-1) * vis).sum(1).clamp_min(1e-9)
    pa = procrustes_align(P, G, vis)
    swap_z = p.clone()
    swap_z[..., 3] = g[..., 3]
    swap_xy = p.clone()
    swap_xy[..., :2] = g[..., :2]
    swap_root = p.clone()
    swap_root[:, ROOT, 3] = g[:, ROOT, 3]
    res["decomp"] = dict([
        summarize("mpjpe", m),
        summarize("scale_fit", mpjpe(P * s[:, None, None], G, vis)),
        summarize("pa_mpjpe", mpjpe(pa, G, vis)),
        summarize("gt_depth_pred_2d", mpjpe(rr(lift(swap_z, d)), G, vis)),
        summarize("gt_2d_pred_depth", mpjpe(rr(lift(swap_xy, d)), G, vis)),
        summarize("gt_root_depth_only", mpjpe(rr(lift(swap_root, d)), G, vis)),
        ("median_scale_ratio_pred_over_gt", round(float((1 / s).median()), 4)),
    ])

    # 2. plausibility — only bones whose both ends carry depth
    jv = torch.cat([vis, torch.ones_like(vis[:, :1])], 1)
    bone_ok = torch.stack([jv[:, a] * jv[:, b] for a, b in BONES], 1)
    Lg, Lp = bone_len(g3), bone_len(p3)
    rel = ((Lp - Lg).abs() / Lg.clamp_min(1e-3))
    per_bone = {}
    for i, nm in enumerate(NAMES):
        ok = bone_ok[:, i] > 0
        if ok.sum() > 50:
            per_bone[nm] = dict(gt_m=round(float(Lg[ok, i].median()), 3), pred_m=round(float(Lp[ok, i].median()), 3),
                                med_abs_rel_err=round(float(rel[ok, i].median()), 3))
    asym = {}
    for l, r in PAIRS:
        ok = (bone_ok[:, l] * bone_ok[:, r]) > 0
        if ok.sum() > 50:
            ag = ((Lg[ok, l] - Lg[ok, r]).abs() / (Lg[ok, l] + Lg[ok, r]) * 2).median()
            ap_ = ((Lp[ok, l] - Lp[ok, r]).abs() / (Lp[ok, l] + Lp[ok, r]) * 2).median()
            asym[NAMES[l][:-2]] = dict(gt=round(float(ag), 3), pred=round(float(ap_), 3))
    res["bones"], res["asym"] = per_bone, asym

    # population lengths: median GT bone length over a population, and proportions normalised by total length
    dm = load(a.mean_from) if a.mean_from else d
    gm3 = lift(dm["g"], dm)
    vm = torch.cat([(dm["g"][:, :ROOT, 2] == 2).double(), torch.ones_like(dm["g"][:, :1, 0])], 1)
    okm = torch.stack([vm[:, x] * vm[:, y] for x, y in BONES], 1) > 0
    Lm_all = bone_len(gm3)
    Lmean = torch.stack([Lm_all[okm[:, i], i].median() for i in range(len(BONES))])
    sym_mean = Lmean.clone()
    for l, r in PAIRS:
        sym_mean[l] = sym_mean[r] = (Lmean[l] + Lmean[r]) / 2
    Lmean = sym_mean
    # proportions at the predicted size: scale the mean skeleton to the predicted total length of the same bones
    k = (Lp * bone_ok).sum(1, keepdim=True) / (Lmean[None] * bone_ok).sum(1, keepdim=True).clamp_min(1e-9)
    Lprop = Lmean[None] * k
    Lsym = Lp.clone()
    for l, r in PAIRS:
        Lsym[:, l] = Lsym[:, r] = (Lp[:, l] + Lp[:, r]) / 2

    # 3. refits: sweep the 2D/depth trust so the oracle is a genuine upper bound
    targets = {"oracle": Lg, "mean": Lmean[None].expand_as(Lg), "prop": Lprop, "sym": Lsym}
    refits = {}
    for s_px in (2.0, 5.0, 15.0):
        for s_z in (0.05, 0.2, 1.0):
            row = {}
            for nm, L in targets.items():
                x = refit(p3, d, L, bone_ok, s_px, s_z)
                X = rr(x)
                row[nm] = dict(mpjpe=round(float(mpjpe(X, G, vis).mean()), 2),
                               pa=round(float(mpjpe(procrustes_align(X, G, vis), G, vis).mean()), 2))
            refits[f"px{s_px}_z{s_z}"] = row
            print("refit", s_px, s_z, json.dumps(row), flush=True)
    res["refit"] = refits

    # 4. anthropometric root depth: rescale z_root so the lifted skeleton's size matches a known metric size
    _, zr_p = decode_z(p[:, :-1, 3], p[:, -1, 3])
    _, zr_g = decode_z(g[:, :-1, 3], g[:, -1, 3])
    size_p = (Lp * bone_ok).sum(1)
    anth = {"learned": round(float(((zr_p - zr_g).abs() / zr_g).mean()), 5)}
    for nm, L in (("oracle", Lg), ("mean", Lmean[None].expand_as(Lg))):
        zn = zr_p * (L * bone_ok).sum(1) / size_p.clamp_min(1e-3)
        anth[nm] = round(float(((zn - zr_g).abs() / zr_g).mean()), 5)
        # geometric blend with the learned estimate, the way a model with both cues would combine them
        zb = torch.sqrt(zn * zr_p)
        anth[nm + "_blend"] = round(float(((zb - zr_g).abs() / zr_g).mean()), 5)
    res["root_absrel"] = anth
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
