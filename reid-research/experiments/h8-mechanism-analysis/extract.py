"""Stage 1 of h8 — one-time artifact extraction for all 5 models.

Writes:
  artifacts/extraction_manifest.json
  artifacts/market_meta.parquet
  artifacts/{model_tag}/embeddings.pt           # {"query": (Nq, D), "gallery": (Ng, D)}
  artifacts/{model_tag}/retrieval.parquet       # per-query rankings + per-query AP / r1/5/10
  artifacts/{model_tag}/feats_p4.pt             # spatial feats for CKA
  artifacts/{model_tag}/feats_p5.pt
  artifacts/{model_tag}/saliency/{image_id}.npy  # IG map per query image

Sanity gate (load-bearing):
  champion R1 ∈ [0.925, 0.928]
  solider  R1 ∈ [0.965, 0.972]

Usage:
  export H8_MARKET_ROOT=/path/to/Market-1501-v15.09.15
  export H8_CHAMPION_CKPT=...  # see models.py registry for full list
  PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/extract.py

Run on westd (1 GPU). Total runtime ~hours dominated by IG (50 steps × 3368 queries × 5 models).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import models as M
import retrieval as R
from data import iter_market_split
from saliency import integrated_gradients
from segmentation import occlusion_score


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEVICE = "cuda:0"

# Sanity-gate thresholds — assert published numbers reproduce.
SANITY = {
    "champion": (0.925, 0.928),
    "solider": (0.965, 0.972),
}


def _imagenet_normalize(img_rgb: np.ndarray, size) -> torch.Tensor:
    """uint8 RGB HxWx3 -> normalized BCHW tensor on DEVICE."""
    if isinstance(size, int):
        size = (size, size)
    img = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return t


def _solider_normalize(img_rgb: np.ndarray, size) -> torch.Tensor:
    """SOLIDER uses mean=std=[0.5,0.5,0.5]."""
    img = cv2.resize(img_rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return t


def _preprocess_for(handle: M.ModelHandle, img_rgb: np.ndarray) -> torch.Tensor:
    if handle.tag == "solider":
        return _solider_normalize(img_rgb, handle.imgsz)
    return _imagenet_normalize(img_rgb, handle.imgsz)


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_market_meta(market_root: str) -> pd.DataFrame:
    """Build market_meta.parquet: one row per image, query + gallery."""
    rows = []
    for split in ("query", "gallery"):
        for rec in iter_market_split(market_root, split):
            bgr = cv2.imread(rec.img_path)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            mean_brightness = float(bgr.mean())
            try:
                occ = occlusion_score(bgr)
            except Exception as e:
                print(f"[meta] occlusion failed for {rec.image_id}: {e}", file=sys.stderr)
                occ = float("nan")
            rows.append(
                {
                    "image_id": rec.image_id,
                    "split": split,
                    "pid": rec.pid,
                    "camid": rec.camid,
                    "img_path": rec.img_path,
                    "aspect_ratio": h / w,
                    "mean_brightness": mean_brightness,
                    "occlusion_score": occ,
                }
            )
    df = pd.DataFrame(rows)
    # pid_gallery_count: count of gallery shots per pid (excluding distractor pid=-1).
    g = df[(df["split"] == "gallery") & (df["pid"] >= 0)]
    counts = g.groupby("pid").size().rename("pid_gallery_count").reset_index()
    df = df.merge(counts, on="pid", how="left").fillna({"pid_gallery_count": 0})
    df["pid_gallery_count"] = df["pid_gallery_count"].astype(int)
    return df


def extract_embeddings_and_features(handle: M.ModelHandle, meta: pd.DataFrame, out_dir: Path):
    """Forward Market through `handle`, save embeddings + p4/p5 feature taps."""
    p4_buf, p5_buf, emb_buf = {}, {}, {}
    p4_buf["query"], p5_buf["query"], emb_buf["query"] = [], [], []
    p4_buf["gallery"], p5_buf["gallery"], emb_buf["gallery"] = [], [], []
    image_ids = {"query": [], "gallery": []}

    captured = {"p4": None, "p5": None}

    def hook_p4(_m, _i, o):
        captured["p4"] = o.detach()

    def hook_p5(_m, _i, o):
        captured["p5"] = o.detach()

    h4 = handle.taps["p4"].register_forward_hook(hook_p4)
    h5 = handle.taps["p5"].register_forward_hook(hook_p5)
    try:
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"forward:{handle.tag}"):
            rgb = _read_rgb(row["img_path"])
            x = _preprocess_for(handle, rgb)
            emb = handle.embed_fn(x).squeeze(0).cpu()
            split = row["split"]
            emb_buf[split].append(emb)
            p4 = captured["p4"]
            p5 = captured["p5"]
            p4_buf[split].append(p4.mean(dim=(-1, -2)).squeeze(0).cpu() if p4.dim() == 4 else p4.squeeze(0).cpu())
            p5_buf[split].append(p5.mean(dim=(-1, -2)).squeeze(0).cpu() if p5.dim() == 4 else p5.squeeze(0).cpu())
            image_ids[split].append(row["image_id"])
    finally:
        h4.remove()
        h5.remove()

    torch.save(
        {"query": torch.stack(emb_buf["query"]), "gallery": torch.stack(emb_buf["gallery"]),
         "query_ids": image_ids["query"], "gallery_ids": image_ids["gallery"]},
        out_dir / "embeddings.pt",
    )
    torch.save(
        {"query": torch.stack(p4_buf["query"]), "gallery": torch.stack(p4_buf["gallery"])},
        out_dir / "feats_p4.pt",
    )
    torch.save(
        {"query": torch.stack(p5_buf["query"]), "gallery": torch.stack(p5_buf["gallery"])},
        out_dir / "feats_p5.pt",
    )


def compute_retrieval(out_dir: Path, meta: pd.DataFrame) -> dict[str, float]:
    """Read embeddings.pt, run junk-filtered ranking, write retrieval.parquet, return aggregate metrics."""
    emb = torch.load(out_dir / "embeddings.pt", weights_only=False)
    qf = emb["query"].numpy().astype(np.float32)
    gf = emb["gallery"].numpy().astype(np.float32)
    qids = emb["query_ids"]
    gids = emb["gallery_ids"]
    q_meta = meta.set_index("image_id").loc[qids]
    g_meta = meta.set_index("image_id").loc[gids]
    ranks = R.rank_with_junk(
        qf, q_meta["pid"].values, q_meta["camid"].values,
        gf, g_meta["pid"].values, g_meta["camid"].values, top_k=50,
    )
    aps = R.per_query_ap(ranks, q_meta["pid"].values, g_meta["pid"].values)
    metrics = R.compute_cmc_map(ranks, q_meta["pid"].values, g_meta["pid"].values)
    rows = []
    for i, qid in enumerate(qids):
        rows.append(
            {
                "query_id": qid,
                "true_pid": int(q_meta["pid"].iloc[i]),
                "true_camid": int(q_meta["camid"].iloc[i]),
                "top50_gallery_ids": [gids[j] for j in ranks.top_gids[i]],
                "top50_distances": ranks.top_dists[i].tolist(),
                "top50_pids": g_meta["pid"].values[ranks.top_gids[i]].tolist(),
                "top50_camids": g_meta["camid"].values[ranks.top_gids[i]].tolist(),
                "r1": int(ranks.matches[i, 0]),
                "r5": int(ranks.matches[i, :5].any()),
                "r10": int(ranks.matches[i, :10].any()),
                "mAP_q": float(aps[i]),
            }
        )
    pd.DataFrame(rows).to_parquet(out_dir / "retrieval.parquet")
    return metrics


def _resolve_next_after_tap(handle: M.ModelHandle):
    """Return the module that consumes the P5 feature tensor (where we'll inject)."""
    entry = M.MODEL_REGISTRY[handle.tag]
    if entry["kind"] in {"yolo_reid", "yolo_reid_mgn"}:
        return handle.model.model[-1]
    if entry["kind"] == "swin":
        return handle.model.base.norm
    raise ValueError(f"unknown kind {entry['kind']!r}")


def compute_saliency(handle: M.ModelHandle, meta: pd.DataFrame, out_dir: Path):
    """IG saliency over the P5 feature map for each query.

    Strategy: register a forward_pre_hook on the NEXT module after P5 that replaces its
    input with a scaled P5 tensor. Then `integrated_gradients` does its 50-step Riemann
    sum by varying the scale factor and accumulating gradients of cos-sim-to-true-match.

    For each query:
        1. Find the closest correct gallery match's embedding.
        2. Run one normal forward to capture the query's native P5 tensor.
        3. Define f(p5_interp) = cos_sim(embed(query | p5=p5_interp), emb_true).
        4. IG on f with baseline=zeros(p5.shape), 50 steps.
        5. Save the resulting saliency map.
    """
    sal_dir = out_dir / "saliency"
    sal_dir.mkdir(exist_ok=True, parents=True)

    emb = torch.load(out_dir / "embeddings.pt", weights_only=False)
    g_embs = emb["gallery"].to(DEVICE)
    g_ids = emb["gallery_ids"]
    q_ids = emb["query_ids"]
    g_meta = meta.set_index("image_id").loc[g_ids]
    q_meta = meta.set_index("image_id").loc[q_ids]
    g_pids = g_meta["pid"].values
    g_camids = g_meta["camid"].values

    next_module = _resolve_next_after_tap(handle)
    captured_p5 = {"val": None}

    def hook_capture(_m, _i, o):
        captured_p5["val"] = o.detach()

    h_cap = handle.taps["p5"].register_forward_hook(hook_capture)

    inject = {"feat": None}

    def hook_inject(_m, inputs):
        if inject["feat"] is not None:
            return (inject["feat"],) + inputs[1:]
        return None

    h_inj = next_module.register_forward_pre_hook(hook_inject)

    skipped = 0
    try:
        for i, qid in enumerate(tqdm(q_ids, desc=f"IG:{handle.tag}")):
            q_row = q_meta.iloc[i]
            q_pid, q_cam = int(q_row["pid"]), int(q_row["camid"])
            valid = (g_pids == q_pid) & (g_camids != q_cam)
            if not valid.any():
                np.save(sal_dir / f"{qid}.npy", np.zeros((1, 1), dtype=np.float32))
                continue
            q_emb = emb["query"][i].to(DEVICE)
            cand = g_embs[valid]
            sims = cand @ q_emb
            best_idx_in_valid = int(sims.argmax().item())
            true_g_emb = cand[best_idx_in_valid]

            rgb = _read_rgb(q_row["img_path"])
            x = _preprocess_for(handle, rgb)
            inject["feat"] = None
            _ = handle.embed_fn(x)
            native_p5 = captured_p5["val"]

            def f(p5_interp: torch.Tensor) -> torch.Tensor:
                inject["feat"] = p5_interp
                try:
                    # embed_fn wraps the forward in torch.no_grad() (correct for the
                    # batch-embedding phase). IG needs the graph, so override here.
                    with torch.enable_grad():
                        emb_q = handle.embed_fn(x).squeeze(0)
                finally:
                    inject["feat"] = None
                return torch.dot(emb_q, true_g_emb)

            try:
                attribution = integrated_gradients(
                    f, native_p5, baseline=torch.zeros_like(native_p5), steps=50
                )
            except (ValueError, RuntimeError) as e:
                skipped += 1
                if skipped > 0.01 * len(q_ids):
                    raise RuntimeError(f"too many IG failures (>1% queries): {e}")
                np.save(sal_dir / f"{qid}.npy", np.zeros((1, 1), dtype=np.float32))
                continue
            attr = attribution.squeeze(0)
            if attr.dim() == 3:
                attr_map = torch.relu(attr.sum(dim=0)).cpu().numpy().astype(np.float32)
            else:
                attr_map = torch.relu(attr).cpu().numpy().astype(np.float32)
            np.save(sal_dir / f"{qid}.npy", attr_map)
    finally:
        h_cap.remove()
        h_inj.remove()


def main():
    market_root = os.environ.get("H8_MARKET_ROOT")
    if not market_root:
        raise EnvironmentError("set H8_MARKET_ROOT to Market-1501-v15.09.15 root")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> building market_meta.parquet (may segment 19k images)…")
    meta = extract_market_meta(market_root)
    meta.to_parquet(ARTIFACTS_DIR / "market_meta.parquet")

    manifest = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "market_root": market_root,
        "models": {},
    }

    for tag in ["champion", "solider", "mgn-t3", "mgn-t4", "t5fix"]:
        out = ARTIFACTS_DIR / tag
        out.mkdir(exist_ok=True, parents=True)
        print(f"\n>>> extracting {tag}")
        handle = M.load_model(tag, device=DEVICE)
        extract_embeddings_and_features(handle, meta, out)
        metrics = compute_retrieval(out, meta)
        print(f"    {tag}: R1={metrics['r1']:.4f} mAP={metrics['mAP']:.4f}")
        manifest["models"][tag] = metrics

        if tag in SANITY:
            lo, hi = SANITY[tag]
            if not (lo <= metrics["r1"] <= hi):
                raise SystemExit(
                    f"SANITY GATE FAILED for {tag}: R1={metrics['r1']:.4f} not in [{lo},{hi}]"
                )

        compute_saliency(handle, meta, out)

        del handle
        torch.cuda.empty_cache()

    with open(ARTIFACTS_DIR / "extraction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("\n>>> Stage 1 complete. Sanity gate passed.")


if __name__ == "__main__":
    main()
