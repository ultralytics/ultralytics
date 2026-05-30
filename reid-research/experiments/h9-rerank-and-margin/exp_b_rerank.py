"""Experiment B: top-50 re-ranking MLP head.

End-to-end:
1. Extract champion embeddings on Market train (12936) + test (3368q + 15913g).
2. Build train candidate top-50 (champion's nearest neighbors per train query).
3. Train an MLP rerank head (q_emb, g_emb, cos) -> score, CE loss over 50 candidates.
4. Apply head to test top-50, compute new R1/mAP.

Outputs all artifacts under /root/expb/.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
sys.path.insert(0, "/root/autodl-tmp/ultralytics_reid/reid-research/experiments/h8-mechanism-analysis")
import models as M
import retrieval as R
from data import iter_market_split

DEVICE = "cuda:0"
MARKET = "/root/.cache/autoresearch/Market-1501-v15.09.15"
OUT = Path("/root/expb")
OUT.mkdir(parents=True, exist_ok=True)
os.environ["H8_CHAMPION_CKPT"] = "/root/autodl-tmp/ultralytics_reid/runs/reid/runs/reid/runs/arch31_imgsz384/weights/best.pt"


def preprocess(path, size):
    if isinstance(size, int):
        size = (size, size)
    pil = Image.open(path).convert("RGB")
    pil = T.Resize(size[0], interpolation=T.InterpolationMode.BILINEAR)(pil)
    pil = T.CenterCrop(size)(pil)
    return T.ToTensor()(pil).unsqueeze(0).to(DEVICE)


def extract_split(handle, split: str, market_root=MARKET) -> dict:
    print(f"\n>>> extracting {split}")
    feats, pids, camids, ids = [], [], [], []
    t0 = time.time()
    for i, rec in enumerate(iter_market_split(market_root, split)):
        x = preprocess(rec.img_path, handle.imgsz)
        emb = handle.embed_fn(x).squeeze(0).cpu().numpy()
        feats.append(emb)
        pids.append(rec.pid); camids.append(rec.camid); ids.append(rec.image_id)
        if (i+1) % 2000 == 0:
            print(f"  {split} {i+1} ({(i+1)/(time.time()-t0):.0f} img/s)", flush=True)
    return {
        "feats": np.stack(feats), "pids": np.array(pids), "camids": np.array(camids), "ids": ids,
    }


def extract_train_split(handle) -> dict:
    """Market train doesn't follow query/gallery layout; iterate bounding_box_train directly."""
    print(f"\n>>> extracting train (bounding_box_train)")
    feats, pids, camids, ids = [], [], [], []
    subdir = Path(MARKET) / "bounding_box_train"
    t0 = time.time()
    for i, p in enumerate(sorted(subdir.glob("*.jpg"))):
        stem = p.stem
        pid_str, rest = stem.split("_", 1)
        cam_str = rest.split("s", 1)[0]
        pid = int(pid_str)
        if pid < 0:
            continue
        camid = int(cam_str[1:])
        x = preprocess(str(p), handle.imgsz)
        emb = handle.embed_fn(x).squeeze(0).cpu().numpy()
        feats.append(emb)
        pids.append(pid); camids.append(camid); ids.append(stem)
        if (i+1) % 2000 == 0:
            print(f"  train {i+1} ({(i+1)/(time.time()-t0):.0f} img/s)", flush=True)
    return {"feats": np.stack(feats), "pids": np.array(pids), "camids": np.array(camids), "ids": ids}


def build_top_k(query, gallery, K=50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (top_gids, top_dists, top_pids), all (Nq, K), using rank_with_junk."""
    ranks = R.rank_with_junk(
        query["feats"], query["pids"], query["camids"],
        gallery["feats"], gallery["pids"], gallery["camids"], top_k=K,
    )
    top_gids = ranks.top_gids
    top_dists = ranks.top_dists
    top_pids = gallery["pids"][top_gids]
    return top_gids, top_dists, top_pids


class RerankHead(nn.Module):
    """MLP scoring head: score(q, g) = MLP([q-g, q*g, cos]).

    Operates on (B, K=50) candidates per query. Output is (B, K) logits over candidates.
    Trained with cross-entropy where target = position-of-true-pid in the top-K (skipped if absent).
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: torch.Tensor, g_set: torch.Tensor) -> torch.Tensor:
        # q: (B, D), g_set: (B, K, D); both assumed L2-normed
        K = g_set.size(1)
        q_e = q.unsqueeze(1).expand(-1, K, -1)
        cos = (q_e * g_set).sum(dim=-1, keepdim=True)  # (B, K, 1)
        feat = torch.cat([q_e - g_set, q_e * g_set, cos], dim=-1)
        return self.mlp(feat).squeeze(-1)  # (B, K)


def train_head(train_data: dict, top_gids, top_dists, top_pids, gallery: dict, epochs=20, lr=5e-4) -> RerankHead:
    Nq = top_gids.shape[0]
    K = top_gids.shape[1]
    D = train_data["feats"].shape[1]
    # Find target position-of-true-pid in top-K (junk excluded already; rank by smallest valid pos).
    # Junk = same-pid-same-cam; gallery here is also train (so query is in gallery, but junk pushes it back).
    targets = np.full(Nq, -1, dtype=np.int64)
    for i in range(Nq):
        # The first position in top-K whose pid matches AND camid != query camid is the true match.
        for k in range(K):
            if top_pids[i, k] == train_data["pids"][i]:
                # check cam — already junk-filtered, so just pid check suffices for "valid"
                targets[i] = k
                break
    valid_mask = targets >= 0
    print(f"\n>>> trainable queries: {valid_mask.sum()}/{Nq} (have true PID in top-{K})")

    q_feats = torch.from_numpy(train_data["feats"]).float()
    g_feats_all = torch.from_numpy(gallery["feats"]).float()
    targets_t = torch.from_numpy(targets).long()
    valid_t = torch.from_numpy(valid_mask).bool()

    head = RerankHead(D).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    valid_idx = torch.nonzero(valid_t).squeeze(-1).tolist()
    BS = 128

    for epoch in range(epochs):
        np.random.shuffle(valid_idx)
        head.train()
        ep_loss = ep_top1 = ep_count = 0
        for start in range(0, len(valid_idx), BS):
            batch_idx = valid_idx[start:start + BS]
            q = q_feats[batch_idx].to(DEVICE)  # (B, D)
            gids = top_gids[batch_idx]  # (B, K)
            tgt = targets_t[batch_idx].to(DEVICE)  # (B,)
            g = g_feats_all[gids.flatten()].view(len(batch_idx), K, D).to(DEVICE)  # (B, K, D)
            logits = head(q, g)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch_idx)
            preds = logits.argmax(-1)
            ep_top1 += (preds == tgt).sum().item()
            ep_count += len(batch_idx)
        sched.step()
        print(f"  epoch {epoch+1}/{epochs}  loss={ep_loss/ep_count:.4f}  train_top1={ep_top1/ep_count:.4f}")
    return head


def eval_rerank(head: RerankHead, q_test: dict, g_test: dict, K=50) -> dict:
    """Apply head to test top-K and compute re-ranked R1/R5/R10/mAP."""
    Nq = len(q_test["ids"])
    top_gids, top_dists, top_pids = build_top_k(q_test, g_test, K=K)
    D = q_test["feats"].shape[1]
    q_feats = torch.from_numpy(q_test["feats"]).float()
    g_feats_all = torch.from_numpy(g_test["feats"]).float()
    new_order_gids = np.zeros_like(top_gids)
    head.eval()
    with torch.no_grad():
        BS = 256
        for s in range(0, Nq, BS):
            e = min(s + BS, Nq)
            q = q_feats[s:e].to(DEVICE)
            gids = top_gids[s:e]
            g = g_feats_all[gids.flatten()].view(e - s, K, D).to(DEVICE)
            logits = head(q, g)  # (B, K) — higher = better match
            # Sort top-K positions by score descending, then reorder gallery ids accordingly
            order = logits.argsort(dim=-1, descending=True).cpu().numpy()
            for i in range(e - s):
                new_order_gids[s + i] = gids[i, order[i]]
    new_pids = g_test["pids"][new_order_gids]
    new_camids = g_test["camids"][new_order_gids]
    matches = (new_pids == q_test["pids"][:, None]) & (new_camids != q_test["camids"][:, None])
    has_any = np.array([np.any(g_test["pids"] == q_test["pids"][i]) & np.any(g_test["camids"] != q_test["camids"][i]) for i in range(Nq)])
    valid = max(1, has_any.sum())
    r1 = matches[:, 0].sum() / valid
    r5 = matches[:, :5].any(axis=1).sum() / valid
    r10 = matches[:, :10].any(axis=1).sum() / valid
    # AP within top-K
    aps = []
    for i in range(Nq):
        n_rel = int(((g_test["pids"] == q_test["pids"][i]) & (g_test["camids"] != q_test["camids"][i])).sum())
        if n_rel == 0:
            aps.append(0.0); continue
        hits = matches[i].astype(np.float64)
        if hits.sum() == 0:
            aps.append(0.0); continue
        cumhits = np.cumsum(hits)
        prec = cumhits / np.arange(1, K + 1)
        aps.append((prec * hits).sum() / n_rel)
    mAP = float(np.mean([a for a, ok in zip(aps, has_any) if ok]))
    return {"r1": float(r1), "r5": float(r5), "r10": float(r10), "mAP": mAP}


def main():
    t_start = time.time()
    # Load champion via h8 models registry
    print(">>> loading champion")
    handle = M.load_model("champion", device=DEVICE)
    print(f"    {sum(p.numel() for p in handle.model.parameters())/1e6:.1f}M params")

    # Extract test query + gallery + train
    cache_train = OUT / "train.npz"
    cache_q = OUT / "test_query.npz"
    cache_g = OUT / "test_gallery.npz"

    if cache_train.exists():
        print("loading cached train"); d = np.load(cache_train, allow_pickle=True); train_data = {k: d[k] for k in d}
    else:
        train_data = extract_train_split(handle)
        np.savez(cache_train, **{k: train_data[k] for k in train_data if k != "ids"}, ids=np.array(train_data["ids"]))
    if cache_q.exists():
        d = np.load(cache_q, allow_pickle=True); q_test = {k: d[k] for k in d}
    else:
        q_test = extract_split(handle, "query")
        np.savez(cache_q, **{k: q_test[k] for k in q_test if k != "ids"}, ids=np.array(q_test["ids"]))
    if cache_g.exists():
        d = np.load(cache_g, allow_pickle=True); g_test = {k: d[k] for k in d}
    else:
        g_test = extract_split(handle, "gallery")
        np.savez(cache_g, **{k: g_test[k] for k in g_test if k != "ids"}, ids=np.array(g_test["ids"]))

    # Baseline R1 sanity check
    print("\n>>> baseline (champion raw, no rerank)")
    base_ranks = R.rank_with_junk(q_test["feats"], q_test["pids"], q_test["camids"],
                                  g_test["feats"], g_test["pids"], g_test["camids"], top_k=50)
    base_metrics = R.compute_cmc_map(base_ranks, q_test["pids"], g_test["pids"])
    print(f"    baseline R1={base_metrics['r1']:.4f}  mAP={base_metrics['mAP']:.4f}")

    # Build train top-50 (train queries against train gallery)
    print("\n>>> building train top-50 candidates")
    top_gids, top_dists, top_pids = build_top_k(train_data, train_data, K=50)
    print(f"    train top-50 shape: {top_gids.shape}")

    # Train the head
    head = train_head(train_data, top_gids, top_dists, top_pids, train_data, epochs=20, lr=5e-4)
    torch.save(head.state_dict(), OUT / "rerank_head.pt")

    # Eval on test
    print("\n>>> applying re-rank head on test")
    new_metrics = eval_rerank(head, q_test, g_test, K=50)
    print(f"    re-ranked R1={new_metrics['r1']:.4f}  R5={new_metrics['r5']:.4f}  R10={new_metrics['r10']:.4f}  mAP={new_metrics['mAP']:.4f}")
    print(f"    baseline  R1={base_metrics['r1']:.4f}  delta = {new_metrics['r1']-base_metrics['r1']:+.4f}")

    with open(OUT / "result.json", "w") as f:
        json.dump({"baseline": base_metrics, "rerank": new_metrics,
                   "delta_r1": new_metrics["r1"] - base_metrics["r1"],
                   "delta_mAP": new_metrics["mAP"] - base_metrics["mAP"],
                   "wall_clock_min": (time.time() - t_start) / 60}, f, indent=2)
    print(f"\n>>> total wall-clock: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
