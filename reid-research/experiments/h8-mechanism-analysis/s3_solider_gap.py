"""Stage 3 of h8 — champion vs SOLIDER feature-space gap analysis on the winnable set W.

W = {q : champion.r1[q] == 0 AND solider.r1[q] == 1}
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cka import linear_cka


ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures" / "s3"
OUT = ROOT / "to_human"


def _build_sets(champ: pd.DataFrame, sol: pd.DataFrame) -> dict[str, np.ndarray]:
    c_ok = set(champ.loc[champ["r1"] == 1, "query_id"])
    s_ok = set(sol.loc[sol["r1"] == 1, "query_id"])
    all_q = set(champ["query_id"])
    W = sorted(s_ok - c_ok)
    S = sorted(c_ok & s_ok)
    H = sorted(all_q - c_ok - s_ok)
    return {"W": np.array(W), "S": np.array(S), "H": np.array(H)}


def _margin_geometry(model_emb_path: Path, retr: pd.DataFrame, qids: np.ndarray) -> np.ndarray:
    emb = torch.load(model_emb_path, weights_only=False)
    qid_to_idx = {qid: i for i, qid in enumerate(emb["query_ids"])}
    gid_to_idx = {gid: i for i, gid in enumerate(emb["gallery_ids"])}
    margins = []
    retr_by_qid = retr.set_index("query_id")
    for qid in qids:
        row = retr_by_qid.loc[qid]
        q_emb = emb["query"][qid_to_idx[qid]].numpy()
        top1_gid = row["top50_gallery_ids"][0]
        top1_emb = emb["gallery"][gid_to_idx[top1_gid]].numpy()
        pids_arr = np.asarray(row["top50_pids"])
        matches = np.where(pids_arr == row["true_pid"])[0]
        if len(matches) > 0:
            ti = int(matches[0])
            true_emb = emb["gallery"][gid_to_idx[row["top50_gallery_ids"][ti]]].numpy()
            cos_true = float(q_emb @ true_emb)
        else:
            cos_true = float("nan")
        cos_top1 = float(q_emb @ top1_emb)
        margins.append(cos_true - cos_top1)
    return np.array(margins)


def _saliency_divergence(sal_dir_a: Path, sal_dir_b: Path, qids: np.ndarray) -> np.ndarray:
    div = []
    for qid in qids:
        a = np.load(sal_dir_a / f"{qid}.npy").flatten()
        b = np.load(sal_dir_b / f"{qid}.npy").flatten()
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        if a.std() == 0 or b.std() == 0:
            div.append(float("nan"))
            continue
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        div.append(1.0 - c)
    return np.array(div)


def _cka_heatmaps(qids_S: np.ndarray, qids_W: np.ndarray) -> dict:
    p4_c = torch.load(ART / "champion" / "feats_p4.pt", weights_only=False)
    p5_c = torch.load(ART / "champion" / "feats_p5.pt", weights_only=False)
    p4_s = torch.load(ART / "solider" / "feats_p4.pt", weights_only=False)
    p5_s = torch.load(ART / "solider" / "feats_p5.pt", weights_only=False)

    emb_c = torch.load(ART / "champion" / "embeddings.pt", weights_only=False)
    qid_to_idx = {qid: i for i, qid in enumerate(emb_c["query_ids"])}

    def gather(qids, feat_dict):
        idx = [qid_to_idx[q] for q in qids]
        return feat_dict["query"][idx].numpy()

    out = {}
    for label, qids in (("S", qids_S[:2000]), ("W", qids_W)):
        if len(qids) == 0:
            continue
        c_p4 = gather(qids, p4_c); c_p5 = gather(qids, p5_c)
        s_p4 = gather(qids, p4_s); s_p5 = gather(qids, p5_s)
        mat = np.array([
            [linear_cka(c_p4, s_p4), linear_cka(c_p4, s_p5)],
            [linear_cka(c_p5, s_p4), linear_cka(c_p5, s_p5)],
        ])
        out[label] = mat
    return out


def _overlay(rgb: np.ndarray, sal: np.ndarray) -> np.ndarray:
    """Resize saliency to rgb shape, normalise to 0..1, blend as red channel."""
    h, w = rgb.shape[:2]
    s = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    if s.max() > 0:
        s = s / s.max()
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0].astype(np.float32) + 200 * s, 0, 255).astype(np.uint8)
    return overlay


def write_high_divergence_contact_sheet(qids: np.ndarray, div: np.ndarray, out_path: Path, n: int = 12):
    """For top-N most divergent queries, save a row each: query | champion-sal-overlay | solider-sal-overlay."""
    meta = pd.read_parquet(ART / "market_meta.parquet").set_index("image_id")
    sal_c = ART / "champion" / "saliency"
    sal_s = ART / "solider" / "saliency"
    order = np.argsort(-np.nan_to_num(div, nan=-1))[:n]
    rows = []
    for i in order:
        qid = qids[i]
        rgb = cv2.cvtColor(cv2.imread(meta.loc[qid, "img_path"]), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (64, 128))
        sc = np.load(sal_c / f"{qid}.npy")
        ss = np.load(sal_s / f"{qid}.npy")
        sc_img = _overlay(rgb, sc)
        ss_img = _overlay(rgb, ss)
        rows.append(np.hstack([rgb, sc_img, ss_img]))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    champ = pd.read_parquet(ART / "champion" / "retrieval.parquet")
    sol = pd.read_parquet(ART / "solider" / "retrieval.parquet")
    sets = _build_sets(champ, sol)

    findings = ["# s3 — Champion vs SOLIDER Gap\n"]
    findings.append(
        "> **CKA caveat (load-bearing):** Cross-architecture CKA between champion "
        "(CNN, P4/P5 spatial-pooled) and SOLIDER (Swin transformer, stage-3/4) is a "
        "*localization hint only*, not a content explanation. Swin's tokenization breaks "
        "the spatial alignment CNN features have. Read CKA as 'where they differ', never "
        "as 'what they encode'.\n"
    )
    findings.append(f"\n## Sets\n\n- |W| = {len(sets['W'])} (winnable: champ wrong, sol right)\n- |S| = {len(sets['S'])}\n- |H| = {len(sets['H'])}\n")

    div = None
    if len(sets["W"]) < 50:
        findings.append("\n**|W| < 50 — Stage 3 downgrades to qualitative case study; quantitative claims suppressed.**\n")
    else:
        m_c = _margin_geometry(ART / "champion" / "embeddings.pt", champ, sets["W"])
        m_s = _margin_geometry(ART / "solider" / "embeddings.pt", sol, sets["W"])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(m_c, m_s, s=8, alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("champion margin (cos_true − cos_top1)")
        ax.set_ylabel("SOLIDER margin (cos_true − cos_top1)")
        ax.set_title(f"Margin geometry on W (n={len(sets['W'])})")
        fig.tight_layout(); fig.savefig(FIG / "margin_scatter_W.png", dpi=150); plt.close(fig)

        sal_c_dir = ART / "champion" / "saliency"
        sal_s_dir = ART / "solider" / "saliency"
        if sal_c_dir.exists() and sal_s_dir.exists() and any(sal_c_dir.iterdir()) and any(sal_s_dir.iterdir()):
            div = _saliency_divergence(sal_c_dir, sal_s_dir, sets["W"])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(div[~np.isnan(div)], bins=30)
            ax.set_xlabel("saliency divergence (1 − cosine)")
            ax.set_ylabel("queries")
            ax.set_title(f"Champion vs SOLIDER saliency divergence on W (n={(~np.isnan(div)).sum()})")
            fig.tight_layout(); fig.savefig(FIG / "saliency_divergence_hist.png", dpi=150); plt.close(fig)
        else:
            div = None
            print("    saliency dirs missing/empty — skipping saliency-divergence + high-divergence contact sheet")

        cka = _cka_heatmaps(sets["S"], sets["W"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (label, mat) in zip(axes, cka.items()):
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["sol_stage3", "sol_stage4"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["champ_p4", "champ_p5"])
            ax.set_title(f"Linear-CKA on {label}")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white")
        fig.tight_layout(); fig.savefig(FIG / "cka_S_vs_W.png", dpi=150); plt.close(fig)

        findings.append(f"\n## Margin geometry on W\n\nSee `figures/s3/margin_scatter_W.png`.\n")
        findings.append(f"- champion margin: median={np.nanmedian(m_c):+.4f}, frac<0 = {(m_c<0).mean():.2f}\n")
        findings.append(f"- SOLIDER  margin: median={np.nanmedian(m_s):+.4f}, frac<0 = {(m_s<0).mean():.2f}\n")
        if div is not None and (~np.isnan(div)).any():
            findings.append(f"\n## Saliency divergence on W\n\nMedian = {np.nanmedian(div):.3f}; top-quartile (≥{np.nanquantile(div, 0.75):.3f}) flagged for qualitative inspection.\n")
        else:
            findings.append("\n## Saliency divergence on W\n\nSkipped (no IG saliency maps produced — H8_SKIP_SALIENCY was set during extraction).\n")
        findings.append(f"\n## CKA on S vs W\n\n```\nS:\n{cka['S']}\n\nW:\n{cka['W']}\n```\n")

    # Cross-reference with Stage 2 cluster labels (if Stage 2 has run).
    clusters_path = ART / "s2_failure_clusters.parquet"
    if clusters_path.exists() and len(sets["W"]) > 0:
        clusters = pd.read_parquet(clusters_path).set_index("query_id")
        in_clusters = [qid for qid in sets["W"] if qid in clusters.index]
        if in_clusters:
            ct = clusters.loc[in_clusters, "cluster"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ct.plot(kind="bar", ax=ax)
            ax.set_xlabel("Stage-2 cluster id")
            ax.set_ylabel("count of W-set queries")
            ax.set_title(f"W set crossed with s2 clusters (n_in_clusters={len(in_clusters)}/{len(sets['W'])})")
            fig.tight_layout(); fig.savefig(FIG / "W_vs_s2_clusters_crosstab.png", dpi=150); plt.close(fig)
            findings.append(f"\n## Bridge to Stage 2\n\n{ct.to_markdown()}\n\nSee `figures/s3/W_vs_s2_clusters_crosstab.png`.\n")
        else:
            findings.append("\n## Bridge to Stage 2\n\nNone of the W-set queries appear in s2 cluster labels (s2 only clusters queries whose true match was within top-50). Bridge skipped.\n")
    else:
        findings.append("\n## Bridge to Stage 2\n\nStage 2 has not produced cluster labels; bridge skipped.\n")

    # High-divergence contact sheet (top 12 by saliency divergence on W).
    if len(sets["W"]) >= 12 and div is not None:
        write_high_divergence_contact_sheet(sets["W"], div, FIG / "contact_sheet_high_divergence.png")

    with open(OUT / "s3_findings.md", "w") as f:
        f.write("".join(findings))


if __name__ == "__main__":
    main()
