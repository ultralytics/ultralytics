"""Stage 2 of h8 — failure taxonomy on the champion's ~246 R1-misses.

Reads:  artifacts/champion/retrieval.parquet, artifacts/champion/embeddings.pt,
        artifacts/market_meta.parquet
Writes: artifacts/s2_failure_clusters.parquet, figures/s2/*.png, to_human/s2_findings.md
"""

from __future__ import annotations

from pathlib import Path

import cv2
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap

import retrieval as R


ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures" / "s2"
OUT = ROOT / "to_human"


def _tag_failures(retr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add taxonomy columns to a copy of `retr` filtered to R1=0 rows only."""
    fail = retr[retr["r1"] == 0].copy()
    q_meta = meta.set_index("image_id")
    fail["query_cam"] = q_meta.loc[fail["query_id"], "camid"].values
    fail["query_brightness"] = q_meta.loc[fail["query_id"], "mean_brightness"].values
    fail["query_aspect"] = q_meta.loc[fail["query_id"], "aspect_ratio"].values
    fail["query_occlusion"] = q_meta.loc[fail["query_id"], "occlusion_score"].values
    fail["pid_gallery_count"] = q_meta.loc[fail["query_id"], "pid_gallery_count"].values
    fail["top1_cam"] = fail["top50_camids"].str[0]
    fail["cross_camera"] = fail["query_cam"] != fail["top1_cam"]

    def classify_confusion(row):
        top1_pid = row["top50_pids"][0]
        if top1_pid == row["true_pid"]:
            return "hard_neg_same_pid"
        if row["true_pid"] in row["top50_pids"]:
            return "hard_neg_distractor"
        return "no_good_match"

    fail["confusion_type"] = fail.apply(classify_confusion, axis=1)

    def signed_margin(row):
        top1_d = row["top50_distances"][0]
        if row["true_pid"] in row["top50_pids"]:
            idx = row["top50_pids"].index(row["true_pid"])
            true_d = row["top50_distances"][idx]
        else:
            true_d = float("nan")
        return top1_d - true_d

    fail["margin_to_truth"] = fail.apply(signed_margin, axis=1)
    full_query = retr.copy()
    full_query["query_occlusion"] = q_meta.loc[full_query["query_id"], "occlusion_score"].values
    full_query["query_brightness"] = q_meta.loc[full_query["query_id"], "mean_brightness"].values
    full_query["query_aspect"] = q_meta.loc[full_query["query_id"], "aspect_ratio"].values
    fail["occlusion_bin"] = pd.qcut(full_query["query_occlusion"].fillna(0), 4, labels=["occ_q1", "occ_q2", "occ_q3", "occ_q4"], duplicates="drop").loc[fail.index]
    fail["brightness_bin"] = pd.qcut(full_query["query_brightness"], 4, labels=["bri_q1", "bri_q2", "bri_q3", "bri_q4"], duplicates="drop").loc[fail.index]
    fail["pose_bin"] = pd.qcut(full_query["query_aspect"], 4, labels=["pose_q1", "pose_q2", "pose_q3", "pose_q4"], duplicates="drop").loc[fail.index]
    fail["pid_rarity"] = (fail["pid_gallery_count"] <= 3).map({True: "low", False: "high"})
    return fail


def _crosstab_plot(fail: pd.DataFrame, out_path: Path):
    ct = pd.crosstab(fail["cross_camera"], fail["confusion_type"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="rocket", ax=ax)
    ax.set_title("Champion R1-failures: cross_camera × confusion_type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _failure_rate_per_bin(retr: pd.DataFrame, meta: pd.DataFrame, axis: str, out_path: Path):
    q_meta = meta.set_index("image_id")
    df = retr.copy()
    df[axis] = q_meta.loc[df["query_id"], axis].values
    df["bin"] = pd.qcut(df[axis].fillna(df[axis].median()), 4, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(failures=("r1", lambda s: (s == 0).sum()), total=("r1", "size"))
    g["rate"] = g["failures"] / g["total"]
    los, his = [], []
    for bin_label, sub in df.groupby("bin", observed=True):
        bits = (sub["r1"] == 0).astype(np.float64).values
        lo, hi = R.bootstrap_mean_ci(bits, n_resamples=1000, seed=0)
        los.append(lo); his.append(hi)
    g["lo"], g["hi"] = los, his
    g["underpowered"] = g["total"] < 20
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(g)), g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]], fmt="o-")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(b) for b in g.index], rotation=30)
    ax.set_ylabel("R1 failure rate")
    ax.set_title(f"Failure rate by {axis}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return g


def _residual_clusters(retr: pd.DataFrame, meta: pd.DataFrame, champion_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cluster failure residuals (UMAP -> HDBSCAN). Returns (umap_2d, labels, choice)."""
    emb = torch.load(champion_dir / "embeddings.pt", weights_only=False)
    qids = emb["query_ids"]; gids = emb["gallery_ids"]
    q_to_idx = {qid: i for i, qid in enumerate(qids)}
    g_to_idx = {gid: i for i, gid in enumerate(gids)}
    fail = retr[retr["r1"] == 0]
    residuals = []
    fail_qids = []
    for _, row in fail.iterrows():
        qid = row["query_id"]
        if row["true_pid"] not in row["top50_pids"]:
            continue
        true_idx_in_top = row["top50_pids"].index(row["true_pid"])
        true_gid = row["top50_gallery_ids"][true_idx_in_top]
        residual = emb["query"][q_to_idx[qid]].numpy() - emb["gallery"][g_to_idx[true_gid]].numpy()
        residuals.append(residual)
        fail_qids.append(qid)
    R_mat = np.stack(residuals)
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15)
    umap_2d = reducer.fit_transform(R_mat)

    best = None
    for mcs in [5, 10, 20]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(umap_2d)
        noise_frac = float((labels == -1).mean())
        choice = {"min_cluster_size": mcs, "noise_frac": noise_frac, "n_clusters": int(labels.max() + 1)}
        if noise_frac < 0.30:
            return umap_2d, labels, choice | {"selected": True, "fail_qids": fail_qids}
        if best is None or noise_frac < best[2]["noise_frac"]:
            best = (umap_2d, labels, choice, fail_qids)
    umap_b, labels_b, choice_b, fail_qids_b = best
    return umap_b, labels_b, choice_b | {"selected": False, "fail_qids": fail_qids_b}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    retr = pd.read_parquet(ART / "champion" / "retrieval.parquet")
    meta = pd.read_parquet(ART / "market_meta.parquet")
    fail = _tag_failures(retr, meta)

    _crosstab_plot(fail, FIG / "failure_crosstab.png")

    bin_summaries = {}
    for axis in ("occlusion_score", "mean_brightness", "aspect_ratio"):
        bin_summaries[axis] = _failure_rate_per_bin(retr, meta, axis, FIG / f"failure_rate_by_{axis}.png")

    umap_2d, labels, choice = _residual_clusters(retr, meta, ART / "champion")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, cmap="tab20", s=8)
    ax.set_title(f"Failure residual UMAP (HDBSCAN min_cluster_size={choice['min_cluster_size']}, noise={choice['noise_frac']:.2f})")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(FIG / "residual_umap.png", dpi=150)
    plt.close(fig)

    # Persist cluster labels so Stage 3 can cross-reference them programmatically.
    cluster_df = pd.DataFrame({"query_id": choice["fail_qids"], "cluster": labels.tolist()})
    cluster_df.to_parquet(ART / "s2_failure_clusters.parquet")

    with open(OUT / "s2_findings.md", "w") as f:
        f.write(f"# s2 — Failure Taxonomy ({len(fail)} champion R1-misses)\n\n")
        f.write(f"## Cross-cam × confusion-type\n\n{pd.crosstab(fail['cross_camera'], fail['confusion_type']).to_markdown()}\n\n")
        for axis, g in bin_summaries.items():
            f.write(f"## Failure rate by {axis}\n\n{g[['failures','total','rate','lo','hi','underpowered']].to_markdown()}\n\n")
        f.write(f"## Residual UMAP/HDBSCAN cluster choice\n\n```\n{choice}\n```\n\n")
        if not choice["selected"]:
            f.write("**WARNING:** no `min_cluster_size` setting achieved <30% noise. Residuals do not cluster cleanly; no taxonomy claim made.\n")


def write_cluster_contact_sheets(retr: pd.DataFrame, meta: pd.DataFrame, clusters_path: Path, out_dir: Path, top_per_cluster: int = 10, top_k: int = 5):
    """For each non-noise cluster, build a contact sheet: top-N worst queries × top-K retrieved gallery thumbs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_parquet(clusters_path)
    qid_to_path = meta.set_index("image_id")["img_path"].to_dict()
    retr_by_qid = retr.set_index("query_id")
    for cid, sub in clusters.groupby("cluster"):
        if cid == -1:
            continue
        qids_in_cluster = sub["query_id"].tolist()
        mAPs = retr_by_qid.loc[qids_in_cluster, "mAP_q"].values
        order = np.argsort(mAPs)[:top_per_cluster]
        chosen = [qids_in_cluster[i] for i in order]
        rows = []
        for qid in chosen:
            r = retr_by_qid.loc[qid]
            q_img = cv2.cvtColor(cv2.imread(qid_to_path[qid]), cv2.COLOR_BGR2RGB)
            tiles = [cv2.resize(q_img, (64, 128))]
            for gid in r["top50_gallery_ids"][:top_k]:
                g_img = cv2.cvtColor(cv2.imread(qid_to_path[gid]), cv2.COLOR_BGR2RGB)
                tiles.append(cv2.resize(g_img, (64, 128)))
            rows.append(np.hstack(tiles))
        sheet = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"contact_sheet_cluster_{cid}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
    write_cluster_contact_sheets(
        pd.read_parquet(ART / "champion" / "retrieval.parquet"),
        pd.read_parquet(ART / "market_meta.parquet"),
        ART / "s2_failure_clusters.parquet",
        FIG,
    )
