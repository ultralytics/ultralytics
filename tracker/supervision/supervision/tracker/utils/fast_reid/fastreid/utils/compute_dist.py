# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# Modified from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/compute_dist.py


import faiss
import numpy as np
import torch
import torch.nn.functional as F

from .faiss_utils import (
    index_init_cpu,
    index_init_gpu,
    search_index_pytorch,
    search_raw_array_pytorch,
)

__all__ = [
    "build_dist",
    "compute_jaccard_distance",
    "compute_euclidean_distance",
    "compute_cosine_distance",
]


@torch.no_grad()
def build_dist(feat_1: torch.Tensor, feat_2: torch.Tensor, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Compute distance between two feature embeddings.

    Args:
        feat_1 (torch.Tensor): 2-D feature with batch dimension.
        feat_2 (torch.Tensor): 2-D feature with batch dimension.
        metric:

    Returns:
        numpy.ndarray: distance matrix.
    """
    assert metric in ["cosine", "euclidean", "jaccard"], "Expected metrics are cosine, euclidean and jaccard, " \
                                                         "but got {}".format(metric)

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)

    elif metric == "jaccard":
        feat = torch.cat((feat_1, feat_2), dim=0)
        dist = compute_jaccard_distance(feat, k1=kwargs["k1"], k2=kwargs["k2"], search_option=0)
        return dist[: feat_1.size(0), feat_1.size(0):]


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


@torch.no_grad()
def compute_jaccard_distance(features, k1=20, k2=6, search_option=0, fp16=False):
    if search_option < 3:
        # torch.cuda.empty_cache()
        features = features.cuda()

    ngpus = faiss.get_num_gpus()
    N = features.size(0)
    mat_type = np.float16 if fp16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index
        )  # element-wise unique

        x = features[i].unsqueeze(0).contiguous()
        y = features[k_reciprocal_expansion_index]
        m, n = x.size(0), y.size(0)
        dist = (
                torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
                + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(x, y.t(), beta=1, alpha=-2)

        if fp16:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            )
        else:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy()
            )

    del nn_k1, nn_k1_half, x, y
    features = features.cpu()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist


@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()
