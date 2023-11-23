"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""

import numpy as np
import torch


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dsr_dist(array1, array2, distmat, scores):
    """ Compute the sptial feature reconstruction of all pairs
     array: [M, N, C] M: the number of query, N: the number of spatial feature, C: the dimension of each spatial feature
     array2: [M, N, C] M: the number of gallery
    :return:
    numpy array with shape [m1, m2]
    """
    dist = 100 * torch.ones(len(array1), len(array2))
    dist = dist.cuda()
    kappa = 0.001
    index = np.argsort(distmat, axis=1)
    T = kappa * torch.eye(110)
    T = T.cuda()
    M = []
    for i in range(0, len(array2)):
        g = array2[i]
        g = torch.FloatTensor(g)
        g = g.view(g.size(0), g.size(1))
        g = g.cuda()
        Proj_M1 = torch.matmul(torch.inverse(torch.matmul(g.t(), g) + T), g.t())
        Proj_M1 = Proj_M1.cpu().numpy()
        M.append(Proj_M1)
    for i in range(0, len(array1)):
        q = torch.FloatTensor(array1[i])
        q = q.view(q.size(0), q.size(1))
        q = q.cuda()
        for j in range(0, 100):
            g = array2[index[i, j]]
            g = torch.FloatTensor(g)
            g = g.view(g.size(0), g.size(1))
            g = g.cuda()
            Proj_M = torch.FloatTensor(M[index[i, j]])
            Proj_M = Proj_M.cuda()
            a = torch.matmul(g, torch.matmul(Proj_M, q)) - q
            dist[i, index[i, j]] = ((torch.pow(a, 2).sum(0).sqrt()) * scores[i].cuda()).sum()
    dist = dist.cpu()
    dist = dist.numpy()

    return dist
