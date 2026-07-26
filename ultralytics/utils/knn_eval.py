# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""ImageNet kNN evaluation for measuring backbone feature quality.

Standard protocol used by DINO, DINOv2, EUPE, and AM-RADIO for evaluating self-supervised and
distilled vision encoders. Extract L2-normalized CLS features from a frozen backbone, then
classify via temperature-weighted k-nearest-neighbor voting on ImageNet-1k.

Algorithm (Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
arXiv:1805.01978, Section 3.4 -- also adopted by DINO arXiv:2104.14294 and all successors):
  1. L2-normalize all features (train + val)
  2. For each val sample, find k nearest train neighbors by dot-product similarity
  3. Weight each neighbor's vote by exp(similarity / temperature)
  4. Predicted class = argmax of accumulated weighted votes

Implementation follows RADIO's single-GPU kNN (NVlabs/RADIO examples/knn_classification.py:193-336)
using the efficient scatter_add_ voting pattern from _get_vote_cls (line 193-209).
EUPE's distributed KnnModule (facebookresearch/eupe eupe/eval/knn.py:96-184) uses mathematically
equivalent softmax(sim/T) voting but with multi-GPU broadcast/gather -- unnecessary for our setup.

Default hyperparameters: k=20, T=0.07 (used by DINO, EUPE Table 1, RADIO, DINOv2).
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("ultralytics")


def yolo_cls_features(model, imgs: torch.Tensor) -> torch.Tensor:
    """Read the CLS-equivalent feature out of a YOLO classification model.

    Feature path matches ImageEncoderModel.loss() (nn/image_encoder.py:201-208): backbone layers 0-9 -> Classify head
    conv (512->1280, 1x1) -> AdaptiveAvgPool2d(1) -> flatten.

    Args:
        model: YOLO classification model (ClassificationModel or ImageEncoderModel).
        imgs (torch.Tensor): Batch of images (B, 3, H, W).

    Returns:
        (torch.Tensor): Un-normalized features (B, D).
    """
    x = imgs
    for m in model.model[:-1]:
        x = m(x)
    # head.conv: Conv(512->1280, k=1) on the 7x7 map. head.pool: AdaptiveAvgPool2d(1), a global average that stands in
    # for a CLS token.
    head = model.model[-1]
    return head.pool(head.conv(x)).flatten(1)


@torch.no_grad()
def extract_features(model, dataloader, device, feature_fn=yolo_cls_features):
    """Extract L2-normalized CLS features from a backbone.

    L2-normalization follows RADIO _build_database (examples/knn_classification.py:355) and EUPE ModelWithNormalize
    (eupe/eval/utils.py:30-36).

    Uses fp32 for feature extraction (matching our val precision rules in val_image_encoder.py:61-72 and UNIC convention
    at unic/main_unic.py:432).

    Args:
        model: Model to read features from, matching ``feature_fn``.
        dataloader: DataLoader yielding {"img": tensor, "cls": tensor}.
        device: torch.device for computation.
        feature_fn (Callable, optional): Maps ``(model, imgs)`` to un-normalized features (B, D). Defaults to the YOLO
            classification path. ``run_knn_eval.py`` passes a ``TeacherModel.encode`` reader to score a released
            reference encoder under the same protocol.

    Returns:
        (tuple): (features, labels) tensors on CPU. features shape (N, D), labels shape (N,).
    """
    was_training = model.training
    model.eval()
    all_features, all_labels = [], []

    for batch in dataloader:
        imgs = batch["img"].to(device, non_blocking=True).float()
        labels = batch["cls"].to(device, non_blocking=True)

        features = F.normalize(feature_fn(model, imgs), dim=1, p=2)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu().long().squeeze())

    if was_training:
        model.train()
    return torch.cat(all_features), torch.cat(all_labels)


def knn_accuracy(
    train_features,
    train_labels,
    val_features,
    val_labels,
    k=20,
    temp=0.07,
    num_classes=1000,
    chunk_size=256,
    device=None,
):
    """Compute kNN top-1 accuracy using temperature-weighted voting.

    For each val image, find k nearest neighbors in train set by cosine similarity (dot product of L2-normalized
    features), weight by exp(sim / temp), accumulate class votes via scatter_add_, predict via argmax.

    Voting follows RADIO _get_vote_cls (examples/knn_classification.py:193-209): weights = exp(sim / 0.07)
    cls_vec.scatter_add_(dim=1, index=labels, src=weights) vote_id = argmax(cls_vec) This is mathematically equivalent
    to EUPE's softmax(sim/T) voting (eupe/eval/knn.py:178) since the softmax denominator is constant across classes and
    cancels in argmax.

    Chunked computation avoids materializing full (N_val x N_train) similarity matrix. With chunk_size=256 and
    N_train=1.28M: ~1.3 GB GPU memory per chunk.

    Args:
        train_features (Tensor): L2-normalized train features (N_train, D) on CPU.
        train_labels (Tensor): Train labels (N_train,) on CPU.
        val_features (Tensor): L2-normalized val features (N_val, D) on CPU.
        val_labels (Tensor): Val labels (N_val,) on CPU.
        k (int): Number of nearest neighbors.
        temp (float): Temperature for softmax weighting.
        num_classes (int): Number of classes.
        chunk_size (int): Val images processed per GPU batch.
        device: torch.device for computation. Defaults to cuda if available.

    Returns:
        (float): Top-1 accuracy as percentage (0-100).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_features_gpu = train_features.to(device)  # ~6.5 GB for 1.28M x 1280 x fp32
    train_labels_gpu = train_labels.to(device)
    correct = 0
    total = 0

    for i in range(0, len(val_features), chunk_size):
        chunk_feats = val_features[i : i + chunk_size].to(device)
        chunk_labels = val_labels[i : i + chunk_size].to(device)

        # Cosine similarity via dot product (features are L2-normalized)
        sims = chunk_feats @ train_features_gpu.T  # (chunk, N_train)
        topk_sims, topk_idx = sims.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels_gpu[topk_idx]  # (chunk, k)

        # Temperature-weighted voting per RADIO _get_vote_cls:199-208
        weights = torch.exp(topk_sims / temp)
        cls_votes = torch.zeros(chunk_feats.shape[0], num_classes, dtype=weights.dtype, device=device)
        cls_votes.scatter_add_(dim=1, index=topk_labels, src=weights)
        preds = cls_votes.argmax(dim=1)

        correct += (preds == chunk_labels).sum().item()
        total += chunk_labels.shape[0]

    return 100.0 * correct / total
