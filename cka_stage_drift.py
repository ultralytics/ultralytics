# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Measure how far each backbone stage moves between a phase 1 checkpoint and its phase 2 fine-tune.

Linear CKA between the two checkpoints' activations at the same trunk row, on the same images. A low value means
that stage's representation was rewritten during fine-tuning, a high value means it survived. The quantity of
interest is the ordering across stages, not any single number, so every stage is measured on the same images.

Reads rows 2, 4, 6 and 8, the blocks emitting P2, P3, P4 and P5. Rows 0-8 are structurally identical in the phase 1
classification model and the phase 2 detection model, so the two are compared directly with no weight transfer.

Run as ``python cka_stage_drift.py <gpu> <phase1.pt> <phase2.pt> <image_dir>``.
"""

import sys
from pathlib import Path

import torch

ROWS = {2: "P2", 4: "P3", 6: "P4", 8: "P5"}
N_IMAGES = 512
CHUNK = 32  # P2 emits 160x160 activations, so the trunk runs in chunks and only the sampled positions are kept
PER_IMAGE = 64  # spatial positions sampled per image per stage
BOOTSTRAP = 200


def load_trunk(path, device):
    """Backbone rows 0-8 of an Ultralytics checkpoint, in eval mode at fp32."""
    model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return model.model[:9].float().eval().to(device)


def features(path, images, device, seed):
    """Map each row in ROWS to an (N*PER_IMAGE, C) activation matrix, sampling spatial positions per image.

    The generator is re-seeded per chunk so a given chunk draws identical positions for either checkpoint, which is what
    makes the two matrices row-aligned and the CKA meaningful.
    """
    trunk = load_trunk(path, device)
    out = {i: [] for i in ROWS}
    for start in range(0, len(images), CHUNK):
        g = torch.Generator(device=device).manual_seed(seed + start)
        x = images[start : start + CHUNK].to(device)
        for i, block in enumerate(trunk):
            x = block(x)
            if i in ROWS:
                n, c, h, w = x.shape
                idx = torch.randint(h * w, (n, PER_IMAGE), generator=g, device=device)
                flat = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
                out[i].append(flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, c)).reshape(-1, c).double().cpu())
    del trunk
    torch.cuda.empty_cache()
    return {i: torch.cat(v).to(device) for i, v in out.items()}


def cka(x, y):
    """Linear CKA between two activation matrices sharing a row order."""
    x, y = x - x.mean(0), y - y.mean(0)
    return (y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())


def load_images(image_dir, n):
    """One CPU batch of 640px images, the same pixels for both checkpoints."""
    import cv2
    import numpy as np

    paths = sorted(Path(image_dir).glob("*.jpg"))[:n]
    assert len(paths) == n, f"found {len(paths)} images in {image_dir}, need {n}"
    batch = np.stack([cv2.resize(cv2.imread(str(p))[..., ::-1], (640, 640)) for p in paths])
    return torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).float().div(255)


if __name__ == "__main__":
    gpu, p1, p2, image_dir = sys.argv[1:5]
    device = torch.device(f"cuda:{gpu}")
    images = load_images(image_dir, N_IMAGES)
    print(f"{N_IMAGES} images, {PER_IMAGE} positions per image per stage\n")

    with torch.no_grad():
        f1 = features(p1, images, device, seed=0)
        f2 = features(p2, images, device, seed=0)

    g = torch.Generator(device=device).manual_seed(1)
    for row, name in ROWS.items():
        a, b = f1[row], f2[row]
        boot = torch.stack(
            [cka(a[i], b[i]) for i in torch.randint(a.shape[0], (BOOTSTRAP, a.shape[0]), generator=g, device=device)]
        )
        lo, hi = boot.quantile(0.025).item(), boot.quantile(0.975).item()
        print(f"{name}  CKA {cka(a, b):.4f}  95% CI [{lo:.4f}, {hi:.4f}]  dim {a.shape[1]}")
