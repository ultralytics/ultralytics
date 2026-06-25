# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prior-mask augmentation for YOLO Anomaly v2.

Training renders a soft hint from GT bboxes and then perturbs it so the model learns to
use a noisy, weak-peak, mis-localized prior instead of treating the mask as a clean answer
key. These are data augmentations on the rendered prior, kept in a dedicated module so the
model class stays focused on architecture and forward logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer


class MaskPriorAugmenter:
    """Apply training-time augmentations to a rendered GT prior mask.

    All probabilities default to 0 (or identity ranges), so an instance with no config acts as
    a no-op passthrough. The class is stateless aside from hyper-parameters; it can be called
    from the model's training forward or from a trainer/dataset preprocessing step.

    Args:
        v2_cfg: Sub-dict under the model YAML ``anomaly_v2`` key. Recognized keys:
            mask_shuffle_p, mask_noise_std, mask_mag_range, mask_blur_sigma_max,
            mask_jitter, mask_box_drop_p, mask_distractor_p, mask_distractor_n,
            mask_erase_p, mask_warp_p, mask_mixup_p, mask_mixup_alpha,
            mask_fragment_p, mask_fragment_n,
            mask_bg_blobs_p, mask_bg_blobs_n, mask_bg_blobs_amp, mask_bg_blobs_sigma,
            mask_coherent_noise_p, mask_coherent_noise_amp, mask_coherent_noise_sigma,
            mask_floor.
    """

    def __init__(self, v2_cfg: dict | None = None):
        v2_cfg = v2_cfg or {}
        self.mask_shuffle_p = float(v2_cfg.get("mask_shuffle_p", 0.0))
        self.mask_noise_std = float(v2_cfg.get("mask_noise_std", 0.0))
        _mag = v2_cfg.get("mask_mag_range", [1.0, 1.0])
        self.mask_mag_range = (float(_mag[0]), float(_mag[1]))
        self.mask_blur_sigma_max = float(v2_cfg.get("mask_blur_sigma_max", 0.0))

        self.mask_jitter = float(v2_cfg.get("mask_jitter", 0.0))
        self.mask_box_drop_p = float(v2_cfg.get("mask_box_drop_p", 0.0))
        self.mask_distractor_p = float(v2_cfg.get("mask_distractor_p", 0.0))
        self.mask_distractor_n = int(v2_cfg.get("mask_distractor_n", 4))
        self.mask_erase_p = float(v2_cfg.get("mask_erase_p", 0.0))
        self.mask_warp_p = float(v2_cfg.get("mask_warp_p", 0.0))
        self.mask_mixup_p = float(v2_cfg.get("mask_mixup_p", 0.0))
        self.mask_mixup_alpha = float(v2_cfg.get("mask_mixup_alpha", 0.5))

        self.mask_fragment_p = float(v2_cfg.get("mask_fragment_p", 0.0))
        self.mask_fragment_n = int(v2_cfg.get("mask_fragment_n", 4))
        self.mask_bg_blobs_p = float(v2_cfg.get("mask_bg_blobs_p", 0.0))
        self.mask_bg_blobs_n = int(v2_cfg.get("mask_bg_blobs_n", 8))
        _bg_amp = v2_cfg.get("mask_bg_blobs_amp", [0.05, 0.15])
        self.mask_bg_blobs_amp = (float(_bg_amp[0]), float(_bg_amp[1]))
        _bg_sig = v2_cfg.get("mask_bg_blobs_sigma", [0.03, 0.08])
        self.mask_bg_blobs_sigma = (float(_bg_sig[0]), float(_bg_sig[1]))
        self.mask_coherent_noise_p = float(v2_cfg.get("mask_coherent_noise_p", 0.0))
        _cn_amp = v2_cfg.get("mask_coherent_noise_amp", [0.02, 0.06])
        self.mask_coherent_noise_amp = (float(_cn_amp[0]), float(_cn_amp[1]))
        _cn_sig = v2_cfg.get("mask_coherent_noise_sigma", [0.05, 0.15])
        self.mask_coherent_noise_sigma = (float(_cn_sig[0]), float(_cn_sig[1]))
        _floor = v2_cfg.get("mask_floor", [0.0, 0.0])
        self.mask_floor = (float(_floor[0]), float(_floor[1]))

    def augment_prior_bboxes(
        self, bboxes: torch.Tensor | None, batch_idx: torch.Tensor | None, training: bool = True
    ):
        """Train-only bbox-level prior augs: per-box drop + center jitter (on a local copy).

        Returns possibly-reduced/perturbed ``(bboxes, batch_idx)`` used ONLY to render the
        fusion prior. The originals (used for detection GT in loss()) are untouched.
        """
        if not training or bboxes is None or bboxes.shape[0] == 0:
            return bboxes, batch_idx
        bb, bi = bboxes, batch_idx
        if self.mask_box_drop_p > 0.0:
            keep = torch.rand(bb.shape[0], device=bb.device) > self.mask_box_drop_p
            bb, bi = bb[keep], bi[keep]
        if self.mask_jitter > 0.0 and bb.shape[0] > 0:
            bb = bb.clone()
            off = (torch.rand(bb.shape[0], 2, device=bb.device) * 2 - 1) * self.mask_jitter
            bb[:, 0] = (bb[:, 0] + off[:, 0]).clamp(0.0, 1.0)
            bb[:, 1] = (bb[:, 1] + off[:, 1]).clamp(0.0, 1.0)
        return bb, bi

    def augment_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Training-only mask augmentation: make the binary GT prior look like an inference heatmap.

        Closes the train(binary GT) vs inference(soft, weak-peak memory-bank / SegBranch heatmap)
        distribution gap. Order: shuffle (wrong-location prior) -> Gaussian blur (soft edges) ->
        per-sample magnitude scaling (peak < 1, mimics weak heatmaps) -> additive noise ->
        memory-bank-style background/coherent noise. GT boxes are unchanged, so the model learns
        "weak/soft signal here still means defect here" while tolerating background false positives.
        """
        b = mask.shape[0]
        if self.mask_shuffle_p > 0.0 and b > 1:
            swap = torch.rand(b, device=mask.device) < self.mask_shuffle_p
            if swap.any():
                # offset in [1, b-1] -> perm[i] != i, so every swapped sample gets a different mask
                offset = torch.randint(1, b, (b,), device=mask.device)
                perm = (torch.arange(b, device=mask.device) + offset) % b
                mask = torch.where(swap.view(-1, 1, 1, 1), mask[perm], mask)
        if self.mask_blur_sigma_max > 0.0:
            sigma = torch.empty(1).uniform_(1e-2, self.mask_blur_sigma_max).item()
            mask = self._gaussian_blur(mask, sigma)
        lo, hi = self.mask_mag_range
        if lo < 1.0 or hi < 1.0:
            mask = mask * torch.empty(b, 1, 1, 1, device=mask.device).uniform_(lo, hi)
        if self.mask_noise_std > 0.0:
            mask = mask + torch.randn_like(mask) * self.mask_noise_std
        mask = self._augment_prior_extra(mask)
        mask = self._augment_mask_mb_style(mask)
        return mask.clamp(0.0, 1.0)

    def visualize(
        self,
        renderer: "BboxMaskRenderer",
        bboxes: torch.Tensor | None,
        batch_idx: torch.Tensor | None,
        batch_size: int,
        save_path: str | Path | None = None,
        images: list["np.ndarray"] | None = None,
        alpha: float = 0.45,
    ) -> dict[str, torch.Tensor]:
        """Render GT and augmented prior masks, optionally saving a side-by-side grid.

        This is a debug helper so users can inspect what the model sees as a training prior.
        It runs the same bbox-level and mask-level augmentations used during training.

        Args:
            renderer: The model's ``BboxMaskRenderer`` (maps bboxes to heatmaps).
            bboxes: Normalized GT bboxes ``(N, 4)`` or ``None``.
            batch_idx: Batch index per bbox ``(N,)`` or ``None``.
            batch_size: Number of samples in the batch.
            save_path: Optional path to write a PNG grid (GT | augmented) per sample.
            images: Optional list of BGR images ``(H, W, 3)`` to blend masks onto.
            alpha: Blend weight for the mask when ``images`` is provided.

        Returns:
            Dictionary with ``"gt"`` and ``"aug"`` masks, each ``(B, 1, H, W)``.
        """
        with torch.no_grad():
            bb_aug, bi_aug = self.augment_prior_bboxes(bboxes, batch_idx, training=True)
            gt = renderer(bboxes, batch_idx, batch_size)
            base = renderer(bb_aug, bi_aug, batch_size)
            aug = self.augment_mask(base)

        if save_path is not None:
            self._save_prior_vis_grid(gt, aug, images, Path(save_path), alpha)
        return {"gt": gt, "aug": aug}

    @staticmethod
    def _save_prior_vis_grid(
        gt: torch.Tensor,
        aug: torch.Tensor,
        images: list | None,
        save_path: Path,
        alpha: float,
    ) -> None:
        """Save a grid image: each row shows [GT mask, augmented mask] for one sample."""
        import cv2
        import numpy as np

        b, _, h, w = gt.shape
        rows = []
        labels = ["GT prior", "Augmented prior"]
        for i in range(b):
            gt_img = MaskPriorAugmenter._mask_to_bgr(gt[i], images[i] if images else None, alpha)
            aug_img = MaskPriorAugmenter._mask_to_bgr(aug[i], images[i] if images else None, alpha)
            rows.append(np.hstack([gt_img, aug_img]))
        grid = np.vstack(rows)

        # Add title bar with labels.
        bar_h = max(24, grid.shape[1] // 40)
        bar = np.full((bar_h, grid.shape[1], 3), (40, 40, 40), dtype=np.uint8)
        half = grid.shape[1] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, bar_h / 60)
        thickness = max(1, int(bar_h / 20))
        for j, text in enumerate(labels):
            x = j * half + 8
            y = bar_h - 6
            cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        grid = np.vstack([bar, grid])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), grid)

    @staticmethod
    def _mask_to_bgr(mask: torch.Tensor, image: "np.ndarray | None" = None, alpha: float = 0.45) -> "np.ndarray":
        """Convert a ``(1, H, W)`` mask in [0, 1] to a BGR color-mapped image."""
        import cv2
        import numpy as np

        h, w = mask.shape[-2:]
        m = mask.squeeze(0).detach().cpu().numpy()
        m = (m * 255).clip(0, 255).astype(np.uint8)
        m_color = cv2.applyColorMap(m, cv2.COLORMAP_JET)
        if image is not None:
            image = cv2.resize(image, (w, h))
            m_color = cv2.addWeighted(image, 1.0 - alpha, m_color, alpha, 0)
        return m_color

    def fragment_prior_bboxes(self, bboxes: torch.Tensor, batch_idx: torch.Tensor):
        """Split each GT box into several sub-boxes to mimic memory-bank's fragmented response.

        Returns updated ``(bboxes, batch_idx)`` with each original box replaced by
        ``self.mask_fragment_n`` smaller boxes sampled inside it.
        """
        if bboxes.numel() == 0:
            return bboxes, batch_idx
        n_frag = self.mask_fragment_n
        frags, frag_idx = [], []
        for box, bi in zip(bboxes.unbind(0), batch_idx.unbind(0)):
            cx, cy, w, h = box.tolist()
            for _ in range(n_frag):
                fcx = cx + (torch.rand(1).item() - 0.5) * w * 0.7
                fcy = cy + (torch.rand(1).item() - 0.5) * h * 0.7
                fw = w * (0.35 + 0.30 * torch.rand(1).item())
                fh = h * (0.35 + 0.30 * torch.rand(1).item())
                frags.append([fcx, fcy, fw, fh])
                frag_idx.append(bi.item())
        return (
            torch.tensor(frags, dtype=bboxes.dtype, device=bboxes.device),
            torch.tensor(frag_idx, dtype=batch_idx.dtype, device=batch_idx.device),
        )

    def _augment_prior_extra(self, mask: torch.Tensor) -> torch.Tensor:
        """Train-only mask-level prior augs: additive mixup, distractor, partial erase, warp."""
        b, _, H, W = mask.shape
        dev = mask.device
        # additive mixup: own + alpha * donor (soft distractor)
        if self.mask_mixup_p > 0.0 and b > 1:
            sel = torch.rand(b, device=dev) < self.mask_mixup_p
            donor = mask[torch.randperm(b, device=dev)]
            mixed = (mask + self.mask_mixup_alpha * donor).clamp(0.0, 1.0)
            mask = torch.where(sel.view(-1, 1, 1, 1), mixed, mask)
        # hard distractor: max-merge up to n random other-sample blobs
        if self.mask_distractor_p > 0.0 and b > 1:
            sel = torch.rand(b, device=dev) < self.mask_distractor_p
            out = mask
            for _ in range(self.mask_distractor_n):
                donor = mask[torch.randperm(b, device=dev)]
                add = sel & (torch.rand(b, device=dev) < 0.6)
                out = torch.where(add.view(-1, 1, 1, 1), torch.maximum(out, donor), out)
            mask = out
        # partial erase: zero a random sub-region of selected samples
        if self.mask_erase_p > 0.0:
            for i in range(b):
                if torch.rand(1, device=dev).item() < self.mask_erase_p:
                    eh = int(H * (0.2 + 0.3 * torch.rand(1).item()))
                    ew = int(W * (0.2 + 0.3 * torch.rand(1).item()))
                    y = int(torch.randint(0, max(1, H - eh), (1,)).item())
                    x = int(torch.randint(0, max(1, W - ew), (1,)).item())
                    mask[i, :, y : y + eh, x : x + ew] = 0.0
        # elastic warp: irregular, non-elliptical blob shape
        if self.mask_warp_p > 0.0:
            sel = torch.rand(b, device=dev) < self.mask_warp_p
            if sel.any():
                warped = self._elastic_warp(mask, alpha=0.06 * H, sigma=max(2.0, 0.03 * H))
                mask = torch.where(sel.view(-1, 1, 1, 1), warped, mask)
        return mask

    def _augment_mask_mb_style(self, mask: torch.Tensor) -> torch.Tensor:
        """Memory-bank-style post-render augmentations (scattered FP blobs + coherent noise + floor)."""
        b, _, H, W = mask.shape
        if self.mask_bg_blobs_p > 0.0 and torch.rand(1).item() < self.mask_bg_blobs_p:
            amp_lo, amp_hi = self.mask_bg_blobs_amp
            sigma_lo, sigma_hi = self.mask_bg_blobs_sigma
            bg = self._make_background_blob_mask(
                b, H, W, self.mask_bg_blobs_n, amp_lo, amp_hi, sigma_lo * H, sigma_hi * H, mask.device
            )
            mask = torch.maximum(mask, bg)
        if self.mask_coherent_noise_p > 0.0 and torch.rand(1).item() < self.mask_coherent_noise_p:
            mask = self._add_coherent_noise(mask)
        if self.mask_floor[1] > 0.0:
            mask = self._apply_mask_floor(mask)
        return mask

    def _add_coherent_noise(self, mask: torch.Tensor) -> torch.Tensor:
        """Add low-frequency coherent blobby noise (not i.i.d. pixel noise)."""
        b, _, H, W = mask.shape
        dev = mask.device
        amp_lo, amp_hi = self.mask_coherent_noise_amp
        sigma_lo, sigma_hi = self.mask_coherent_noise_sigma
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing="ij",
        )
        out = mask.clone()
        for _ in range(b):
            n_centers = int(torch.randint(2, 6, (1,)).item())
            for _ in range(n_centers):
                cx = torch.rand(1).item() * 1.8 - 0.9
                cy = torch.rand(1).item() * 1.8 - 0.9
                sigma = sigma_lo + torch.rand(1).item() * (sigma_hi - sigma_lo)
                amp = amp_lo + torch.rand(1).item() * (amp_hi - amp_lo)
                blob = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
                out = out + amp * blob
        return out

    def _apply_mask_floor(self, mask: torch.Tensor) -> torch.Tensor:
        """Add a per-sample uniform noise floor to the whole heatmap."""
        lo, hi = self.mask_floor
        if lo <= 0.0 and hi <= 0.0:
            return mask
        b = mask.shape[0]
        floor = torch.empty(b, 1, 1, 1, device=mask.device).uniform_(lo, hi)
        return mask + floor

    @staticmethod
    def _make_background_blob_mask(b, H, W, n_blobs, amp_lo, amp_hi, sigma_lo, sigma_hi, device):
        """Create a ``(B, 1, H, W)`` tensor of scattered low-amplitude Gaussian blobs."""
        bg = torch.zeros(b, 1, H, W, device=device)
        if n_blobs <= 0:
            return bg
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        for _ in range(n_blobs):
            cx = torch.rand(1).item() * 1.9 - 0.95
            cy = torch.rand(1).item() * 1.9 - 0.95
            sigma = sigma_lo + torch.rand(1).item() * (sigma_hi - sigma_lo)
            amp = amp_lo + torch.rand(1).item() * (amp_hi - amp_lo)
            blob = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
            bg = torch.maximum(bg, amp * blob)
        return bg

    @staticmethod
    def _elastic_warp(mask, alpha, sigma):
        """Random low-frequency elastic deformation of a ``(B, 1, H, W)`` mask (per-sample field)."""
        b, _, H, W = mask.shape
        dev = mask.device
        disp = torch.randn(b * 2, 1, H, W, device=dev)
        disp = MaskPriorAugmenter._gaussian_blur(disp, sigma).reshape(b, 2, H, W) * alpha
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij"
        )
        base = torch.stack((xs, ys), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        grid = base + torch.stack((disp[:, 0] / (W / 2), disp[:, 1] / (H / 2)), dim=-1)
        return F.grid_sample(mask, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @staticmethod
    def _gaussian_blur(mask, sigma):
        """Separable Gaussian blur of a ``(B, 1, H, W)`` mask with ``sigma`` in pixels."""
        k = max(3, int(2 * round(3.0 * sigma) + 1))  # odd kernel spanning ~3 sigma
        coords = torch.arange(k, device=mask.device, dtype=mask.dtype) - k // 2
        g = torch.exp(-(coords**2) / (2.0 * sigma**2))
        g = g / g.sum()
        mask = F.conv2d(mask, g.view(1, 1, 1, k), padding=(0, k // 2))
        mask = F.conv2d(mask, g.view(1, 1, k, 1), padding=(k // 2, 0))
        return mask
