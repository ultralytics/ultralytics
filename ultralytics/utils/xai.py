# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class YOLO_XAI_Extractor:
    """A utility class to extract feature maps and gradients from YOLO bottlenecks for Explainable AI (XAI) feature
    dominance mapping (Grad-CAM).
    """

    def __init__(self, model, target_layer_index=None):
        self.model = model.model
        # Dynamically select the penultimate layer if no index is provided, preventing IndexError on non-detection models.
        layer_idx = target_layer_index if target_layer_index is not None else -2
        self.target_layer = self.model.model[layer_idx]
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Removes PyTorch hooks to prevent memory leaks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x):
        """Executes the forward pass."""
        return self.model(x)


def generate_gradcam_heatmap(activations, gradients, image_shape=(640, 640)):
    """Mathematically projects high-dimensional gradients and activations into a 2D spatial heatmap.

    Args:
        activations (torch.Tensor): The forward pass feature maps.
        gradients (torch.Tensor): The backward pass gradient maps.
        image_shape (tuple): The target (H, W) dimensions for the output heatmap.

    Returns:
        np.ndarray: A normalized 2D heatmap strictly between 0 and 1.
    """
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample and normalize
    cam = F.interpolate(cam, size=image_shape, mode="bilinear", align_corners=False)
    cam_final = cam.squeeze().cpu().detach().numpy()
    cam_normalized = (cam_final - cam_final.min()) / (cam_final.max() - cam_final.min() + 1e-8)

    return cam_normalized


def validate_heatmap(model, img_bgr, heatmap, target_class_idx):
    """Evaluates the faithfulness of an XAI heatmap using Deletion and Insertion metrics.

    Args:
        model (YOLO): The loaded Ultralytics YOLO model.
        img_bgr (np.ndarray): The original BGR image array.
        heatmap (np.ndarray): The normalized 2D heatmap.
        target_class_idx (int): The class index to evaluate.

    Returns:
        tuple: (Area Under Deletion Curve, Area Under Insertion Curve)
    """

    def get_confidence(img):
        results = model(img, verbose=False)
        if len(results[0].boxes) == 0:
            return 0.0
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()
        mask = cls_ids == target_class_idx
        return np.max(confs[mask]) if np.any(mask) else 0.0

    percentiles = np.arange(0, 105, 5)
    deletion_confs = []
    insertion_confs = []

    blurred_baseline = cv2.GaussianBlur(img_bgr, (51, 51), 0)

    # Ensure heatmap matches the exact dimensions of the input image to prevent broadcasting errors
    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    for pct in percentiles:
        # 1. Deletion Masking
        if pct == 100:
            masked_img = np.zeros_like(img_bgr)
        elif pct == 0:
            masked_img = img_bgr.copy()
        else:
            # Corrected logic: delete the top pct% of salient pixels
            thr = np.percentile(heatmap_resized, 100 - pct)
            mask = (heatmap_resized > thr).astype(np.uint8)
            deletion_mask = 1 - mask
            masked_img = (img_bgr * deletion_mask[..., np.newaxis]).astype(np.uint8)

        deletion_confs.append(get_confidence(masked_img))

        # 2. Insertion Masking
        if pct == 100:
            inserted_img = img_bgr.copy()
        elif pct == 0:
            inserted_img = blurred_baseline.copy()
        else:
            thr = np.percentile(heatmap_resized, 100 - pct)
            mask = (heatmap_resized >= thr).astype(np.uint8)
            inserted_img = (img_bgr * mask[..., np.newaxis] + blurred_baseline * (1 - mask[..., np.newaxis])).astype(
                np.uint8
            )

        insertion_confs.append(get_confidence(inserted_img))

    # Handle NumPy version compatibility for Area Under the Curve calculation
    try:
        integration_func = np.trapezoid
    except AttributeError:
        integration_func = np.trapz

    audc = integration_func(deletion_confs, percentiles) / 100.0
    auic = integration_func(insertion_confs, percentiles) / 100.0

    return audc, auic
