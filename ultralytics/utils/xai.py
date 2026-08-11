import torch
import torch.nn.functional as F
import cv2
import numpy as np

class YOLO_XAI_Extractor:
    """
    A utility class to extract feature maps and gradients from YOLO bottlenecks
    for Explainable AI (XAI) feature dominance mapping (Grad-CAM).
    """
    def __init__(self, model, target_layer_index=22):
        self.model = model.model
        self.target_layer = self.model.model[target_layer_index]
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
    """
    Mathematically projects high-dimensional gradients and activations 
    into a 2D spatial heatmap.
    """
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Upsample and normalize
    cam = F.interpolate(cam, size=image_shape, mode='bilinear', align_corners=False)
    cam_final = cam.squeeze().cpu().detach().numpy()
    cam_normalized = (cam_final - cam_final.min()) / (cam_final.max() - cam_final.min() + 1e-8)
    
    return cam_normalized