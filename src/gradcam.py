"""
gradcam.py
==========
Phase 4 - Gradient-Weighted Class Activation Mapping (Grad-CAM)
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

This module provides visual interpretability for CNN models predicting Diabetic
Retinopathy from retinal fundus photographs.

Clinical Importance:
  In medical imaging, "black-box" predictions are insufficient for clinical adoption.
  Grad-CAM visualises the salient spatial regions that directed the CNN's prediction,
  allowing ophthalmologists and evaluators to verify that the network attended to
  actual pathological lesions (e.g. microaneurysms, hard exudates, cotton wool spots,
  neovascular fronds) rather than spurious background camera artefacts or black borders.

Mathematical Foundation:
  1. Neuron importance weights:
     alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
  2. Class Activation Map:
     L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )

Reference:
  Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).
  "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization."
  IEEE International Conference on Computer Vision (ICCV), pp. 618-626.

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import os
from typing import Any, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM class activation mapping utility.

    Attaches forward and backward hooks to the specified target convolutional layer,
    computes gradient weights, and generates normalized saliency heatmaps.
    """

    def __init__(
        self,
        model: Any,
        target_layer: Optional[Any] = None,
    ) -> None:
        self.model: Any = model

        # Resolve target layer robustly
        target: Optional[Any] = target_layer

        if target is None and hasattr(model, "get_gradcam_target_layer"):
            getter: Any = getattr(model, "get_gradcam_target_layer")
            if callable(getter):
                target = getter()

        if target is None and hasattr(model, "backbone"):
            bb: Any = getattr(model, "backbone")
            if hasattr(bb, "modules"):
                convs: List[Any] = [m for m in bb.modules() if isinstance(m, nn.Conv2d)]
                if convs:
                    target = convs[-1]
            if target is None and hasattr(bb, "children"):
                children: List[Any] = list(bb.children())
                if children:
                    target = children[-1]

        if target is None and hasattr(model, "modules"):
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if convs:
                target = convs[-1]

        if target is None:
            raise ValueError("Cannot automatically determine target layer for Grad-CAM.")

        self.target_layer: Any = target
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_handle: Optional[Any] = None
        self._bwd_handle: Optional[Any] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: Any, inp: Any, outp: torch.Tensor) -> None:
            self.activations = outp.detach()

        def backward_hook(module: Any, grad_inp: Any, grad_outp: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_outp[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes 2D class activation heatmap for the specified class.

        Args:
          input_tensor: (1, 3, H, W) PyTorch tensor on same device as model.
          target_class: Integer class index (0..4). If None, uses argmax(logits).

        Returns:
          cam: 2D numpy float array of shape (H, W) normalized to [0, 1].
        """
        self.model.eval()
        tensor_var = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(tensor_var)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling of gradients over spatial dimensions (H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Linear combination of forward feature activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_f, W_f)

        # Rectified Linear Unit (ReLU) to isolate positive evidentiary features
        cam = F.relu(cam)

        # Interpolate to input resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay_on_image(
        self,
        rgb_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlays the 2D heatmap onto the original RGB fundus image.

        Args:
          rgb_image: uint8 numpy array (H, W, 3) in RGB space.
          cam: float numpy array (H, W) with values in [0, 1].
          alpha: Blending weight (0.0 = original image, 1.0 = heatmap only).
          colormap: Matplotlib colormap name ('jet', 'viridis', 'turbo', 'inferno').

        Returns:
          blended: uint8 RGB numpy array (H, W, 3).
        """
        # Ensure dimensions match
        if cam.shape != rgb_image.shape[:2]:
            cam = cv2.resize(cam, (rgb_image.shape[1], rgb_image.shape[0]))

        # Dynamically retrieve colormap to support all matplotlib versions
        get_cmap_fn: Any = getattr(plt, "get_cmap")
        cmap: Any = get_cmap_fn(colormap)

        heatmap_rgba = cmap(cam)  # shape (H, W, 4) in [0, 1]
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)

        blended = (alpha * heatmap_rgb + (1.0 - alpha) * rgb_image).astype(np.uint8)
        return blended

    def close(self) -> None:
        """Cleanly deregister hooks from PyTorch module graph."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def __del__(self) -> None:
        self.close()


def generate_gradcam_heatmap(
    model: Any,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[Any] = None,
) -> np.ndarray:
    """
    Convenience function to generate a normalized 2D Grad-CAM heatmap.
    """
    cam_generator = GradCAM(model, target_layer=target_layer)
    try:
        heatmap = cam_generator.generate_cam(input_tensor, target_class=target_class)
    finally:
        cam_generator.close()
    return heatmap
