"""Model interpretability with Grad-CAM and LIME.

Provides visual explanations of which image regions drive the model's
predictions, supporting both gradient-based (Grad-CAM) and
perturbation-based (LIME) approaches.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from typing import Optional


def generate_gradcam(
    model: torch.nn.Module,
    img: torch.Tensor,
    target_layer: torch.nn.Module,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image.

    Args:
        model: Trained PyTorch model.
        img: Input tensor of shape (1, C, H, W).
        target_layer: Layer to compute gradients for.
        save_path: If provided, save the visualization.

    Returns:
        Grad-CAM heatmap as uint8 numpy array.
    """
    model.train()  # needed for gradient computation

    features, gradients = None, None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(img)
    target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward(retain_graph=True)

    fwd_handle.remove()
    bwd_handle.remove()

    if gradients is None:
        raise ValueError("Gradients are None — backward pass may have failed.")

    # Weight feature maps by mean gradient per channel
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.size(1)):
        features[0, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(features, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Prepare original image for overlay
    orig = img.cpu().squeeze().permute(1, 2, 0).numpy()
    orig = ((orig - orig.min()) / (orig.max() - orig.min()) * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return heatmap_colored


def generate_lime_explanation(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    num_samples: int = 1000,
    num_features: int = 5,
    save_path: Optional[str] = None,
):
    """Generate LIME explanation for a single image.

    Args:
        model: Trained PyTorch model.
        img: Image as numpy array (C, H, W) or (H, W, C).
        device: Torch device.
        num_samples: Number of perturbed samples for LIME.
        num_features: Number of superpixel features to highlight.
        save_path: If provided, save the visualization.

    Returns:
        LIME Explanation object.
    """
    # Convert to HWC uint8
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img).convert("RGB")

    def predict_fn(images):
        batch = torch.stack([
            torch.tensor(im.transpose(2, 0, 1)).float() for im in images
        ]).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(pil_img),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pil_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="RdBu_r")
    axes[1].set_title("LIME Importance Mask")
    axes[1].axis("off")

    axes[2].imshow(temp)
    axes[2].set_title("LIME Explanation")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return explanation
