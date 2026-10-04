"""Run Grad-CAM and LIME explanations on a trained model."""

import argparse
import torch
import numpy as np

from src.dataset import get_dataloaders
from src.model import SkinCancerResNet
from src.explainability import generate_gradcam, generate_lime_explanation
from configs.config import DataConfig, ModelConfig, ExplainabilityConfig


def explain(
    image_idx: int = 2,
    cfg_data: DataConfig = DataConfig(),
    cfg_model: ModelConfig = ModelConfig(),
    cfg_explain: ExplainabilityConfig = ExplainabilityConfig(),
) -> None:
    """Generate Grad-CAM and LIME explanations for a test image.

    Args:
        image_idx: Index of the test image to explain.
        cfg_data: Data configuration.
        cfg_model: Model configuration.
        cfg_explain: Explainability configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SkinCancerResNet(num_classes=cfg_data.num_classes)
    model.load_state_dict(torch.load(cfg_model.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    _, _, _, test_dataset = get_dataloaders(cfg_data)
    img, label = test_dataset[image_idx]

    # Grad-CAM
    print(f"Generating Grad-CAM for test image {image_idx} (true label: {label})...")
    target_layer = getattr(model, cfg_explain.gradcam_target_layer)
    img_tensor = img.unsqueeze(0).to(device)
    generate_gradcam(model, img_tensor, target_layer, save_path="results/gradcam.png")

    # LIME
    print("Generating LIME explanation...")
    img_np = img.numpy()
    generate_lime_explanation(
        model, img_np, device,
        num_samples=cfg_explain.lime_num_samples,
        num_features=cfg_explain.lime_num_features,
        save_path="results/lime.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain model predictions")
    parser.add_argument("--image-idx", type=int, default=2, help="Test image index")
    args = parser.parse_args()
    explain(image_idx=args.image_idx)
