"""Configuration for Skin Cancer Classification pipeline."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    image_dirs: List[str] = field(default_factory=lambda: [
        "../data/HAM10000_images_part_1",
        "../data/HAM10000_images_part_2",
    ])
    metadata_path: str = "../data/HAM10000_metadata.csv"
    image_size: Tuple[int, int] = (100, 75)
    test_size: float = 0.2
    batch_size: int = 32
    random_state: int = 123
    num_classes: int = 7


@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    epochs: int = 10
    dropout_rate: float = 0.5
    checkpoint_path: str = "checkpoints/best_model.pth"


@dataclass
class ExplainabilityConfig:
    gradcam_target_layer: str = "layer2"
    lime_num_samples: int = 1000
    lime_num_features: int = 5


LESION_TYPES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}
