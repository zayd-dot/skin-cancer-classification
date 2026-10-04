"""Dataset loading and preprocessing for HAM10000 skin lesion images."""

import os
import pandas as pd
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional

from configs.config import DataConfig, LESION_TYPES


class SkinCancerDataset(Dataset):
    """PyTorch Dataset for HAM10000 skin cancer images.

    Args:
        dataframe: DataFrame with 'path' and 'cell_type_idx' columns.
        transform: Torchvision transforms to apply to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        img_path = row["path"]

        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = row["cell_type_idx"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_metadata(cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Load HAM10000 metadata and map image paths.

    Args:
        cfg: Data configuration.

    Returns:
        DataFrame with path, cell_type, and cell_type_idx columns.
    """
    # Collect image paths from all directories
    imageid_path_dict: Dict[str, str] = {}
    for img_dir in cfg.image_dirs:
        paths = glob(os.path.join(img_dir, "*.jpg"))
        imageid_path_dict.update({os.path.splitext(os.path.basename(p))[0]: p for p in paths})

    print(f"Total images found: {len(imageid_path_dict)}")

    skin_df = pd.read_csv(cfg.metadata_path)
    skin_df["path"] = skin_df["image_id"].map(imageid_path_dict.get)
    skin_df["cell_type"] = skin_df["dx"].map(LESION_TYPES.get)
    skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes
    skin_df["age"] = skin_df["age"].fillna(skin_df["age"].mean())

    return skin_df


def get_dataloaders(cfg: DataConfig = DataConfig()) -> Tuple[DataLoader, DataLoader, SkinCancerDataset, SkinCancerDataset]:
    """Create train and test DataLoaders.

    Args:
        cfg: Data configuration.

    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset).
    """
    skin_df = load_metadata(cfg)

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_df, test_df = train_test_split(skin_df, test_size=cfg.test_size, random_state=cfg.random_state)

    train_dataset = SkinCancerDataset(train_df, transform=transform)
    test_dataset = SkinCancerDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset
