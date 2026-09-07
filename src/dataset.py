"""
Dataset Engine and Spatial Augmentation Pipelines for Depth & Risk Analysis
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import random
from typing import Tuple, Optional, Dict
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP: Dict[str, int] = {'N': 0, 'M': 1, 'F': 2}
IDX_MAP: Dict[int, str] = {0: 'Near', 1: 'Middle', 2: 'Far'}
QUADRANT_NAMES = ["TL", "TR", "BL", "BR"]


def get_transforms(is_train: bool = False) -> transforms.Compose:
    """Build standardized image transformations."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DepthDataset(Dataset):
    """
    Dataset handling 4-quadrant proximity classification labels
    with coordinate-aware horizontal flipping augmentations.
    """

    def __init__(
        self,
        df_or_csv,
        img_dir: str = ".",
        transform: Optional[transforms.Compose] = None,
        is_train: bool = False
    ):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv).reset_index(drop=True)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform or get_transforms(is_train=is_train)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        img_path = os.path.join(self.img_dir, filename)

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback black canvas for testing/missing files
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        tl = LABEL_MAP[str(row["TL"]).strip()]
        tr = LABEL_MAP[str(row["TR"]).strip()]
        bl = LABEL_MAP[str(row["BL"]).strip()]
        br = LABEL_MAP[str(row["BR"]).strip()]

        # Coordinate-aware horizontal flip: swap Left and Right labels
        if self.is_train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            tl, tr = tr, tl
            bl, br = br, bl

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([tl, tr, bl, br], dtype=torch.long)
        return image, labels, filename
