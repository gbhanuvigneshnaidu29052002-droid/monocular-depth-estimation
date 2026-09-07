"""
Neural Network Architectures for Monocular Depth & Spatial Quadrant Risk Classification
Author: Bhanu Vignesh Naidu Ganeshna
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


class DepthCNN(nn.Module):
    """
    Multi-Task Monocular Spatial Quadrant Risk Classification Network.
    Employs a ResNet-18 encoder backbone with 4 independent classification heads
    corresponding to 4 spatial hazard zones: Top-Left (TL), Top-Right (TR),
    Bottom-Left (BL), and Bottom-Right (BR).
    """

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.3, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 4 independent processing heads matching the 2x2 spatial divisions
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(128, num_classes)
            ) for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        Returns:
            List of 4 logits tensors, each of shape (B, num_classes), for [TL, TR, BL, BR].
        """
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


def build_model(num_classes: int = 3, dropout_p: float = 0.3, pretrained: bool = False, device: str = "cpu") -> DepthCNN:
    """Factory function for initializing DepthCNN model."""
    model = DepthCNN(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    return model.to(device)
