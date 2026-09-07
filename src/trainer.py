"""
Training Pipeline for Multi-Task Monocular Depth Risk Network
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import DepthCNN
from .dataset import DepthDataset, get_transforms
from .losses import MultiTaskQuadrantLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

    return float(running_loss / max(1, total_samples))


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = "best_model.pth"
) -> Dict[str, Any]:
    """Runs end-to-end model training."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthCNN(dropout_p=0.3).to(dev)
    criterion = MultiTaskQuadrantLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    print(f"🚀 Training DepthCNN on device: {dev} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, dev)
        history["train_loss"].append(loss)

        if save_path and loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss:.4f}")

    return {"model": model, "history": history}
