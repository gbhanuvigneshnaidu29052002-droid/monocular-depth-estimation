"""
Loss Functions and Evaluation Metrics for Monocular Depth Estimation & Risk Analysis
Author: Bhanu Vignesh Naidu Ganeshna
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union


def scale_invariant_log_loss(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    scale_weight: float = 1.0,
    eps: float = 1e-6
) -> float:
    """
    Computes Scale-Invariant Logarithmic Depth Loss (Eigen et al.):
        L_depth = (1 / N) * sum(d_i^2) - (lambda / N^2) * (sum(d_i))^2
    where d_i = log(y_i) - log(y_hat_i), and lambda in [0, 1] (lambda=1.0 is fully scale-invariant).

    Args:
        y_true: Ground truth depth map (must be > 0).
        y_pred: Predicted depth map (must be > 0).
        scale_weight: Lambda weighting factor for scale cancellation (default 1.0).
        eps: Small constant to avoid log(0).

    Returns:
        Scale-invariant logarithmic loss as a float.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Valid mask for positive depths
    mask = (y_true > 0) & (y_pred > 0)
    if not np.any(mask):
        return 0.0

    t = y_true[mask]
    p = y_pred[mask]
    n = float(t.size)

    d = np.log(np.clip(t, eps, None)) - np.log(np.clip(p, eps, None))
    loss = (1.0 / n) * np.sum(d ** 2) - (scale_weight / (n ** 2)) * (np.sum(d) ** 2)
    return float(max(0.0, loss))


def compute_abs_rel(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-6
) -> float:
    """
    Computes Absolute Relative Error (Abs Rel):
        AbsRel = (1 / N) * sum(|y_i - y_hat_i| / y_i)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = y_true > 0
    if not np.any(mask):
        return 0.0

    diff = np.abs(y_true[mask] - y_pred[mask])
    rel = diff / np.clip(y_true[mask], eps, None)
    return float(np.mean(rel))


def compute_rmse(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Computes Root Mean Squared Error (RMSE):
        RMSE = sqrt((1 / N) * sum((y_i - y_hat_i)^2))
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff ** 2)))


class MultiTaskQuadrantLoss(nn.Module):
    """
    Joint loss function summing CrossEntropy across all 4 spatial quadrants:
        L_total = sum_{q in {TL, TR, BL, BR}} CrossEntropy(logits_q, target_q)
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_list: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_list: List of 4 tensors each (B, num_classes).
            targets: Tensor of shape (B, 4) containing ground-truth class IDs.
        Returns:
            Scalar combined loss tensor.
        """
        total_loss = 0.0
        for i in range(4):
            total_loss = total_loss + self.criterion(logits_list[i], targets[:, i])
        return total_loss
