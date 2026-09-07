"""
Monocular Depth Estimation & Spatial 4-Quadrant Hazard Risk Analysis Package
Author: Bhanu Vignesh Naidu Ganeshna
"""

from .models import DepthCNN, build_model
from .dataset import DepthDataset, get_transforms, LABEL_MAP, IDX_MAP, QUADRANT_NAMES
from .losses import (
    scale_invariant_log_loss,
    compute_abs_rel,
    compute_rmse,
    MultiTaskQuadrantLoss
)
from .evaluator import evaluate_model, format_evaluation_report
from .visualizer import create_quadrant_overlay
from .trainer import train_model, train_epoch

__version__ = "1.0.0"
__author__ = "Bhanu Vignesh Naidu Ganeshna"
__all__ = [
    "DepthCNN",
    "build_model",
    "DepthDataset",
    "get_transforms",
    "LABEL_MAP",
    "IDX_MAP",
    "QUADRANT_NAMES",
    "scale_invariant_log_loss",
    "compute_abs_rel",
    "compute_rmse",
    "MultiTaskQuadrantLoss",
    "evaluate_model",
    "format_evaluation_report",
    "create_quadrant_overlay",
    "train_model",
    "train_epoch"
]
