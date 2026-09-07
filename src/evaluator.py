"""
Evaluation and Metrics Engine for Depth & Quadrant Hazard Classification
Author: Bhanu Vignesh Naidu Ganeshna
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from .dataset import QUADRANT_NAMES, IDX_MAP


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates multi-head depth CNN across validation/test DataLoader.

    Returns:
        Dictionary containing overall accuracy, quadrant-wise accuracies, and predictions.
    """
    model.eval()
    all_preds: List[List[int]] = [[] for _ in range(4)]
    all_labels: List[List[int]] = [[] for _ in range(4)]

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            labels = batch[1]
            outputs = model(imgs)

            for i in range(4):
                preds = outputs[i].argmax(dim=1).cpu().tolist()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].tolist())

    quad_accs = {}
    for i, name in enumerate(QUADRANT_NAMES):
        acc = accuracy_score(all_labels[i], all_preds[i]) if len(all_labels[i]) > 0 else 0.0
        quad_accs[name] = float(acc)

    overall_acc = float(np.mean(list(quad_accs.values()))) if quad_accs else 0.0

    return {
        "overall_accuracy": overall_acc,
        "quadrant_accuracies": quad_accs,
        "predictions": all_preds,
        "labels": all_labels
    }


def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Formats a clean terminal summary string of evaluation metrics."""
    all_preds = results["predictions"]
    all_labels = results["labels"]
    lines = [
        "=" * 60,
        "🏆 4-QUADRANT SPATIAL RISK EVALUATION REPORT",
        "=" * 60,
        f"Overall Mean Quadrant Accuracy: {results['overall_accuracy'] * 100:.2f}%",
        "-" * 60
    ]

    for i, name in enumerate(QUADRANT_NAMES):
        acc = results["quadrant_accuracies"].get(name, 0.0)
        lines.append(f"  - [{name}] Quadrant Accuracy: {acc * 100:.2f}%")

    lines.append("=" * 60)
    return "\n".join(lines)
