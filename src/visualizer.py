"""
4-Quadrant Spatial Hazard Risk Overlay Visualization Engine
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),    # Red (Near - High Hazard Proximity)
    1: (0, 255, 255),  # Yellow (Middle - Moderate Caution)
    2: (0, 255, 0)     # Green (Far - Safe Clearance)
}

LABEL_MAP: Dict[int, str] = {0: "Near", 1: "Middle", 2: "Far"}
QUAD_NAMES = ["TL", "TR", "BL", "BR"]


def create_quadrant_overlay(
    image: Union[np.ndarray, Image.Image],
    preds: List[int],
    gts: Optional[List[Union[int, str]]] = None,
    alpha: float = 0.35
) -> np.ndarray:
    """
    Renders 4-quadrant colored spatial bounding zones, risk labels, and optional ground-truth.

    Args:
        image: BGR numpy image or PIL image.
        preds: List of 4 predicted class indices [TL, TR, BL, BR] in {0, 1, 2}.
        gts: Optional ground-truth classes.
        alpha: Transparency factor for hazard zones.

    Returns:
        Annotated BGR image as numpy ndarray.
    """
    if isinstance(image, Image.Image):
        orig_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        orig_img = image.copy()

    h, w = orig_img.shape[:2]
    overlay = orig_img.copy()

    # Coordinates for the 2x2 grid
    quad_coords = [
        (0, 0, w // 2, h // 2),        # Top-Left (TL)
        (w // 2, 0, w, h // 2),        # Top-Right (TR)
        (0, h // 2, w // 2, h),        # Bottom-Left (BL)
        (w // 2, h // 2, w, h)         # Bottom-Right (BR)
    ]

    for q_idx, (x1, y1, x2, y2) in enumerate(quad_coords):
        pred_cls = preds[q_idx]
        color = COLOR_MAP.get(pred_cls, (255, 255, 255))

        # Fill quadrant zone on overlay mask
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        pred_name = LABEL_MAP.get(pred_cls, str(pred_cls))
        label_text = f"{QUAD_NAMES[q_idx]} Pred: {pred_name}"

        # Draw labels with high-contrast text outlines
        font_scale = max(0.6, min(w, h) / 600.0)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(orig_img, label_text, (x1 + 15, y1 + int(40 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(orig_img, label_text, (x1 + 15, y1 + int(40 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if gts is not None and len(gts) > q_idx:
            gt_val = gts[q_idx]
            gt_str = LABEL_MAP.get(gt_val, str(gt_val)) if isinstance(gt_val, int) else str(gt_val)
            gt_text = f"GT: {gt_str}"
            is_match = (gt_str.startswith("N") and pred_cls == 0) or \
                       (gt_str.startswith("M") and pred_cls == 1) or \
                       (gt_str.startswith("F") and pred_cls == 2)
            gt_color = (0, 255, 0) if is_match else (0, 0, 255)

            cv2.putText(orig_img, gt_text, (x1 + 15, y1 + int(80 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(orig_img, gt_text, (x1 + 15, y1 + int(80 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, gt_color, thickness, cv2.LINE_AA)

    # Dividing grid crosshairs
    cv2.line(orig_img, (w // 2, 0), (w // 2, h), (0, 0, 0), max(2, thickness))
    cv2.line(orig_img, (0, h // 2), (w, h // 2), (0, 0, 0), max(2, thickness))

    # Alpha blending
    return cv2.addWeighted(overlay, alpha, orig_img, 1.0 - alpha, 0)
