import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
MODEL_PATH = "best_model.pth"
TEST_CSV = "test.csv"
IMAGE_DIR = "."
OUTPUT_FOLDER = "depth_predictions_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output layout color maps (BGR format for OpenCV)
# Near (N) -> Red, Middle (M) -> Yellow, Far (F) -> Green
COLOR_MAP = {
    0: (0, 0, 255),    # Red (Near)
    1: (0, 255, 255),  # Yellow (Middle)
    2: (0, 255, 0)     # Green (Far)
}
LABEL_MAP = {0: "Near", 1: "Middle", 2: "Far"}

# Validation/Test transformation strategy
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Recreate the exact network architecture used in training
class DepthCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(128, 3)
            ) for _ in range(4)
        ])

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]

# ==========================================
# PRODUCTION PIPELINE ENGINE
# ==========================================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing weight target '{MODEL_PATH}'. Run train.py first.")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Missing test CSV index file '{TEST_CSV}'.")

    # Create output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: '{OUTPUT_FOLDER}'")
    else:
        print(f"Target directory '{OUTPUT_FOLDER}' verified.")

    # Load Model State
    model = DepthCNN(dropout_p=0.5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Model weights successfully initialized. Processing test images...")

    # Load test split metadata
    df_test = pd.read_csv(TEST_CSV)
    print(f"Found {len(df_test)} test images. Starting generation sequence...")

    success_count = 0
    for idx, row in df_test.iterrows():
        img_name = row['filename']
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} missing. Skipping.")
            continue

        # Load original image
        orig_img = cv2.imread(img_path)
        h, w, _ = orig_img.shape

        # Build tensor input
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            preds = [out.argmax(dim=1).item() for out in outputs]

        # Ground truths
        gts = [row['TL'], row['TR'], row['BL'], row['BR']]
        gt_mapped = [LABEL_MAP[0] if g == 'N' else (LABEL_MAP[1] if g == 'M' else LABEL_MAP[2]) for g in gts]

        # Initialize transparent overlay mask
        overlay = orig_img.copy()

        # Coordinate bounding anchors for the 2x2 multi-output grid
        quad_coords = [
            (0, 0, w//2, h//2),         # 0: Top-Left (TL)
            (w//2, 0, w, h//2),         # 1: Top-Right (TR)
            (0, h//2, w//2, h),         # 2: Bottom-Left (BL)
            (w//2, h//2, w, h)          # 3: Bottom-Right (BR)
        ]
        names = ["TL", "TR", "BL", "BR"]

        for q_idx, (x1, y1, x2, y2) in enumerate(quad_coords):
            pred_class = preds[q_idx]
            color = COLOR_MAP[pred_class]
            
            # Format labels: Pred vs Ground Truth
            label_text = f"{names[q_idx]} Pred: {LABEL_MAP[pred_class]}"
            gt_text   = f"GT: {gt_mapped[q_idx]}"

            # Generate alpha blended geometric boundaries on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw labels with drop shadows for maximum legibility
            cv2.putText(orig_img, label_text, (x1 + 30, y1 + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(orig_img, label_text, (x1 + 30, y1 + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(orig_img, gt_text, (x1 + 30, y1 + 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(orig_img, gt_text, (x1 + 30, y1 + 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if LABEL_MAP[pred_class] == gt_mapped[q_idx] else (0, 0, 255), 2, cv2.LINE_AA)

        # Clear layout grid lines
        cv2.line(orig_img, (w//2, 0), (w//2, h), (0, 0, 0), 4)
        cv2.line(orig_img, (0, h//2), (w, h//2), (0, 0, 0), 4)

        # Apply specific blending opacity alpha parameter (35% overlay thickness)
        alpha = 0.35
        final_output = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

        # Export completed file matrix directly into targeted folder 
        out_name = f"pred_{img_name}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, final_output)
        success_count += 1
        print(f"Generated visualization for: {img_name}")
        
    print(f"\n=======================================================")
    print(f"SUCCESS: Pipeline execution complete.")
    print(f"Generated {success_count} colored overlay prediction files.")
    print(f"All items saved in: '{os.path.abspath(OUTPUT_FOLDER)}'")
    print(f"=======================================================")

if __name__ == "__main__":
    main()