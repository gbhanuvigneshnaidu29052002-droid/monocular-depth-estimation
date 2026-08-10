import os
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
from sklearn.metrics import classification_report

# ==========================================
# 1. PIPELINE CONFIGURATION & DIRECTORIES
# ==========================================
CSV_FILE = "dataset.csv"         
IMAGE_DIR = "."
OUTPUT_DIR = "presentation_assets"

# Generate timestamped directories to prevent overwriting results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("results", f"run_{timestamp}")
OUTPUT_DIR = os.path.join(RUN_DIR, "presentation_assets")
PRED_FOLDER = os.path.join(RUN_DIR, "depth_predictions_output")
MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_depth_model.pth")

# Static folders (for copying latest results for easy evaluation)
STATIC_OUTPUT_DIR = "presentation_assets"
STATIC_PRED_FOLDER = "depth_predictions_output"
STATIC_MODEL_PATH = "best_depth_model.pth"

BATCH_SIZE = 8
EPOCHS = 50                  
INITIAL_LR = 1e-4            # Stable warmup learning rate for Phase 1
FINETUNE_LR = 8e-6           # Balanced fine-tuning rate for Phase 2
WEIGHT_DECAY = 1e-3          # Balanced L2 Regularization to improve accuracy
DROPOUT_RATE = 0.3           # Moderate dropout to prevent underfitting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure secure directory trees exist
for folder in [OUTPUT_DIR, PRED_FOLDER, STATIC_OUTPUT_DIR, STATIC_PRED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set random seeds for strict repeatability
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"=======================================================")
print(f"SYSTEM: Monocular Depth Pipeline initiated on: {DEVICE}")
print(f"Run output directory: {RUN_DIR}")
print(f"=======================================================")

# ==========================================
# 2. TRANSFORMS & DATA ENGINE SETUP
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.RandomRotation(8),                                       
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Mimics depth changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RobustDepthDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.label_map = {'N': 0, 'M': 1, 'F': 2}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        
        tl = self.label_map[row["TL"]]
        tr = self.label_map[row["TR"]]
        bl = self.label_map[row["BL"]]
        br = self.label_map[row["BR"]]
        
        # Spatial Data Augmentation matching coordinate flips
        if self.is_train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            tl, tr = tr, tl
            bl, br = br, bl
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([tl, tr, bl, br], dtype=torch.long), row["filename"]

# ==========================================
# 3. ARCHITECTURE DEFINITION
# ==========================================
class RobustDepthCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # 4 independent processing heads matching the 2x2 spatial divisions
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

# Load and split dataset
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing core index spreadsheet file '{CSV_FILE}'.")

df = pd.read_csv(CSV_FILE)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n_total = len(df)
n_val = max(1, int(n_total * 0.12))
n_test = max(1, int(n_total * 0.12))

df_test = df.iloc[:n_test]
df_val = df.iloc[n_test : n_test + n_val]
df_train = df.iloc[n_test + n_val:]

train_loader = DataLoader(RobustDepthDataset(df_train, IMAGE_DIR, train_transform, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RobustDepthDataset(df_val, IMAGE_DIR, val_transform, is_train=False), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(RobustDepthDataset(df_test, IMAGE_DIR, val_transform, is_train=False), batch_size=BATCH_SIZE, shuffle=False)

model = RobustDepthCNN(dropout_p=DROPOUT_RATE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler (disabled in Phase 1, initialized when Phase 2 starts)
scheduler = None

# Tracking vectors for plotting
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# ==========================================
# 4. TWO-PHASE OPTIMIZATION LOOP
# ==========================================
print("\n--- Phase 1: Training Classification Heads (Backbone Frozen) ---")
for param in model.backbone.parameters():
    param.requires_grad = False

best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    if epoch == 11:
        print("\n--- Phase 2: Fine-tuning Full Network (Backbone Unfrozen) ---")
        for param in model.backbone.parameters():
            param.requires_grad = True
        for param_group in optimizer.param_groups:
            param_group['lr'] = FINETUNE_LR
        # Initialize learning rate decay scheduler for fine-tuning stability
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
            
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        for i in range(4):
            correct += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()
            total += labels.size(0)
            
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += sum(criterion(outputs[i], labels[:, i]).item() for i in range(4))
            for i in range(4):
                val_correct += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()
                val_total += labels.size(0)
                
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_accuracy = correct / total
    val_accuracy = val_correct / val_total
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_accuracy)
    history['val_acc'].append(val_accuracy)
    
    # Step scheduler in Phase 2
    if scheduler is not None:
        scheduler.step()
        
    print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} || Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# ==========================================
# 5. DYNAMIC CHART GENERATION (MATPLOTLIB)
# ==========================================
print("\n--- Generating Metric and Performance Curves ---")
import matplotlib.pyplot as plt
epochs_arr = np.arange(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet18 Multi-Head Architecture: Balanced Convergence Curves", fontsize=14, fontweight='bold', y=1.02)

ax1.plot(epochs_arr, history['train_loss'], label="Train Loss", color="#1f77b4", linewidth=2.5)
ax1.plot(epochs_arr, history['val_loss'], label="Val Loss", color="#ff7f0e", linewidth=2.5, linestyle="--")
ax1.axvline(x=10.5, color="purple", linestyle=":", alpha=0.7, label="Phase 2 Start")
ax1.set_title("Multi-Head Summed Cross-Entropy Loss", fontsize=12)
ax1.set_xlabel("Epochs", fontsize=10)
ax1.set_ylabel("Loss Magnitude", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

ax2.plot(epochs_arr, history['train_acc'], label="Train Accuracy", color="#2ca02c", linewidth=2.5)
ax2.plot(epochs_arr, history['val_acc'], label="Val Accuracy", color="#d62728", linewidth=2.5, linestyle="--")
ax2.axvline(x=10.5, color="purple", linestyle=":", alpha=0.7)
ax2.set_title("Average Structural Grid Spatial Accuracy", fontsize=12)
ax2.set_xlabel("Epochs", fontsize=10)
ax2.set_ylabel("Accuracy (%)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_convergence_curves.png")
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Graphics Matrix Exported to: '{curve_path}'")

# ==========================================
# 6. PIPELINE QUANTITATIVE EVALUATION
# ==========================================
print("\n--- Running Evaluation Metrics on Saved Optimized Weights ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
model.eval()

all_labels, all_preds = {i: [] for i in range(4)}, {i: [] for i in range(4)}
with torch.no_grad():
    for images, labels, _ in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        for i in range(4):
            all_labels[i].extend(labels[:, i].cpu().numpy())
            all_preds[i].extend(outputs[i].argmax(dim=1).cpu().numpy())

names = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Left (BL)", "Bottom-Right (BR)"]
for i in range(4):
    print(f"\nReport for {names[i]}:")
    print(classification_report(all_labels[i], all_preds[i], target_names=["Near", "Middle", "Far"], labels=[0,1,2], zero_division=0))

# ==========================================
# 7. QUALITATIVE OVERLAY PROCESSING GENERATOR
# ==========================================
print("\n--- Running Batch Inference and Visual Overlay Generation ---")
COLOR_MAP = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}  # Red, Yellow, Green (BGR)
LABEL_MAP = {0: "Near", 1: "Middle", 2: "Far"}

with torch.no_grad():
    for images, labels, filenames in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        
        # Batch unpacking
        for b in range(images.size(0)):
            img_name = filenames[b]
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            orig_img = cv2.imread(img_path)
            h, w, _ = orig_img.shape
            overlay = orig_img.copy()
            
            preds = [outputs[i][b].argmax().item() for i in range(4)]
            
            quad_coords = [
                (0, 0, w//2, h//2),         # TL
                (w//2, 0, w, h//2),         # TR
                (0, h//2, w//2, h),         # BL
                (w//2, h//2, w, h)          # BR
            ]
            quad_names = ["TL", "TR", "BL", "BR"]
            
            for q_idx, (x1, y1, x2, y2) in enumerate(quad_coords):
                pred_class = preds[q_idx]
                color = COLOR_MAP[pred_class]
                label_text = f"{quad_names[q_idx]}: {LABEL_MAP[pred_class]}"
                
                # Draw semi-transparent background color block
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Draw a solid background box behind text for high contrast on cutting mats
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size[0], text_size[1]
                cv2.rectangle(orig_img, (x1 + 10, y1 + 15), (x1 + 20 + text_w, y1 + 45), (0, 0, 0), -1)
                
                # Render high-contrast label text
                cv2.putText(orig_img, label_text, (x1 + 15, y1 + 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Sharp alignment grid lines splitting quadrants
            cv2.line(orig_img, (w//2, 0), (w//2, h), (0, 0, 0), 2)
            cv2.line(orig_img, (0, h//2), (w, h//2), (0, 0, 0), 2)
            
            # Apply alpha blending (35% overlay thickness)
            final_output = cv2.addWeighted(overlay, 0.35, orig_img, 0.65, 0)
            
            out_path = os.path.join(PRED_FOLDER, f"pred_{img_name}")
            cv2.imwrite(out_path, final_output)

# Copy run outputs to default/static directories for evaluation scripts
if os.path.exists(MODEL_SAVE_PATH):
    shutil.copy2(MODEL_SAVE_PATH, STATIC_MODEL_PATH)
    
for filename in os.listdir(OUTPUT_DIR):
    shutil.copy2(os.path.join(OUTPUT_DIR, filename), os.path.join(STATIC_OUTPUT_DIR, filename))
    
for filename in os.listdir(PRED_FOLDER):
    shutil.copy2(os.path.join(PRED_FOLDER, filename), os.path.join(STATIC_PRED_FOLDER, filename))

print(f"\n=======================================================")
print(f"SUCCESS: Pipeline fully executed.")
print(f"📂 All run results saved in timestamped directory: {RUN_DIR}")
print(f"✅ Latest model validation weights copied to: '{STATIC_MODEL_PATH}'")
print(f"✅ Latest statistical curves copied to: '{STATIC_OUTPUT_DIR}'")
print(f"✅ Latest visual prediction overlays copied to: '{STATIC_PRED_FOLDER}'")
print(f"=======================================================")