import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report

# ==========================================
# OPTIMIZED HYPERPARAMETERS
# ==========================================
CSV_FILE = "dataset.csv"         
IMAGE_DIR = "."
BATCH_SIZE = 8
EPOCHS = 40                  
INITIAL_LR = 1e-4            # Stable warmup learning rate
FINETUNE_LR = 8e-6           # Balanced fine-tuning rate (slightly increased from 3e-6)
WEIGHT_DECAY = 1e-3          # Optimized L2 Regularization (softened from 1e-2)
DROPOUT_RATE = 0.3           # Balanced dropout (reduced from 0.5 to fix train deficit)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training engine spinning up on target device: {DEVICE}")

# Transform Strategies for Small Tabletop Datasets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.RandomRotation(8),                                       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Engine with Coordinated Horizontal Shifting
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
        
        # Spatial Data Augmentation matching layout shifts
        if self.is_train and np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            tl, tr = tr, tl
            bl, br = br, bl
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([tl, tr, bl, br], dtype=torch.long)

# Network Architecture
class RobustDepthCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # Classification heads optimized with modern dropout constraints
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

# Train / Val / Test Split Management
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing target file '{CSV_FILE}'.")

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

# Phase 1: Warming Up Heads
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
            
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
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
            
    # Evaluation Verification Block
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += sum(criterion(outputs[i], labels[:, i]).item() for i in range(4))
            for i in range(4):
                val_correct += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()
                val_total += labels.size(0)
                
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {correct/total:.4f} || Val Loss: {avg_val_loss:.4f} | Val Acc: {val_correct/val_total:.4f}")
    
    # Save parameters when validation loss decreases
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_depth_model.pth")

# Final Verification
print("\n--- Running Evaluation Metrics on Saved Optimized Weights ---")
model.load_state_dict(torch.load("best_depth_model.pth", weights_only=True))
model.eval()

all_labels, all_preds = {i: [] for i in range(4)}, {i: [] for i in range(4)}
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        for i in range(4):
            all_labels[i].extend(labels[:, i].cpu().numpy())
            all_preds[i].extend(outputs[i].argmax(dim=1).cpu().numpy())

names = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Left (BL)", "Bottom-Right (BR)"]
for i in range(4):
    print(f"\nReport for {names[i]}:")
    print(classification_report(all_labels[i], all_preds[i], target_names=["Near", "Middle", "Far"], labels=[0,1,2], zero_division=0))