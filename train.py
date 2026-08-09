import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
FOLDER = "."  # Use relative path for portability
EPOCHS = 50
BATCH = 8
INITIAL_LR = 1e-4
FINETUNE_LR = 3e-6
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.5
NUM_CLASS = 3
LABEL_MAP = {'N': 0, 'M': 1, 'F': 2}
IDX_MAP = {0: 'N', 1: 'M', 2: 'F'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────────────────────────────
# Robust to lighting and minor rotations
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset with Horizontal Flip label swap ──────────────────────────────────
class DepthDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        
        tl = LABEL_MAP[row["TL"]]
        tr = LABEL_MAP[row["TR"]]
        bl = LABEL_MAP[row["BL"]]
        br = LABEL_MAP[row["BR"]]

        # Spatial data augmentation (flip left-right and swap labels)
        if self.is_train and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            tl, tr = tr, tl
            bl, br = br, bl

        img = self.transform(img)
        labs = torch.tensor([tl, tr, bl, br], dtype=torch.long)
        return img, labs

# ── Model ──
class DepthCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(128, NUM_CLASS)
            ) for _ in range(4)
        ])

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]

# ── Main guard ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Training pipeline spinning up on device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found. Training on CPU.")

    # Datasets
    train_ds = DepthDataset(os.path.join(FOLDER, "train.csv"), FOLDER, train_tf, is_train=True)
    val_ds   = DepthDataset(os.path.join(FOLDER, "val.csv"),   FOLDER, val_tf, is_train=False)
    test_ds  = DepthDataset(os.path.join(FOLDER, "test.csv"),  FOLDER, val_tf, is_train=False)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    # Calculate class weights dynamically on train set to combat imbalance
    train_df = pd.read_csv(os.path.join(FOLDER, "train.csv"))
    WEIGHTS = {}
    regions = ["TL", "TR", "BL", "BR"]
    for reg in regions:
        counts = train_df[reg].value_counts().to_dict()
        total = len(train_df)
        weights_list = []
        for class_char in ['N', 'M', 'F']:
            count = counts.get(class_char, 0)
            if count > 0:
                w = total / (3.0 * count)
            else:
                w = 1.0  # fallback for missing classes in split
            weights_list.append(w)
        # Normalize weights to sum to 3
        weights_sum = sum(weights_list)
        weights_list = [w * 3.0 / weights_sum for w in weights_list]
        WEIGHTS[reg] = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
        print(f"Weights for {reg}: {weights_list}")

    model = DepthCNN(dropout_p=DROPOUT_RATE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    best_path = os.path.join(FOLDER, "best_model.pth")
    patience = 7
    patience_counter = 0

    def compute_loss(outputs, labels):
        total = 0
        for i, region in enumerate(regions):
            criterion = nn.CrossEntropyLoss(weight=WEIGHTS[region])
            total += criterion(outputs[i], labels[:, i])
        return total

    print("\n" + "="*70)
    print(f"{'Epoch':>5} | {'Phase':>8} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("="*70)

    for epoch in range(1, EPOCHS + 1):
        # Two-phase training freeze/unfreeze
        if epoch == 1:
            phase_name = "Warmup"
            # Freeze backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == 11:
            phase_name = "FineTune"
            # Unfreeze backbone
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Set fine-tuning learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = FINETUNE_LR
            # Reset validation tracking to protect model weights of phase 1
            # but keep best checkpoint if it was good
        else:
            phase_name = "Warmup" if epoch < 11 else "FineTune"

        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            for i in range(4):
                t_correct += (outputs[i].argmax(1) == labels[:, i]).sum().item()
                t_total += labels.size(0)

        scheduler.step()
        train_acc = t_correct / t_total
        train_loss = t_loss / len(train_loader)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss_batch = compute_loss(outputs, labels)
                v_loss += val_loss_batch.item()
                for i in range(4):
                    v_correct += (outputs[i].argmax(1) == labels[:, i]).sum().item()
                    v_total += labels.size(0)
        
        val_acc = v_correct / v_total
        val_loss = v_loss / len(val_loader)

        # Checkpoint and early stopping monitoring (patience checked only in fine-tuning phase)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            flag = " <- best"
            patience_counter = 0
        else:
            flag = ""
            if epoch >= 11:
                patience_counter += 1

        print(f"{epoch:>5} | {phase_name:>8} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f}{flag}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] Model stabilized at Epoch {epoch}.")
            break

    print("="*70)
    print(f"Model saved to: {best_path}")

    # ── Test Set Evaluation ───────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    all_preds  = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            for i in range(4):
                all_preds[i].extend(outputs[i].argmax(1).cpu().tolist())
                all_labels[i].extend(labels[:, i].cpu().tolist())

    for i, region in enumerate(regions):
        print(f"\n[{region}] Classification Report:")
        print(classification_report(
            all_labels[i],
            all_preds[i],
            labels=[0, 1, 2],
            target_names=["Near", "Middle", "Far"],
            zero_division=0
        ))
