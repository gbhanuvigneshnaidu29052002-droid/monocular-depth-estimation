import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report

# ---------- Config ----------
FOLDER    = "."
BATCH     = 8
NUM_CLASS = 3
LABEL_MAP = {'N': 0, 'M': 1, 'F': 2}
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Transform (same as validation) ----------
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Dataset ----------
class DepthDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor([LABEL_MAP[row["TL"]], LABEL_MAP[row["TR"]],
                               LABEL_MAP[row["BL"]], LABEL_MAP[row["BR"]]], dtype=torch.long)
        return img, labels

# ---------- Model definition (must match training) ----------
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
                nn.Linear(128, NUM_CLASS)
            ) for _ in range(4)
        ])
    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]

if __name__ == "__main__":
    print(f"Evaluating on device: {DEVICE}")

    # Load test data
    test_ds = DepthDataset(os.path.join(FOLDER, "test.csv"), FOLDER, val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Load best model
    model = DepthCNN(dropout_p=0.3).to(DEVICE)
    best_path = os.path.join(FOLDER, "best_model.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Model file not found at {best_path}. Please train the model first.")
    
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Collect predictions
    regions = ["TL", "TR", "BL", "BR"]
    all_preds = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            for i in range(4):
                preds = outputs[i].argmax(1).cpu().tolist()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].tolist())

    # Print reports with explicit labels
    print("\n" + "="*60)
    print("Test Set Evaluation (best_model.pth)")
    print("="*60)
    for i, region in enumerate(regions):
        print(f"\n[{region}] Classification Report:")
        print(classification_report(
            all_labels[i],
            all_preds[i],
            labels=[0, 1, 2],
            target_names=["Near", "Middle", "Far"],
            zero_division=0
        ))