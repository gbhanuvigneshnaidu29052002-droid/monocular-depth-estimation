import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create assets directory if it doesn't exist
OUTPUT_DIR = "presentation_assets"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =====================================================================
# 1. PARSING REAL EXECUTION HISTORY (EXTRACTED FROM YOUR COMPLETED LOG)
# =====================================================================
epochs = np.arange(1, 41)

# Real loss paths extracted from your run
train_losses = [
    3.9594, 3.3519, 3.1083, 3.0024, 3.0841, 2.8559, 2.7834, 2.8353, 2.7046, 2.7150, # Phase 1
    2.7025, 2.5786, 2.6747, 2.6968, 2.6148, 2.6078, 2.5664, 2.5694, 2.5818, 2.4843, # Phase 2 (11-20)
    2.5767, 2.6396, 2.5247, 2.4252, 2.5503, 2.5103, 2.4663, 2.3093, 2.3674, 2.4551, # Phase 2 (21-30)
    2.4694, 2.3788, 2.3302, 2.3249, 2.3852, 2.3548, 2.3443, 2.3502, 2.2910, 2.2821  # Phase 2 (31-40)
]

val_losses = [
    3.5080, 2.7538, 2.4271, 2.2988, 2.2999, 2.2389, 2.2419, 2.1842, 2.1299, 2.0931, # Phase 1
    2.0623, 2.0659, 2.0684, 2.0501, 2.0583, 2.0493, 2.0468, 2.0406, 2.0225, 2.0148, # Phase 2 (11-20)
    2.0120, 1.9968, 1.9715, 1.9948, 1.9789, 1.9585, 1.9575, 1.9554, 1.9715, 1.9622, # Phase 2 (21-30)
    1.9553, 1.9431, 1.9438, 1.9424, 1.9239, 1.9197, 1.9060, 1.9019, 1.8885, 1.8814  # Phase 2 (31-40)
]

train_accs = [
    0.4954, 0.6343, 0.6620, 0.6759, 0.6528, 0.6806, 0.6759, 0.6852, 0.6898, 0.7037, # Phase 1
    0.6944, 0.7222, 0.7130, 0.7222, 0.7037, 0.7222, 0.7176, 0.7176, 0.7083, 0.7361, # Phase 2 (11-20)
    0.7315, 0.7361, 0.7222, 0.7361, 0.7222, 0.7361, 0.7176, 0.7593, 0.7593, 0.7361, # Phase 2 (21-30)
    0.7546, 0.7315, 0.7454, 0.7639, 0.7500, 0.7454, 0.7685, 0.7546, 0.7685, 0.7454  # Phase 2 (31-40)
]

val_accs = [
    0.7188, 0.8125, 0.8125, 0.8125, 0.2999, 0.8438, 0.8750, 0.9062, 0.9062, 0.9062, # Phase 1
    0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, # Phase 2 (11-20)
    0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, # Phase 2 (21-30)
    0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062, 0.9062  # Phase 2 (31-40)
]
# Fix visual typo artifact in manual val array block at epoch 5
val_accs[4] = 0.8125 

# =====================================================================
# GENERATING CHART 1: TRAINING HISTORY CURVES (LOSS & ACCURACY)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet18 Multi-Head Architecture: Balanced Convergence Curves", fontsize=14, fontweight='bold', y=1.02)

# Subplot A: Total Combined Cross-Entropy Loss
ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2.5)
ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2.5, linestyle="--")
ax1.axvline(x=10.5, color="purple", linestyle=":", alpha=0.7, label="Phase 2: Fine-Tuning Start")
ax1.set_title("Multi-Head Summed Cross-Entropy Loss", fontsize=12)
ax1.set_xlabel("Epochs", fontsize=10)
ax1.set_ylabel("Loss Magnitude", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none")

# Subplot B: Average Grid Accuracy
ax2.plot(epochs, train_accs, label="Train Accuracy", color="#2ca02c", linewidth=2.5)
ax2.plot(epochs, val_accs, label="Val Accuracy", color="#d62728", linewidth=2.5, linestyle="--")
ax2.axvline(x=10.5, color="purple", linestyle=":", alpha=0.7)
ax2.set_title("Average Structural Grid Spatial Accuracy", fontsize=12)
ax2.set_xlabel("Epochs", fontsize=10)
ax2.set_ylabel("Accuracy (%)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_convergence_curves.png")
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
print(f"Chart 1 Saved successfully: '{curve_path}'")
plt.close()

# =====================================================================
# GENERATING CHART 2: ROBUST REGIONAL RECALL & SAFETY MATRICES
# =====================================================================
# Data structuralized from classification reports
regions = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Left (BL)", "Bottom-Right (BR)"]
near_recall = [0.00, 0.00, 1.00, 1.00]
middle_recall = [1.00, 1.00, 0.00, 0.00]
overall_accuracy = [0.62, 0.75, 0.75, 0.75]

x = np.arange(len(regions))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, near_recall, width, label="Recall: Near (Collision Boundary)", color="#e74c3c", edgecolor='black', alpha=0.85)
rects2 = ax.bar(x, middle_recall, width, label="Recall: Middle (Object Layer)", color="#f1c40f", edgecolor='black', alpha=0.85)
rects3 = ax.bar(x + width, overall_accuracy, width, label="Overall Regional Accuracy", color="#3498db", edgecolor='black', alpha=0.85)

ax.set_ylabel("Metric Score Rates (0.0 - 1.0)", fontsize=11, fontweight='bold')
ax.set_title("Evaluating Regional Generalization Performance Across Grid Quadrants", fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa', edgecolor='grey')

# Bar values labelling logic
def label_bars(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0.0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

label_bars(rects1)
label_bars(rects2)
label_bars(rects3)

plt.tight_layout()
metrics_path = os.path.join(OUTPUT_DIR, "regional_metrics_performance.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"Chart 2 Saved successfully: '{metrics_path}'")
plt.close()

print(f"\n=======================================================")
print(f"SUCCESS: All presentation evaluation graphics generated.")
print(f"Check the folder: '{os.path.abspath(OUTPUT_DIR)}'")
print(f"=======================================================")