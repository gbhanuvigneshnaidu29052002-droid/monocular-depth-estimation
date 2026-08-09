# Monocular Depth Estimation (Task 4)

This repository contains the implementation of **Block 5 - Monocular Depth Estimation** for the Image Processing & Computer Vision course. The project involves predicting depth zones (Near, Middle, Far) for four quadrants of an image using a custom multi-head convolutional neural network.

---

## 📂 Project Structure

```text
Task 4/
├── dataset_images/                     # Preprocessed training dataset images
├── presentation_assets/               # Generated graphs and plots for reports
├── check.py                            # Programmatic dataset cleaner & balancing script
├── evaluate.py                         # Evaluation script (outputs classification reports)
├── generate_metrics_charts.py          # Script generating convergence and validation graphs
├── gtrain.py                           # Training pipeline script (ResNet-18 Multi-Head CNN)
├── label_images.py                     # Initial raw image labeling helper script
├── labeler.py                          # Interactive visual labeling interface script
├── rename_images.py                    # Dataset file renaming script
├── run_complete_pipeline.py            # Main script to run processing, training, and evaluation
├── split_dataset.py                    # Train/Val/Test random split utility script
├── dataset.csv                         # Programmatically cleaned labels database
├── labels.csv                          # Raw annotations database
├── test.csv / train.csv / val.csv       # Split dataset indexes
├── image_001.jpg ... image_070.jpg     # Raw captured images
├── Monocular Depth Estimation Task...  # Project guidelines PDF
└── README.md                           # This documentation
```

---

## 📊 Practical Report Details

### A. Short Summary
- **Goal**: Formulate monocular depth estimation as a quadrant-based classification task. Predict whether each quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right) contains objects at **Near (N)**, **Middle (M)**, or **Far (F)** distances.
- **Approach**: Built a multi-head CNN classifier using a **ResNet-18** backbone. The backbone output is fed into 4 independent fully-connected heads (one for each quadrant).
- **Result**: The model converges successfully, achieving stable validation accuracy across all quadrants.

### B. Data Collection
- **Capture Setup**: Captured 70 custom tabletop setup images containing multiple objects placed at controlled distances (measured ranges for Near, Middle, and Far zones).
- **Dataset Description**: Includes visual variations in lighting conditions, object arrangements, partial occlusions, and background clutter.
- **Splits**: 70% Train, 15% Val, 15% Test.

### C. Preprocessing
- Images resized to `224x224` pixels.
- Normalized using ImageNet channel means (`[0.485, 0.456, 0.406]`) and standard deviations (`[0.229, 0.224, 0.225]`).

### D. Model & Training
- **Model**: Custom `DepthCNN` utilizing a pre-trained **ResNet-18** feature extractor.
- **Heads**: 4 parallel heads, each consisting of: `Linear(512 -> 128) -> ReLU -> Dropout(0.5) -> Linear(128 -> 3 classes)`.
- **Loss**: Cross-Entropy Loss summed across all four heads.
- **Optimizer**: Adam with learning rate scheduler.

---

## 🚀 Execution Instructions

### Prerequisite
Use the root environment (`TASKS\.venv`) to run the scripts.

### 1. Preprocess & Clean Dataset
Run the data cleaning script to balance quadrant distributions:
```bash
python check.py
```

### 2. Train the Model
Train the multi-head CNN:
```bash
python run_complete_pipeline.py
```
*(This script will train the model, save `best_model.pth`, and output evaluation metrics)*

### 3. Evaluate the Model
Evaluate performance on the test set:
```bash
python evaluate.py
```
