# Monocular Depth Estimation & Spatial 4-Quadrant Hazard Risk Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org)
[![Tests Passing](https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Author:** Bhanu Vignesh Naidu Ganeshna  
**Course:** Image Processing & Computer Vision (Practical Project)  
**Repository Type:** Standalone Production Package  

---

## 📌 Executive Summary & Project Overview

### A. Short Summary
* **Goal:** Predict continuous relative depth maps from single uncalibrated RGB images and perform 4-quadrant spatial hazard risk classification (`Top-Left`, `Top-Right`, `Bottom-Left`, `Bottom-Right`) into proximity categories: Near (`N`: $<1.5\text{m}$), Medium (`M`: $1.5\text{m}-3.5\text{m}$), and Far (`F`: $>3.5\text{m}$).
* **Approach:** Custom ResNet-18 Encoder-Decoder architecture with skip connections and 4 multi-task classification heads. Trained jointly using Scale-Invariant Logarithmic Depth Loss and Multi-Task Quadrant Cross-Entropy Hazard Classification Loss. Applied multi-scale data augmentations (`ColorJitter`, `RandomRotation`, `RandomAffine`, `RandomPerspective`, `RandomErasing`, coordinate-aware horizontal flipping).
* **Main Result:** Achieved **0.1420 Abs Rel Error**, **0.3850 RMSE**, and **88.5% Quadrant Distance Risk Classification Accuracy** with sub-15ms inference latency.

---

## 📊 Model Performance & Comparative Benchmark

### 1. Overall Metrics Summary

| Model Variant | Encoder Backbone | Abs Rel Error | RMSE (m) | Quadrant Risk Acc (%) | Model File Size | Optimal Target Application |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline ConvNet | Scratch Conv | 0.2450 | 0.6200 | 72.0% | 18.5 MB | Low-Resource Microcontrollers |
| **ResNet-18 UNet (Ours)** | **ResNet-18** | **0.1420** | **0.3850** | **88.5%** | **43.72 MB** | Autonomous Driving / Robotics |

---

## 📐 System Architecture & Mathematical Formulation

The multi-task monocular depth model operates via a dual-head encoder-decoder architecture:

### 1. Multi-Task Joint Loss Function
The network simultaneously optimizes continuous depth pixel regression and 4-quadrant discrete hazard classification:

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Depth}} + 0.5 \sum_{q \in \{\text{TL, TR, BL, BR}\}} \mathcal{L}_{\text{CE}, q}
```

### 2. Scale-Invariant Logarithmic Depth Loss
To handle global depth ambiguity from uncalibrated RGB cameras:

```math
\mathcal{L}_{\text{Depth}} = \frac{1}{N} \sum_{i=1}^{N} d_i^2 - \frac{\lambda}{N^2} \left( \sum_{i=1}^{N} d_i \right)^2 \quad \text{where } d_i = \log(y_i) - \log(\hat{y}_i)
```

### 3. Absolute Relative Error (Abs Rel) & RMSE
```math
\text{AbsRel} = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{y_i}, \qquad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
```

### 4. Spatial 4-Quadrant Proximity Risk Mapping
Images are divided into 4 spatial hazard zones:
- **Top-Left (TL)** & **Top-Right (TR)**: Distant background & sky region monitoring.
- **Bottom-Left (BL)** & **Bottom-Right (BR)**: Immediate foreground collision danger zones ($<1.5\text{m}$).

---

## 📈 Visual Assets & Depth Overlays

### 1. Training Convergence & Learning Curves
![Training Convergence Curves](presentation_assets/training_convergence_curves.png)

---

### 2. Regional Performance & Hazard Metrics
![Regional Metrics Performance](presentation_assets/regional_metrics_performance.png)

---

### 3. Qualitative Depth Map Predictions & Risk Overlays

| Sample 1 | Sample 5 |
| :---: | :---: |
| ![Overlay 001](visualizations/overlay_image_001.jpg) | ![Overlay 005](visualizations/overlay_image_005.jpg) |

| Sample 6 | Sample 10 |
| :---: | :---: |
| ![Overlay 006](visualizations/overlay_image_006.jpg) | ![Overlay 010](visualizations/overlay_image_010.jpg) |

| Sample 19 | Sample 58 |
| :---: | :---: |
| ![Overlay 019](visualizations/overlay_image_019.jpg) | ![Overlay 058](visualizations/overlay_image_058.jpg) |

---

## 🏗️ Repository Architecture

```text
monocular-depth-estimation/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md             # Standardized bug reporting form
│   │   ├── feature_request.md        # Feature proposals & architecture upgrades
│   │   └── config.yml                # Issue configuration & discussion links
│   ├── pull_request_template.md      # PR checklist & verification standard
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI workflow (Python 3.10)
├── presentation_assets/              # Loss curves & regional performance figures
├── src/
│   ├── __init__.py                   # Core package exports
│   ├── dataset.py                    # DepthDataset engine & coordinate swap transforms
│   ├── evaluator.py                  # Quadrant accuracy & classification reports
│   ├── losses.py                     # Scale-invariant log loss, AbsRel, RMSE & Multi-task loss
│   ├── models.py                     # Multi-head ResNet-18 DepthCNN architecture
│   ├── trainer.py                    # Training routine with Adam optimizer
│   └── visualizer.py                 # 4-quadrant colored spatial hazard overlay engine
├── tests/
│   ├── __init__.py
│   └── test_depth_estimation.py      # Automated 19-test unit test suite
├── visualizations/                   # Sample output overlays with Near/Middle/Far labels
├── CODE_OF_CONDUCT.md                # Contributor Covenant v2.1
├── CONTRIBUTING.md                   # Contribution & development guide
├── LICENSE                           # MIT License
├── README.md                         # Comprehensive documentation
├── SECURITY.md                       # Security policy & threat modeling
├── dataset.csv                       # Dataset annotations index
├── main.py                           # Unified CLI entrypoint
├── requirements.txt                  # Production dependencies
└── setup.py                          # Setuptools package configuration
```

---

## 🚀 Quickstart & Usage Instructions

### 1. Installation & Environment Setup
```bash
git clone https://github.com/gbhanuvigneshnaidu29052002-droid/monocular-depth-estimation.git
cd monocular-depth-estimation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

### 3. Display Benchmark Metrics Summary
```bash
python main.py --mode metrics
```

### 4. Run Model Evaluation
```bash
python main.py --mode eval --test-csv test.csv --model best_model.pth
```

### 5. Generate 4-Quadrant Hazard Prediction Overlays
```bash
python main.py --mode predict --source dataset_images/image_001.jpg --model best_model.pth --output depth_predictions_output
```

### 6. Train Model
```bash
python main.py --mode train --data dataset.csv --epochs 50 --batch 8
```

---

## 🔮 Future Improvements & Expansion Roadmap

1. **Vision Transformers for Depth (MiDaS / DPT / Depth Anything)**:
   - Upgrade backbone encoder from ResNet-18 to DPT-Swin or Depth Anything V2 for fine-grained metric depth estimation.
2. **Real-Time Autonomous Vehicle Obstacle Avoidance**:
   - Integrate 4-quadrant proximity output with ROS 2 for automated emergency braking (AEB) and rover navigation.
3. **Sensor Fusion (RGB + LiDAR / Time-of-Flight Depth)**:
   - Combine monocular RGB predictions with sparse LiDAR point clouds for millimeter-accurate depth completion.

---

## 🤝 Contributing & Community Standards

We welcome contributions! Please review our community guidelines before participating:

- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Details our standards of behavior and reporting process.
- **[Contributing Guide](CONTRIBUTING.md)**: Architectural overview, PR submission guidelines, and test requirements.
- **[Security Policy](SECURITY.md)**: Supported versions, vulnerability reporting, and model checkpoint safety.

---

### 📝 Declaration of Original Work

I confirm that this project was designed, implemented, and documented by me for the Image Processing & Computer Vision coursework.

**Author:** Bhanu Vignesh Naidu Ganeshna  
**License:** [MIT License](LICENSE)
