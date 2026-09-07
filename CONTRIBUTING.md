# Contributing to Monocular Depth Estimation & Spatial Hazard Analysis

Thank you for your interest in contributing to **Monocular Depth Estimation & Spatial 4-Quadrant Hazard Risk Analysis**! We welcome contributions from computer vision researchers, autonomous vehicle perception engineers, robotics developers, and open-source practitioners.

Please read through this guide before submitting issues or pull requests.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainer.

---

## Areas of Contribution

We encourage contributions across several computer vision and autonomous robotics perception domains:

- **Backbone & Transformer Upgrades**: Implement Vision Transformer depth backbones (e.g. DPT, MiDaS v3.1, Depth Anything V2) to compare with ResNet-18.
- **Continuous Depth & PointCloud Generation**: Add camera intrinsic matrix ($K$) inversion to reproject 2D depth predictions into 3D metric PointCloud ($X, Y, Z$).
- **Robotics & ROS 2 Integration**: Implement a ROS 2 node publishing `/camera/depth_image` (`sensor_msgs/Image`) and `/hazard/quadrant_risk` (`std_msgs/Header` + risk vectors) for robot obstacle avoidance.
- **Sensor Fusion (RGB + LiDAR)**: Build sparse-to-dense depth completion pipelines fusing sparse LiDAR range scans with monocular camera images.
- **Edge Deployment & TensorRT**: Add ONNX export and TensorRT FP16/INT8 quantization benchmarks for NVIDIA Jetson platforms.

---

## Reporting Issues & Bugs

Before opening a new issue, please check existing [GitHub Issues](https://github.com/gbhanuvigneshnaidu29052002-droid/monocular-depth-estimation/issues) to avoid duplicates.

When reporting a bug:
1. Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
2. Specify your environment details:
   - OS: Linux (Ubuntu) / Windows / macOS
   - Python Version: Python 3.10+
   - PyTorch Version: PyTorch 2.x
   - Hardware: CPU or GPU (CUDA version, GPU model)
3. Include minimal reproduction steps and code snippets.
4. Attach full error tracebacks and terminal outputs.

---

## Development Workflow

### 1. Fork & Clone Repository
```bash
git clone https://github.com/<your-username>/monocular-depth-estimation.git
cd monocular-depth-estimation
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Run Automated Tests
Before submitting changes, ensure all unit tests pass cleanly:
```bash
python3 -m unittest discover -s tests -v
```

### 5. Commit & Push Changes
Follow conventional commit messages:
- `feat: Add Depth Anything V2 backbone encoder`
- `fix: Correct coordinate swap logic in horizontal flip augmentation`
- `perf: Accelerate quadrant overlay blending with vectorized OpenCV ops`
- `docs: Expand mathematical loss derivation in README`
- `test: Add unit test for scale-invariant depth loss`

```bash
git add .
git commit -m "feat: Describe your change cleanly"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
Submit your PR against the `main` branch using our [Pull Request Template](.github/pull_request_template.md).

---

## Code Style & Guidelines

- **PyTorch Idioms**: Write clean, modular `torch.nn.Module` classes with clear tensor shape documentation `(B, C, H, W)`.
- **Device Agnostic**: Ensure code runs seamlessly on both CPU and CUDA devices (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- **Reproducibility**: Set seeds (`torch.manual_seed`, `np.random.seed`) when adding tests or benchmark training routines.
- **Documentation**: Keep comments and docstrings clear and informative.

---

## Recognition

Contributors who have meaningful pull requests merged will be acknowledged in the project documentation. Thank you for contributing!
