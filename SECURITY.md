# Security Policy

## Supported Versions

We actively maintain and provide security patches for the current release of **Monocular Depth Estimation & Spatial 4-Quadrant Hazard Risk Analysis**.

| Version | Supported |
| :--- | :---: |
| 1.0.x | :white_check_mark: |
| < 1.0.0 | :x: |

---

## Threat Model & Considerations

Please observe the following security considerations when deploying this monocular depth and spatial risk analysis system:

1. **Model Checkpoint Deserialization**: PyTorch checkpoint loading (`torch.load` via `.pth` files) utilizes Python pickle, which can execute arbitrary code if model weights originate from untrusted sources. Only load checkpoint weights from verified, trusted sources and always employ safe deserialization (`torch.load(..., weights_only=True)`).
2. **Image Ingestion & Decompression Safety**: Image loading through PIL (`Image.open`) or OpenCV (`cv2.imread`) should be guarded against decompression bomb vulnerabilities and malformed image file headers.
3. **Adversarial Depth Attacks in Autonomous Driving**: In safety-critical obstacle avoidance systems, adversarial physical patterns or optical projector attacks on road surfaces could manipulate predicted depth or suppress proximity hazard warnings. Systems should utilize multi-sensor fusion (camera + LiDAR / Radar) and temporal persistence filtering to validate depth predictions before triggering automated emergency braking.
4. **Supply Chain Integrity**: Ensure PyTorch, Torchvision, CUDA, and OpenCV dependencies are kept up to date with upstream security patches.

---

## Reporting a Vulnerability

If you identify a security issue or vulnerability within this project, please follow responsible disclosure:

1. **Do not create a public issue**: Refrain from submitting publicly accessible bug reports for potential security exploits.
2. **Contact Maintainer**: Reach out to the project maintainer via their GitHub profile at [gbhanuvigneshnaidu29052002-droid](https://github.com/gbhanuvigneshnaidu29052002-droid) or submit a private security advisory through the GitHub repository's **Security** tab.
3. **Provide Detailed Information**:
   - Description of the vulnerability and its potential exploit vectors
   - Affected modules (`src/models.py`, `src/dataset.py`, `main.py`, dependencies)
   - Step-by-step reproduction instructions or Proof-of-Concept (PoC) script

We will acknowledge reports within 48 hours and coordinate a timely resolution.
