# Contributing to BraTS Medical AI & 3D Spatial Mesh Engine

Thank you for your interest in contributing to this clinical medical AI and spatial computing project!

## Development Guidelines

1. **Code Standards:**
   - Follow PEP 8 style for Python code (`black`, `flake8`).
   - Use type annotations for all core pipeline functions.
   - Maintain reproducible random seeds in training scripts.

2. **Model Architectures & Training:**
   - When introducing new neural network backbones (e.g. SwinUNETR, nnU-Net, 3D Attention U-Net), place modules under `models/` and export a unified inference interface matching `models.inference_engine.BraTSInferenceEngine`.
   - Always evaluate models across all three BraTS sub-region classes:
     - **Whole Tumor (WT):** ED + NCR + ET
     - **Tumor Core (TC):** NCR + ET
     - **Enhancing Tumor (ET):** Active enhancing rim

3. **3D Reconstruction & Mesh Processing:**
   - Ensure mesh outputs preserve medical coordinate space (RAS orientation).
   - Test decimation ratios and Laplacian smoothing parameters to avoid topological self-intersections or volumetric shrinkage.

4. **Pull Request Process:**
   - Fork the repository and create a descriptive feature branch (`feature/3d-transformer-backbone` or `fix/mesh-normals`).
   - Run tests before opening a PR:
     ```bash
     python test_ai_prediction.py
     python test_reconstruction.py
     ```
   - Provide visual benchmarks or metric deltas in your PR description.

## Clinical Disclaimer
This system is intended for research, neurosurgical planning visualization, and educational purposes. It is not an FDA-cleared diagnostic device.
