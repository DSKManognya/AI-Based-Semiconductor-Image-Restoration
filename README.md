# AI-Based Restoration of Degraded Semiconductor Inspection Images

## Problem Statement

This project focuses on restoring degraded semiconductor inspection images using deep learning. The input images are affected by Gaussian noise, speckle noise, and low-resolution degradation. The objective is to reconstruct high-quality images that closely match the corresponding Ground Truth (GT) images.

## Objectives

- Restore degraded semiconductor inspection images.
- Develop a robust deep learning-based image restoration pipeline.
- Evaluate the model using standard image restoration metrics.
- Provide a reproducible implementation for training and inference.

## Technology Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/DSKManognya/AI-Based-Semiconductor-Image-Restoration.git
```

Move into the project directory:

```bash
cd AI-Based-Semiconductor-Image-Restoration
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Workflow

```text
Noisy Image
      │
      ▼
Dataset Loader
      │
      ▼
Deep Learning Model
      │
      ▼
Restored Image
      │
      ▼
Evaluation (PSNR, SSIM, LPIPS)
```
## Dataset

### Training Dataset

- Ground Truth (GT)
- NoisyLR

### Test Dataset

- NoisyLR

## Project Structure

```text
AI-Based-Semiconductor-Image-Restoration/
│
├── Dataset/
│   ├── train/
│   └── test/
│
├── checkpoints/
├── models/
├── notebooks/
├── outputs/
├── src/
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Structure

```text
Dataset/
├── train/
│   ├── GT/
│   └── NoisyLR/
│
└── test/
    └── NoisyLR/
```
## Current Progress

| Task | Status |
|------|--------|
| Project Setup | Completed |
| Dataset Exploration | Completed |
| Dataset Validation | Completed |
| Baseline Model Selection | In Progress |
| Model Integration | Pending |
| Training | Pending |
| Evaluation | Pending |

## Repository Contents

- Dataset inspection and validation scripts
- Training pipeline
- Inference pipeline
- Evaluation scripts
- Model checkpoints
- Restored output images

## Evaluation Metrics

The final model will be evaluated using:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

