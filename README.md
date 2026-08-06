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

## Authors

- Durvasula Manognya
- DeepSaanvi

## License

This repository has been developed as part of the AI-Based Restoration of Degraded Images for Semiconductor Inspection Hackathon.