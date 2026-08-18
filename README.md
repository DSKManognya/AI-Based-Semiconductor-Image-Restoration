# AI-Based Restoration of Degraded Semiconductor Inspection Images

## About the Project

Semiconductor inspection images can be difficult to analyze when noise and loss of resolution hide small features and defects. The goal of this project is to recover a cleaner, higher-resolution image from a degraded inspection image.

We use **NAFNetSR**, an open-source image restoration and super-resolution architecture, and adapt it to the single-channel semiconductor images provided for this challenge. The model is trained directly on the provided degraded-image and ground-truth pairs.

---

## Problem

The input images are:

- Grayscale
- 128 × 128 pixels
- Affected by Gaussian noise, speckle noise, and reduced spatial resolution

The corresponding ground-truth images are:

- Grayscale
- 256 × 256 pixels
- Higher-quality reference images

The task can be summarized as:

```text
128 × 128 degraded image
          │
          ▼
      NAFNetSR
          │
          ▼
256 × 256 restored image
```

---

## Our Approach

We use NAFNetSR as the main restoration model and train it on the paired semiconductor images provided for the challenge.

The model takes a single-channel 128 × 128 degraded image and produces a 256 × 256 restored image. Since the input already contains the combined degradation, the model learns the restoration mapping directly from the degraded image to its corresponding ground truth.

### Training Setup

- Dataset: 3,200 paired images
- Training images: 2,880
- Validation images: 320
- Train/validation split: 90/10
- Input: 1 × 128 × 128
- Output: 1 × 256 × 256
- Upscaling factor: 2×
- Model width: 32
- NAF blocks: 4
- Loss: L1 Loss
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 1
- Training epochs: 10
- Random seed: 42
- Data augmentation: None in the final training setup

---

## Evaluation

The final model was evaluated on the 320-image validation split using PSNR, SSIM and LPIPS.

| Metric | Result |
|---|---:|
| PSNR | **27.3066 dB** |
| SSIM | **0.7224** |
| LPIPS | **0.3342** |

For comparison, we also evaluated a simple 2× bilinear interpolation baseline:

| Method | PSNR | SSIM |
|---|---:|---:|
| Bilinear interpolation | 24.7825 dB | 0.6045 |
| **NAFNetSR** | **27.3066 dB** | **0.7224** |

The NAFNetSR pipeline improved the validation result by **2.5241 dB in PSNR** and **0.1179 in SSIM** compared with bilinear interpolation.

---

## Experiments

We tested several changes to the baseline training setup rather than assuming that the first configuration was automatically the best one.

| Experiment | Best PSNR | Best SSIM |
|---|---:|---:|
| L1 baseline | **27.3066 dB** | 0.7224 |
| Cosine learning-rate schedule | 27.1418 dB | 0.7215 |
| Charbonnier loss | 27.3001 dB | 0.7221 |
| Edge-aware loss (λ = 0.10) | 27.3129 dB | **0.7274** |
| Edge-aware loss (λ = 0.05) | 27.3032 dB | **0.7275** |

The edge-aware experiments produced small improvements in some metrics, but the difference was not large enough to justify replacing the simpler baseline. We therefore selected the L1-trained baseline as the final model.

More details about these experiments are recorded in `experiments.md`.

---

## Test Set

The final model was run on all **400 unseen test images**.

The challenge test set does not contain ground-truth images, so PSNR, SSIM and LPIPS cannot be calculated for these images. The restored outputs are therefore provided for visual evaluation.

All 400 test images were successfully processed.

---

## Inference Speed

Inference was benchmarked on an:

**NVIDIA GeForce RTX 3050 6GB Laptop GPU**

For the 400-image test set:

- Total inference time: **14.165 seconds**
- Average inference time: **35.41 ms/image**
- Throughput: **28.24 images/second**

A short GPU warm-up was performed before measuring inference time.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/DSKManognya/AI-Based-Semiconductor-Image-Restoration.git
cd AI-Based-Semiconductor-Image-Restoration
```

### 2. Install dependencies

Create and activate a Python virtual environment if needed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run inference

The inference script accepts three arguments:

- `--input_dir`: directory containing degraded `.npy` images
- `--output_dir`: directory where restored images will be saved
- `--checkpoint`: path to the trained model checkpoint

Example:

```bash
python src/inference.py \
    --input_dir Dataset/test/NoisyLR \
    --output_dir outputs/test_predictions \
    --checkpoint checkpoints_submission/nafnet_sr_baseline.pth
```

For Windows PowerShell:

```powershell
python src\inference.py `
    --input_dir Dataset\test\NoisyLR `
    --output_dir outputs\test_predictions `
    --checkpoint checkpoints_submission\nafnet_sr_baseline.pth
```

The script reads every `.npy` image from the input directory and saves the corresponding restored 256 × 256 `.npy` image to the output directory.

No changes to the Python source code are required.

---

## Training From Scratch

The training pipeline is provided in `train.py`.

Run:

```bash
python train.py
```

The training data should be arranged as:

```text
Dataset/
├── train/
│   ├── NoisyLR/
│   └── GT/
└── test/
    └── NoisyLR/
```

The training script creates the fixed 90/10 train-validation split and saves the best model based on validation PSNR.

---

## Repository Structure

```text
AI-Based-Semiconductor-Image-Restoration/
│
├── checkpoints_submission/
│   └── nafnet_sr_baseline.pth
│
├── NAFNet/
│   └── NAFSSR implementation
│
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── evaluate_lpips.py
│   ├── benchmark_inference.py
│   ├── compare_checkpoints.py
│   └── ...
│
├── train.py
├── experiments.md
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset and generated outputs are not stored directly in the Git repository because of their size.

---

## Final Model

The final submitted checkpoint is:

```text
checkpoints_submission/nafnet_sr_baseline.pth
```

The checkpoint is approximately **507 KB** and can be loaded directly by `src/inference.py`.

---

## Restored Test Outputs

The final model produced restored outputs for all 400 test images.

The complete restored test set contains **400 `.npy` files** and is approximately **91 MB when compressed**. It is provided separately from the Git repository because of its size.

---

## Open-Source Attribution

This project uses the publicly available **NAFNet/NAFSSR** implementation as its restoration backbone.

The NAFNetSR architecture itself was not developed by our team. Our work focused on adapting the architecture to the provided single-channel semiconductor dataset, building the dataset and training pipeline, running controlled experiments, evaluating the results, and preparing the inference pipeline for the challenge.

Please refer to the original NAFNet/NAFSSR work for the architecture and implementation.

---

## References

- NAFNet: *Simple Baselines for Image Restoration*
- NAFSSR: NAFNet-based image super-resolution work and implementation
- PyTorch
- scikit-image
- LPIPS: *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*

---

## Team

**BVRIT Hyderabad College of Engineering for Women**

Developed as part of the i4C / KLA hackathon challenge on AI-based restoration of degraded semiconductor inspection images.