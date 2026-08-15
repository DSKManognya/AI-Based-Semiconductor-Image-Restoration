# Experiment Log

## E0 - Baseline

### Configuration

- Model: NAFNetSR
- Upscaling: 2×
- Input channels: 1
- Width: 32
- Blocks: 4
- Dataset: 3200 paired samples
- Training split: 2880
- Validation split: 320
- Batch size: 1
- Epochs: 10
- Loss: L1
- Optimizer: Adam
- Learning rate: 1e-4
- Scheduler: None
- Augmentation: None
- Normalization: None
- Random seed: 42

### Results

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Bilinear ×2 | 24.7825 | 0.6045 |
| NAFNetSR | 27.3066 | 0.7224 |

Improvement over bilinear:

- PSNR: +2.5241 dB
- SSIM: +0.1179

**Decision:** Baseline retained.

---

## E1 - Cosine Learning Rate

### Hypothesis

A gradually decreasing learning rate may improve convergence during the 10-epoch training budget.

### Change

Added:

- CosineAnnealingLR
- T_max = 10
- eta_min = 1e-6

All other baseline settings were unchanged.

### Results

| Metric | Baseline | E1 |
|---|---:|---:|
| PSNR | 27.3066 | 27.1418 |
| SSIM | 0.7224 | 0.7215 |

**Decision:** Rejected.

**Conclusion:** Cosine learning-rate scheduling reduced performance under the current 10-epoch training configuration.

---

## E2 - Charbonnier Loss

### Hypothesis

Charbonnier loss may provide more robust image restoration and improve structural similarity.

### Change

Replaced L1 loss with Charbonnier loss.

All other baseline settings were unchanged.

### Results

| Metric | Baseline | E2 |
|---|---:|---:|
| Best PSNR | 27.3066 | 27.3001 |
| Best SSIM | 0.7224 | 0.7263 |

**Decision:** Baseline retained.

**Conclusion:** Charbonnier loss produced slightly higher SSIM but marginally lower PSNR. Since the baseline achieved the strongest PSNR and the SSIM difference was small, the L1 baseline was retained as the final model.