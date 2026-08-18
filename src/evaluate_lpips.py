import sys
import numpy as np
import torch
import torch.nn.functional as F
import lpips

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR
from dataset import SemiconductorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = "checkpoints/nafnet_sr_baseline.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SemiconductorDataset(
    noisy_dir="Dataset/train/NoisyLR",
    gt_dir="Dataset/train/GT"
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

_, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Validation images:", len(val_dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

model = NAFNetSR(
    up_scale=2,
    width=32,
    num_blks=4,
    img_channel=1,
    dual=False
).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint:", CHECKPOINT)


# --------------------------------------------------
# LPIPS model
# --------------------------------------------------

loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

print("LPIPS model loaded.")


# --------------------------------------------------
# Metrics
# --------------------------------------------------

from skimage.metrics import structural_similarity


def calculate_psnr(pred, target):

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    return (
        10 * torch.log10(1.0 / mse)
    ).item()


def calculate_ssim(pred, target):

    pred_np = pred.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()

    return structural_similarity(
        target_np,
        pred_np,
        data_range=1.0
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0

with torch.no_grad():

    for i in range(len(val_dataset)):

        noisy, gt = val_dataset[i]

        noisy = noisy.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        # NAFNetSR restoration
        restored = model(noisy)

        restored = torch.clamp(restored, 0.0, 1.0)

        # PSNR / SSIM
        total_psnr += calculate_psnr(
            restored,
            gt
        )

        total_ssim += calculate_ssim(
            restored,
            gt
        )

        # LPIPS expects 3-channel images
        restored_rgb = restored.repeat(1, 3, 1, 1)
        gt_rgb = gt.repeat(1, 3, 1, 1)

        # LPIPS expects [-1, 1]
        restored_rgb = restored_rgb * 2.0 - 1.0
        gt_rgb = gt_rgb * 2.0 - 1.0

        lpips_value = loss_fn(
            restored_rgb,
            gt_rgb
        ).item()

        total_lpips += lpips_value

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(val_dataset)}"
            )


# --------------------------------------------------
# Final results
# --------------------------------------------------

avg_psnr = total_psnr / len(val_dataset)
avg_ssim = total_ssim / len(val_dataset)
avg_lpips = total_lpips / len(val_dataset)

print()
print("================================")
print("FINAL VALIDATION RESULTS")
print("================================")

print(f"Images evaluated : {len(val_dataset)}")
print(f"PSNR              : {avg_psnr:.4f} dB")
print(f"SSIM              : {avg_ssim:.4f}")
print(f"LPIPS             : {avg_lpips:.4f}")