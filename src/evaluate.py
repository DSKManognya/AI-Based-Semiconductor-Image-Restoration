import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from skimage.metrics import structural_similarity

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR
from dataset import SemiconductorDataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = "checkpoints/nafnet_sr_best.pth"

MODEL_WIDTH = 32
NUM_BLOCKS = 4

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

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(val_dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

model = NAFNetSR(
    up_scale=2,
    width=MODEL_WIDTH,
    num_blks=NUM_BLOCKS,
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
# Metric functions
# --------------------------------------------------

def calculate_psnr(pred, target):

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean((pred - target) ** 2)

    if mse.item() == 0:
        return float("inf")

    return (
        10 * torch.log10(1.0 / mse)
    ).item()


def calculate_ssim(pred, target):

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    return structural_similarity(
        target,
        pred,
        data_range=1.0
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

bilinear_psnr = 0.0
bilinear_ssim = 0.0

nafnet_psnr = 0.0
nafnet_ssim = 0.0


with torch.no_grad():

    for i, (noisy, gt) in enumerate(val_loader):

        noisy = noisy.to(device)
        gt = gt.to(device)

        # ------------------------------------------
        # Bilinear baseline
        # ------------------------------------------

        bilinear = F.interpolate(
            noisy,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        bilinear_psnr += calculate_psnr(
            bilinear,
            gt
        )

        bilinear_ssim += calculate_ssim(
            bilinear,
            gt
        )

        # ------------------------------------------
        # NAFNet
        # ------------------------------------------

        output = model(noisy)

        nafnet_psnr += calculate_psnr(
            output,
            gt
        )

        nafnet_ssim += calculate_ssim(
            output,
            gt
        )

        if (i + 1) % 50 == 0:
            print(
                f"Evaluated {i + 1}/{len(val_loader)}"
            )


# --------------------------------------------------
# Average results
# --------------------------------------------------

n = len(val_loader)

bilinear_psnr /= n
bilinear_ssim /= n

nafnet_psnr /= n
nafnet_ssim /= n


print("\n==============================")
print("FINAL VALIDATION RESULTS")
print("==============================")

print(
    f"Bilinear PSNR : {bilinear_psnr:.4f} dB"
)

print(
    f"Bilinear SSIM : {bilinear_ssim:.4f}"
)

print(
    f"NAFNet PSNR   : {nafnet_psnr:.4f} dB"
)

print(
    f"NAFNet SSIM   : {nafnet_ssim:.4f}"
)

print(
    f"\nPSNR improvement : "
    f"{nafnet_psnr - bilinear_psnr:+.4f} dB"
)

print(
    f"SSIM improvement : "
    f"{nafnet_ssim - bilinear_ssim:+.4f}"
)
