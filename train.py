import sys
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR
from src.dataset import SemiconductorDataset

import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

MODEL_WIDTH = 32
NUM_BLOCKS = 4

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

full_dataset = SemiconductorDataset(
    noisy_dir="Dataset/train/NoisyLR",
    gt_dir="Dataset/train/GT"
)

print("Total dataset size:", len(full_dataset))


# 90% train / 10% validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = NAFNetSR(
    up_scale=2,
    width=MODEL_WIDTH,
    num_blks=NUM_BLOCKS,
    img_channel=1,
    dual=False
)

model = model.to(device)

print("Model created.")


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# PSNR
# --------------------------------------------------

def calculate_psnr(pred, target):

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr.item()


# --------------------------------------------------
# SSIM
# --------------------------------------------------

from skimage.metrics import structural_similarity


def calculate_ssim(pred, target):

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = pred.squeeze()
    target = target.squeeze()

    return structural_similarity(
        target,
        pred,
        data_range=1.0
    )


# --------------------------------------------------
# Training
# --------------------------------------------------



best_psnr = -float("inf")

print("\nStarting training...\n")

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    model.train()

    train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, gt)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(device)
            gt = gt.to(device)

            output = model(noisy)

            loss = criterion(output, gt)

            val_loss += loss.item()

            total_psnr += calculate_psnr(output, gt)

            total_ssim += calculate_ssim(output, gt)

    val_loss /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"PSNR: {avg_psnr:.4f} dB | "
        f"SSIM: {avg_ssim:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if avg_psnr > best_psnr:

        best_psnr = avg_psnr

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "nafnet_sr_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": avg_psnr,
                "ssim": avg_ssim
            },
            checkpoint_path
        )

        print(
            f"  Best model saved: {checkpoint_path}"
        )


print("\nTraining completed.")
print(f"Best validation PSNR: {best_psnr:.4f} dB")