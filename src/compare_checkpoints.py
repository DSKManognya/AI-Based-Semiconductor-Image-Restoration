import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR
from dataset import SemiconductorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASELINE_CHECKPOINT = "checkpoints/nafnet_sr_baseline.pth"
EDGE_CHECKPOINT = "checkpoints/nafnet_sr_edge_01.pth"

OUTPUT_DIR = "outputs/experiment_comparisons"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


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
# Model loader
# --------------------------------------------------

def load_model(checkpoint_path):

    model = NAFNetSR(
        up_scale=2,
        width=32,
        num_blks=4,
        img_channel=1,
        dual=False
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded: {checkpoint_path} | "
        f"PSNR: {checkpoint.get('psnr', 'N/A')} | "
        f"SSIM: {checkpoint.get('ssim', 'N/A')}"
    )

    return model


baseline_model = load_model(
    BASELINE_CHECKPOINT
)

edge_model = load_model(
    EDGE_CHECKPOINT
)


# --------------------------------------------------
# Visual comparisons
# --------------------------------------------------

indices = [0, 50, 100, 150, 200]

with torch.no_grad():

    for count, idx in enumerate(indices):

        noisy, gt = val_dataset[idx]

        noisy_input = noisy.unsqueeze(0).to(device)

        gt_np = gt.squeeze().numpy()
        noisy_np = noisy.squeeze().numpy()

        # ------------------------------------------
        # Bilinear
        # ------------------------------------------

        bilinear = F.interpolate(
            noisy_input,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # ------------------------------------------
        # Baseline NAFNet
        # ------------------------------------------

        baseline_output = baseline_model(
            noisy_input
        )

        # ------------------------------------------
        # Edge-aware NAFNet
        # ------------------------------------------

        edge_output = edge_model(
            noisy_input
        )

        # ------------------------------------------
        # Convert to NumPy
        # ------------------------------------------

        bilinear_np = (
            bilinear.squeeze()
            .cpu()
            .numpy()
        )

        baseline_np = (
            baseline_output.squeeze()
            .cpu()
            .numpy()
        )

        edge_np = (
            edge_output.squeeze()
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # Clamp for display
        # ------------------------------------------

        noisy_display = np.clip(
            noisy_np, 0, 1
        )

        bilinear_display = np.clip(
            bilinear_np, 0, 1
        )

        baseline_display = np.clip(
            baseline_np, 0, 1
        )

        edge_display = np.clip(
            edge_np, 0, 1
        )

        gt_display = np.clip(
            gt_np, 0, 1
        )

        # ------------------------------------------
        # Plot
        # ------------------------------------------

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        axes[0].imshow(
            noisy_display,
            cmap="gray"
        )
        axes[0].set_title(
            "Noisy LR"
        )

        axes[1].imshow(
            bilinear_display,
            cmap="gray"
        )
        axes[1].set_title(
            "Bilinear"
        )

        axes[2].imshow(
            baseline_display,
            cmap="gray"
        )
        axes[2].set_title(
            "Baseline NAFNetSR"
        )

        axes[3].imshow(
            edge_display,
            cmap="gray"
        )
        axes[3].set_title(
            "Edge-Aware NAFNetSR"
        )

        axes[4].imshow(
            gt_display,
            cmap="gray"
        )
        axes[4].set_title(
            "Ground Truth"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{count + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output_path)


print()
print(
    "Experiment comparison images generated successfully."
)