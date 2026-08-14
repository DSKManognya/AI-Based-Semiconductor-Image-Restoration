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

CHECKPOINT = "checkpoints/nafnet_sr_best.pth"
OUTPUT_DIR = "outputs/comparisons"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SemiconductorDataset(
    noisy_dir="Dataset/train/NoisyLR",
    gt_dir="Dataset/train/GT"
)

# Same validation split as training/evaluation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

_, val_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


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


# --------------------------------------------------
# Generate visual comparisons
# --------------------------------------------------

indices = [0, 50, 100, 150, 200]

with torch.no_grad():

    for count, idx in enumerate(indices):

        noisy, gt = val_dataset[idx]

        noisy_input = noisy.unsqueeze(0).to(device)
        gt_np = gt.squeeze().numpy()

        # Bilinear baseline
        bilinear = F.interpolate(
            noisy_input,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # NAFNet
        restored = model(noisy_input)

        noisy_np = noisy.squeeze().numpy()
        bilinear_np = bilinear.squeeze().cpu().numpy()
        restored_np = restored.squeeze().cpu().numpy()

        # Clamp only for visualization
        noisy_display = np.clip(noisy_np, 0, 1)
        bilinear_display = np.clip(bilinear_np, 0, 1)
        restored_display = np.clip(restored_np, 0, 1)
        gt_display = np.clip(gt_np, 0, 1)

        # ------------------------------------------
        # Plot
        # ------------------------------------------

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(noisy_display, cmap="gray")
        axes[0].set_title("Noisy LR (128×128)")

        axes[1].imshow(bilinear_display, cmap="gray")
        axes[1].set_title("Bilinear (256×256)")

        axes[2].imshow(restored_display, cmap="gray")
        axes[2].set_title("NAFNetSR (256×256)")

        axes[3].imshow(gt_display, cmap="gray")
        axes[3].set_title("Ground Truth")

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


print("\nVisual comparisons generated successfully.")