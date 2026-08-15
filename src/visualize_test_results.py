import os
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = "Dataset/test/NoisyLR"
OUTPUT_DIR = "outputs/test_predictions"
VIS_DIR = "outputs/test_comparisons"

os.makedirs(VIS_DIR, exist_ok=True)


# Five representative test images
TEST_FILES = [
    "000000.npy",
    "000001.npy",
    "000002.npy",
    "000003.npy",
    "000004.npy",
]


# --------------------------------------------------
# Bilinear interpolation
# --------------------------------------------------

def bilinear_resize(image, target_height, target_width):

    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(image).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized = F.interpolate(
        tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False
    )

    return resized.squeeze().numpy()


# --------------------------------------------------
# Generate comparisons
# --------------------------------------------------

for idx, filename in enumerate(TEST_FILES):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    lr = np.load(input_path).astype(np.float32)

    nafnet = np.load(output_path).astype(np.float32)

    bilinear = bilinear_resize(
        lr,
        256,
        256
    )

    # Keep displayed values in image range
    lr_display = np.clip(lr, 0.0, 1.0)
    bilinear_display = np.clip(bilinear, 0.0, 1.0)
    nafnet_display = np.clip(nafnet, 0.0, 1.0)


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        lr_display,
        cmap="gray"
    )

    axes[0].set_title(
        "Test LR Input\n128 × 128"
    )

    axes[1].imshow(
        bilinear_display,
        cmap="gray"
    )

    axes[1].set_title(
        "Bilinear ×2\n256 × 256"
    )

    axes[2].imshow(
        nafnet_display,
        cmap="gray"
    )

    axes[2].set_title(
        "NAFNetSR ×2\n256 × 256"
    )


    for ax in axes:
        ax.axis("off")


    plt.tight_layout()


    output_filename = os.path.join(
        VIS_DIR,
        f"test_comparison_{idx + 1}.png"
    )

    plt.savefig(
        output_filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_filename}"
    )


print()
print("Test visual comparisons generated successfully.")