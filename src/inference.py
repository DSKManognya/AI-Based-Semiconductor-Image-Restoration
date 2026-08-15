import sys
import os
import argparse

import numpy as np
import torch

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR


# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="NAFNetSR inference for semiconductor image restoration"
)

parser.add_argument(
    "--input_dir",
    required=True,
    help="Directory containing degraded .npy images"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory where restored .npy images will be saved"
)

parser.add_argument(
    "--checkpoint",
    required=True,
    help="Path to trained NAFNetSR checkpoint"
)

args = parser.parse_args()


INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
CHECKPOINT_PATH = args.checkpoint

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# Model
# --------------------------------------------------

model = NAFNetSR(
    up_scale=2,
    width=32,
    num_blks=4,
    img_channel=1,
    dual=False
)

model = model.to(device)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint:", CHECKPOINT_PATH)

if "psnr" in checkpoint:
    print("Checkpoint PSNR:", checkpoint["psnr"])

if "ssim" in checkpoint:
    print("Checkpoint SSIM:", checkpoint["ssim"])


# --------------------------------------------------
# Test images

test_files = sorted(
    [
        f
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".npy")
    ]
)

print("Input images:", len(test_files))



# Inference


with torch.no_grad():

    for i, filename in enumerate(test_files):

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        image = np.load(input_path).astype(
            np.float32
        )

        # [128,128]
        image_tensor = torch.from_numpy(image)

        # [128,128] -> [1,128,128]
        image_tensor = image_tensor.unsqueeze(0)

        # [1,128,128] -> [1,1,128,128]
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

        # Model inference
        output = model(image_tensor)

        # Keep output in valid image range
        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        # [1,1,256,256] -> [256,256]
        output = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        np.save(
            output_path,
            output
        )

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(test_files)}"
            )


print()

print("INFERENCE COMPLETED")

print("Input images :", len(test_files))
print("Output folder:", OUTPUT_DIR)