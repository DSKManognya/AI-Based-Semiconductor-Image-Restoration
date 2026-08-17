import sys
import os
import time

import numpy as np
import torch

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = "Dataset/test/NoisyLR"
CHECKPOINT = "checkpoints/nafnet_sr_baseline.pth"

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
# Input files
# --------------------------------------------------

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".npy")
)

print("Test images:", len(files))


# --------------------------------------------------
# Warm-up
# --------------------------------------------------

warmup_image = np.load(
    os.path.join(INPUT_DIR, files[0])
).astype(np.float32)

warmup_tensor = (
    torch.from_numpy(warmup_image)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)

with torch.no_grad():
    for _ in range(10):
        _ = model(warmup_tensor)

if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Warm-up completed.")


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

start_time = time.perf_counter()

with torch.no_grad():

    for filename in files:

        image = np.load(
            os.path.join(INPUT_DIR, filename)
        ).astype(np.float32)

        image_tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        _ = model(image_tensor)

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.perf_counter()


# --------------------------------------------------
# Results
# --------------------------------------------------

total_time = end_time - start_time

num_images = len(files)

average_time = total_time / num_images

images_per_second = num_images / total_time

print()
print("================================")
print("INFERENCE SPEED BENCHMARK")
print("================================")

print(f"Images processed : {num_images}")
print(f"Total time       : {total_time:.3f} seconds")
print(f"Average / image  : {average_time * 1000:.2f} ms")
print(f"Throughput       : {images_per_second:.2f} images/sec")