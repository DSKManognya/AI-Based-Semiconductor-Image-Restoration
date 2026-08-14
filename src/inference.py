import sys
import numpy as np
import torch

sys.path.append("NAFNet")

from basicsr.models.archs.NAFSSR_arch import NAFNetSR


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Create single-image NAFNetSR
model = NAFNetSR(
    up_scale=2,
    width=32,
    num_blks=4,
    img_channel=1,
    dual=False
)

model = model.to(device)
model.eval()

print("NAFNetSR created successfully!")


# Load one degraded image
image_path = "Dataset/train/NoisyLR/000000.npy"

image = np.load(image_path)

print("Input NumPy shape:", image.shape)
print("Input dtype:", image.dtype)
print("Input range:", image.min(), "to", image.max())


# NumPy → PyTorch
image_tensor = torch.from_numpy(image).float()

# [128,128] → [1,128,128]
image_tensor = image_tensor.unsqueeze(0)

# [1,128,128] → [1,1,128,128]
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)

print("Input tensor shape:", image_tensor.shape)


# Inference
with torch.no_grad():
    output = model(image_tensor)


print("Output tensor shape:", output.shape)
print(
    "Output range:",
    output.min().item(),
    "to",
    output.max().item()
)


# Save output
output = output.squeeze().cpu().numpy()

np.save("outputs/nafnet_sr_test.npy", output)

print("Saved output to: outputs/nafnet_sr_test.npy")
