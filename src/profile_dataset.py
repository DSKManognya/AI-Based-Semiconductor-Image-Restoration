import os
import numpy as np

GT_PATH = "Dataset/train/GT"
NOISY_PATH = "Dataset/train/NoisyLR"

gt_files = sorted(os.listdir(GT_PATH))
noisy_files = sorted(os.listdir(NOISY_PATH))

print("Number of GT images:", len(gt_files))
print("Number of Noisy images:", len(noisy_files))

print("\nChecking first 10 image pairs...\n")

for i in range(10):
    gt = np.load(os.path.join(GT_PATH, gt_files[i]))
    noisy = np.load(os.path.join(NOISY_PATH, noisy_files[i]))

    print(f"Pair {i}:")
    print(f"  GT    shape={gt.shape}, dtype={gt.dtype}, "
          f"range=({gt.min():.6f}, {gt.max():.6f})")

    print(f"  Noisy shape={noisy.shape}, dtype={noisy.dtype}, "
          f"range=({noisy.min():.6f}, {noisy.max():.6f})")