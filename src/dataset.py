import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.noisy_files = sorted(
            [f for f in os.listdir(noisy_dir) if f.endswith(".npy")]
        )

        self.gt_files = sorted(
            [f for f in os.listdir(gt_dir) if f.endswith(".npy")]
        )

        assert len(self.noisy_files) == len(self.gt_files), \
            f"Dataset size mismatch: {len(self.noisy_files)} noisy vs {len(self.gt_files)} GT"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):

        noisy_path = os.path.join(
            self.noisy_dir,
            self.noisy_files[idx]
        )

        gt_path = os.path.join(
            self.gt_dir,
            self.gt_files[idx]
        )

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        # NumPy:
        # Noisy: [128, 128]
        # GT:    [256, 256]

        # PyTorch:
        # Noisy: [1, 128, 128]
        # GT:    [1, 256, 256]

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


if __name__ == "__main__":

    dataset = SemiconductorDataset(
        noisy_dir="Dataset/train/NoisyLR",
        gt_dir="Dataset/train/GT"
    )

    print("Dataset size:", len(dataset))

    noisy, gt = dataset[0]

    print("Noisy shape:", noisy.shape)
    print("GT shape:", gt.shape)

    print("Noisy dtype:", noisy.dtype)
    print("GT dtype:", gt.dtype)

    print(
        "Noisy range:",
        noisy.min().item(),
        "to",
        noisy.max().item()
    )

    print(
        "GT range:",
        gt.min().item(),
        "to",
        gt.max().item()
    )
    