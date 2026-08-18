import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "NAFNet"))

from basicsr.models.archs.NAFSSR_arch import NAFNetSR


def main():
    parser = argparse.ArgumentParser(
        description="KLA Semiconductor Image Restoration Inference"
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing degraded .npy images"
    )

    parser.add_argument(
        "output_dir",
        help="Directory where restored .npy images will be saved"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "nafnet_sr_baseline.pth"
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

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

    test_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".npy")
    )

    print("Input images:", len(test_files))

    if len(test_files) == 0:
        raise RuntimeError(
            "No .npy files found in the input directory."
        )

    processed = 0

    with torch.no_grad():

        for filename in test_files:

            input_path = os.path.join(
                input_dir,
                filename
            )

            output_path = os.path.join(
                output_dir,
                filename
            )

            image = np.load(input_path).astype(
                np.float32
            )

            if image.ndim != 2:
                raise ValueError(
                    f"{filename}: expected grayscale array "
                    f"with shape (H, W), got {image.shape}"
                )

            if not np.isfinite(image).all():
                raise ValueError(
                    f"{filename}: input contains NaN or Inf values"
                )

            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            restored = model(image_tensor)

            restored = torch.clamp(
                restored,
                0.0,
                1.0
            )

            restored = (
                restored
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if restored.ndim != 2:
                raise ValueError(
                    f"{filename}: unexpected output shape "
                    f"{restored.shape}"
                )

            if not np.isfinite(restored).all():
                raise ValueError(
                    f"{filename}: output contains NaN or Inf values"
                )

            if restored.min() < 0.0 or restored.max() > 1.0:
                raise ValueError(
                    f"{filename}: output values outside [0, 1]"
                )

            np.save(
                output_path,
                restored
            )

            processed += 1

            if processed % 50 == 0:
                print(
                    f"Processed {processed}/{len(test_files)}"
                )

    print()
    print("INFERENCE COMPLETED")
    print("Input images :", len(test_files))
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()