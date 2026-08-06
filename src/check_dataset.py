import os

# Dataset paths
train_gt_path = "Dataset/train/GT"
train_noisy_path = "Dataset/train/NoisyLR"
test_noisy_path = "Dataset/test/NoisyLR"

# Get filenames
gt_files = sorted(os.listdir(train_gt_path))
noisy_files = sorted(os.listdir(train_noisy_path))
test_files = sorted(os.listdir(test_noisy_path))

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Training GT Images      : {len(gt_files)}")
print(f"Training Noisy Images   : {len(noisy_files)}")
print(f"Testing Noisy Images    : {len(test_files)}")

print("\nFirst 5 GT files:")
print(gt_files[:5])

print("\nFirst 5 Noisy files:")
print(noisy_files[:5])

print("\nLast 5 GT files:")
print(gt_files[-5:])

print("\nLast 5 Noisy files:")
print(noisy_files[-5:])

# Check if filenames match
if gt_files == noisy_files:
    print("\n✅ Training dataset is perfectly paired!")
else:
    print("\n❌ Mismatch found between GT and NoisyLR files!")

    missing_in_gt = set(noisy_files) - set(gt_files)
    missing_in_noisy = set(gt_files) - set(noisy_files)

    if missing_in_gt:
        print("\nFiles missing in GT:")
        print(sorted(missing_in_gt))

    if missing_in_noisy:
        print("\nFiles missing in NoisyLR:")
        print(sorted(missing_in_noisy))

print("\nDataset check completed.")