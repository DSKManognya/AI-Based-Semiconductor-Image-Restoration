import numpy as np
import matplotlib.pyplot as plt

# Load images
gt = np.load("train/GT/000000.npy")
noisy = np.load("train/NoisyLR/000000.npy")

print("Ground Truth")
print("Shape:", gt.shape)
print("Datatype:", gt.dtype)
print("Min:", gt.min())
print("Max:", gt.max())

print("\nNoisy Image")
print("Shape:", noisy.shape)
print("Datatype:", noisy.dtype)
print("Min:", noisy.min())
print("Max:", noisy.max())

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Input")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.show()