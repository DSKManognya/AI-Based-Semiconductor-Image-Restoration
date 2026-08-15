# AI-Based Restoration of Degraded Semiconductor Inspection Images

## About the Project

Semiconductor inspection images can lose detail because of noise and reduced spatial resolution. This makes it harder to see small features and defects clearly.

In this project, we use a deep learning-based image restoration model to take a degraded low-resolution semiconductor image and reconstruct a cleaner, higher-resolution version.

We use **NAFNetSR** as the main restoration backbone and adapt it for our single-channel semiconductor images.

---

## What We Are Trying to Do

Our input images are:

- Grayscale
- 128 × 128 pixels
- Affected by noise and resolution degradation

The corresponding ground-truth images are:

- Grayscale
- 256 × 256 pixels
- Clean reference images

So the main task is:

```text
128 × 128 degraded image
          ↓
       NAFNetSR
          ↓
256 × 256 restored image