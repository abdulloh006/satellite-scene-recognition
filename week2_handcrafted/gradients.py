"""Week 2: image gradients, the raw material of HOG.

Compares the simple [-1, 0, 1] derivative (what HOG uses) with the smoothed 3x3
Sobel filter, and shows magnitude and direction.

    python week2_handcrafted/gradients.py                    # uses a EuroSAT River tile
    python week2_handcrafted/gradients.py --image samples/x.jpg
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, load_gray

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--data-root", default="./data")
args = ap.parse_args()

img, name = load_gray(args.image, args.size, args.data_root)
f = img.astype(np.float32)

# Simple centred difference, as in Dalal & Triggs (2005)
gx = cv2.filter2D(f, -1, np.array([[-1, 0, 1]], np.float32))
gy = cv2.filter2D(f, -1, np.array([[-1], [0], [1]], np.float32))
mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# Sobel 3x3: the same derivative, smoothed across the other axis (less noise)
sx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
sy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sx, sy)

# HOG uses "unsigned" orientation: 10 degrees and 190 degrees are the same edge
unsigned = ang % 180
print(f"Image: {name} {img.shape}  | mean gradient magnitude {mag.mean():.1f}")
hist, _ = np.histogram(unsigned, bins=9, range=(0, 180), weights=mag)
print("Magnitude-weighted orientation histogram (9 bins, 0-180 deg):")
print(np.round(hist / hist.sum(), 3))

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
panels = [(img, "gray", "Original"), (np.abs(gx), "gray", "|Gx|"),
          (mag, "gray", "Magnitude [-1,0,1]"), (sobel_mag, "gray", "Magnitude Sobel 3x3"),
          (unsigned, "hsv", "Direction (0-180 deg)")]
for ax, (data, cmap, title) in zip(axes, panels):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / f"gradients_{name}.png"
plt.savefig(out, dpi=150)
print("Saved", out)
plt.show()
