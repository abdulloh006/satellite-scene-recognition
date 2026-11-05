"""Week 2: visualise HOG features at three cell sizes.

Smaller cells capture finer texture but give a much longer feature vector.

    python week2_handcrafted/hog_visualize.py
    python week2_handcrafted/hog_visualize.py --image samples/x.jpg
"""
import argparse

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import hog

from common import OUT_DIR, load_gray

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--data-root", default="./data")
args = ap.parse_args()

img, name = load_gray(args.image, args.size, args.data_root)
cell_sizes = [4, 8, 16]

fig, axes = plt.subplots(1, len(cell_sizes) + 1, figsize=(4 * (len(cell_sizes) + 1), 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"Input {img.shape[0]}x{img.shape[1]}")
for ax, c in zip(axes[1:], cell_sizes):
    feats, vis = hog(img, orientations=9, pixels_per_cell=(c, c), cells_per_block=(2, 2),
                     block_norm="L2-Hys", visualize=True)
    print(f"cell {c:>2}x{c:<2} -> feature vector length {feats.shape[0]:,}")
    ax.imshow(exposure.rescale_intensity(vis, in_range=(0, vis.max() / 2)), cmap="gray")
    ax.set_title(f"HOG, {c}x{c} cells ({feats.shape[0]:,} dims)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / f"hog_{name}.png"
plt.savefig(out, dpi=150)
print("Saved", out)
plt.show()
