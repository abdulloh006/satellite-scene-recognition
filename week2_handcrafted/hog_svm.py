"""Week 2: hand-crafted features + SVM on EuroSAT.

Question: for satellite tiles, does shape (HOG) or colour matter more?
Three feature sets are compared:
  1. HOG only (edges and texture, grayscale)
  2. HSV colour histogram only
  3. HOG + colour concatenated
each with a linear SVM, plus an RBF SVM on the best set.

    python week2_handcrafted/hog_svm.py --per-class 300
"""
import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import OUT_DIR
from scenelab.data import CLASSES, load_eurosat, per_class_subset
from scenelab.metrics import confusion, top_confusions

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="./data")
ap.add_argument("--per-class", type=int, default=300, help="images per class to use")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


def hog_features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys")


def colour_features(rgb, bins=16):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]  # OpenCV hue runs 0-179
    hists = [np.histogram(hsv[..., c], bins=bins, range=r)[0] for c, r in enumerate(ranges)]
    h = np.concatenate(hists).astype(np.float32)
    return h / h.sum()


ds = load_eurosat(args.data_root)
idx = per_class_subset(range(len(ds)), ds.targets, args.per_class, args.seed)
print(f"Extracting features from {len(idx)} tiles ({args.per_class} per class)...")
t0 = time.time()
H, C, y = [], [], []
for i in idx:
    img, label = ds[i]
    rgb = np.asarray(img.convert("RGB"))
    H.append(hog_features(rgb))
    C.append(colour_features(rgb))
    y.append(label)
H, C, y = np.array(H), np.array(C), np.array(y)
print(f"done in {time.time() - t0:.0f}s | HOG dims {H.shape[1]} | colour dims {C.shape[1]}")

feature_sets = {"HOG": H, "Colour (HSV hist)": C, "HOG + Colour": np.hstack([H, C])}
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y,
                                  random_state=args.seed)

results = {}
print(f"\n{'features':<20}{'classifier':<12}{'test acc':>9}{'fit time':>10}")
for name, X in feature_sets.items():
    clf = make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=20000))
    t0 = time.time()
    clf.fit(X[idx_tr], y[idx_tr])
    acc = accuracy_score(y[idx_te], clf.predict(X[idx_te]))
    results[(name, "linear")] = (acc, clf)
    print(f"{name:<20}{'linear SVM':<12}{acc:>9.1%}{time.time() - t0:>9.1f}s")

best_name = max(feature_sets, key=lambda n: results[(n, "linear")][0])
X = feature_sets[best_name]
rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
t0 = time.time()
rbf.fit(X[idx_tr], y[idx_tr])
acc = accuracy_score(y[idx_te], rbf.predict(X[idx_te]))
results[(best_name, "rbf")] = (acc, rbf)
print(f"{best_name:<20}{'RBF SVM':<12}{acc:>9.1%}{time.time() - t0:>9.1f}s")

(best_feat, best_kind), (best_acc, best_clf) = max(results.items(), key=lambda kv: kv[1][0])
y_pred = best_clf.predict(feature_sets[best_feat][idx_te])
print(f"\nBest: {best_feat} + {best_kind} SVM ({best_acc:.1%})\n")
print(classification_report(y[idx_te], y_pred, target_names=CLASSES, digits=3, zero_division=0))

cm = confusion(y[idx_te], y_pred, len(CLASSES))
print("Most common mistakes:")
for t, p, n in top_confusions(cm, CLASSES) or [("none", "none", 0)]:
    print(f"  {t} -> {p}: {n}")

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(10), CLASSES, fontsize=8)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"{best_feat} + {best_kind} SVM: {best_acc:.1%}")
plt.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUT_DIR / "hog_svm_confusion.png", dpi=150)
print("\nSaved", OUT_DIR / "hog_svm_confusion.png")
