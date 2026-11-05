"""Shared helpers for the Week 2 scripts."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs"


def load_gray(image_path, size, data_root="./data"):
    """Read an image as grayscale. With no path, use the first EuroSAT River tile."""
    if image_path:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_path)
        name = Path(image_path).stem
    else:
        from scenelab.data import CLASSES, load_eurosat
        ds = load_eurosat(data_root)
        i = int(np.flatnonzero(np.asarray(ds.targets) == CLASSES.index("River"))[0])
        img = cv2.cvtColor(np.asarray(ds[i][0]), cv2.COLOR_RGB2GRAY)
        name = "eurosat_river"
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), name
