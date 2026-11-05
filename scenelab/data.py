"""EuroSAT loading, stratified splitting and transforms.

EuroSAT (Helber et al., 2019) has 27,000 Sentinel-2 satellite tiles, 64x64 RGB,
in 10 land-use classes. It has no official train/test split, so we build a
stratified one with a fixed seed. Every notebook uses the same seed, which means
the Week 3 and Week 4 test sets are identical and their numbers are comparable.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Coarse land-use groups, used for "scene-level" evaluation.
GROUPS = {
    "urban": ["Highway", "Industrial", "Residential"],
    "agriculture": ["AnnualCrop", "PermanentCrop", "Pasture"],
    "natural": ["Forest", "HerbaceousVegetation"],
    "water": ["River", "SeaLake"],
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_eurosat(root: str = "./data", download: bool = True):
    """Return the raw EuroSAT dataset (PIL images, no transform). ~90 MB download."""
    ds = torchvision.datasets.EuroSAT(root, download=download)
    if list(ds.classes) != CLASSES:
        raise RuntimeError(f"Unexpected class list: {ds.classes}")
    return ds


class TransformSubset(Dataset):
    """A subset of a base dataset with its own transform.

    The raw dataset is loaded once; train/val/test views each get their own
    transform instead of loading the whole dataset three times.
    """

    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def targets(self):
        return np.asarray(self.base.targets)[self.indices]


def stratified_split(targets, fractions=(0.7, 0.15, 0.15), seed: int = 42):
    """Split indices per class by the given fractions. Returns one list per fraction."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    splits = [[] for _ in fractions]
    for c in np.unique(targets):
        idx = rng.permutation(np.flatnonzero(targets == c))
        cuts = np.cumsum([int(round(f * len(idx))) for f in fractions[:-1]])
        for split, part in zip(splits, np.split(idx, cuts)):
            split.extend(part.tolist())
    for a in range(len(splits)):
        for b in range(a + 1, len(splits)):
            assert not set(splits[a]) & set(splits[b]), "LEAKAGE: splits overlap"
    return [sorted(s) for s in splits]


def per_class_subset(indices, targets, n_per_class: int, seed: int = 42):
    """Take at most n_per_class random indices of every class from `indices`."""
    targets = np.asarray(targets)
    idx = np.asarray(indices)
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(targets[idx]):
        pool = idx[targets[idx] == c]
        out.extend(rng.permutation(pool)[:n_per_class].tolist())
    return sorted(out)


def class_counts(indices, targets, n_classes: int = len(CLASSES)):
    return np.bincount(np.asarray(targets)[list(indices)], minlength=n_classes)


def compute_mean_std(base, indices, max_images: int = 3000, seed: int = 42):
    """Per-channel mean/std over (a sample of) the training images only."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(indices, size=min(max_images, len(indices)), replace=False)
    to_tensor = T.ToTensor()
    s, sq, n = torch.zeros(3), torch.zeros(3), 0
    for i in pick:
        x = to_tensor(base[int(i)][0])
        s += x.sum(dim=(1, 2))
        sq += (x ** 2).sum(dim=(1, 2))
        n += x[0].numel()
    mean = s / n
    std = (sq / n - mean ** 2).sqrt()
    return tuple(round(v, 4) for v in mean.tolist()), tuple(round(v, 4) for v in std.tolist())


def make_transforms(mean, std, size: int | None = None, augment: bool = False):
    """Eval transform, or a training transform with satellite-friendly augmentation.

    A satellite tile has no 'up': vertical flips and 90-degree rotations give
    valid new images, which is not true for ordinary photos.
    """
    ops = []
    if augment:
        ops += [
            T.RandomResizedCrop(size or 64, scale=(0.75, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    elif size is not None:
        ops.append(T.Resize(size, antialias=True))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def make_loader(base, indices, transform, batch_size=128, shuffle=False,
                num_workers=2, sampler=None):
    return DataLoader(
        TransformSubset(base, indices, transform),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def denormalize(x, mean, std):
    """Undo Normalize for display: (C,H,W) tensor -> (H,W,C) in [0, 1]."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0)
