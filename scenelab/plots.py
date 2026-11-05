"""Plotting helpers used by the notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show_tiles(base, indices, classes, cols=8, title=None, size=1.8):
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        img, label = base[i]
        ax.imshow(img)
        ax.set_title(classes[label], fontsize=7)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_class_counts(counts, classes, title="Images per class"):
    plt.figure(figsize=(9, 3))
    plt.bar(range(len(classes)), counts, color="#3b7ea1")
    plt.xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_histories(hists: dict):
    """One column per run: accuracy on top, loss below, train vs validation."""
    n = len(hists)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.5), squeeze=False)
    for j, (name, h) in enumerate(hists.items()):
        e = range(1, len(h["val_acc"]) + 1)
        gap = h["train_acc"][-1] - h["val_acc"][-1]
        axes[0, j].plot(e, h["train_acc"], "o-", label="train")
        axes[0, j].plot(e, h["val_acc"], "s-", label="validation")
        axes[0, j].set_ylim(0, 1)
        axes[0, j].set_title(f"{name}\naccuracy (final gap {gap:+.1%})", fontsize=10)
        axes[1, j].plot(e, h["train_loss"], "o-", label="train")
        axes[1, j].plot(e, h["val_loss"], "s-", label="validation")
        axes[1, j].set_title("loss", fontsize=10)
        axes[1, j].set_xlabel("epoch")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_confusion(cm, classes, normalize=True, title="Confusion matrix"):
    data = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.7 * n + 2.5, 0.6 * n + 2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else int(v), ha="center", va="center",
                    fontsize=6, color="white" if v > data.max() / 2 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.show()


def show_confident_mistakes(base, indices, probs, targets, classes, k=8):
    """The k wrong predictions the model was most sure about."""
    conf, preds = probs.max(1)
    wrong = (preds != targets).nonzero().flatten()
    if len(wrong) == 0:
        print("No mistakes to show.")
        return
    worst = wrong[conf[wrong].argsort(descending=True)[:k]].tolist()
    fig, axes = plt.subplots(1, len(worst), figsize=(2.1 * len(worst), 2.8), squeeze=False)
    for ax, pos in zip(axes[0], worst):
        img, _ = base[indices[pos]]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"true: {classes[targets[pos]]}\npred: {classes[preds[pos]]}\n"
                     f"({conf[pos]:.0%} sure)", fontsize=7)
    plt.tight_layout()
    plt.show()
