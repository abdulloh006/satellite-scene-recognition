"""Evaluation helpers: confusion matrix, per-class scores, common mistakes, group accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def to_numpy(a):
    return a.numpy() if hasattr(a, "numpy") else np.asarray(a)


def confusion(y_true, y_pred, n_classes):
    """Rows = true class, columns = predicted class."""
    return confusion_matrix(to_numpy(y_true), to_numpy(y_pred), labels=list(range(n_classes)))


def per_class_table(y_true, y_pred, classes, extra_recall=None, extra_name="other"):
    """Print precision / recall / F1 per class. Returns macro-F1."""
    p, r, f1, support = precision_recall_fscore_support(
        to_numpy(y_true), to_numpy(y_pred), labels=list(range(len(classes))), zero_division=0
    )
    head = f"{'class':<22}{'precision':>10}{'recall':>8}{'F1':>7}{'n':>6}"
    if extra_recall is not None:
        head += f"{'recall (' + extra_name + ')':>22}{'gain':>8}"
    print(head)
    for i, c in enumerate(classes):
        line = f"{c:<22}{p[i]:>10.1%}{r[i]:>8.1%}{f1[i]:>7.2f}{support[i]:>6}"
        if extra_recall is not None:
            line += f"{extra_recall[i]:>22.1%}{r[i] - extra_recall[i]:>+8.1%}"
        print(line)
    print(f"\nmacro-F1: {f1.mean():.3f}")
    return f1.mean()


def recall_per_class(y_true, y_pred, n_classes):
    cm = confusion(y_true, y_pred, n_classes)
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def top_confusions(cm, classes, k=3):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    out = []
    for idx in np.argsort(-off.ravel())[:k]:
        t, p = divmod(int(idx), len(classes))
        if off[t, p] == 0:
            break
        out.append((classes[t], classes[p], int(off[t, p])))
    return out


def group_labels(labels, classes, groups):
    """Map fine class ids to coarse group ids, e.g. River -> water."""
    lookup = {}
    for g_id, members in enumerate(groups.values()):
        for name in members:
            lookup[classes.index(name)] = g_id
    return np.array([lookup[int(l)] for l in to_numpy(labels)])


def group_accuracy(y_true, y_pred, classes, groups):
    """Accuracy at the coarse group level plus the group confusion matrix."""
    gt = group_labels(y_true, classes, groups)
    gp = group_labels(y_pred, classes, groups)
    return (gt == gp).mean(), confusion_matrix(gt, gp, labels=list(range(len(groups))))
