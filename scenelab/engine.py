"""Training and evaluation loops shared by every experiment."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_train_mode(model: nn.Module):
    """model.train(), but frozen BatchNorm layers stay in eval mode.

    Otherwise a 'frozen' backbone still changes: BN running statistics update
    in train mode even when its weights get no gradient.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in m.parameters()
        ):
            m.eval()


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    """One pass over `loader`. Trains if an optimizer is given. Returns (loss, accuracy)."""
    training = optimizer is not None
    set_train_mode(model) if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:  # per-batch schedulers such as OneCycleLR
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, device, epochs, optimizer,
        criterion=None, scheduler=None, tag="model", restore_best=True):
    """Train for `epochs`, track history, and (by default) restore the best-validation weights."""
    criterion = criterion or nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        vl, va = run_epoch(model, val_loader, device, nn.CrossEntropyLoss())
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)):
            hist[k].append(v)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        if va > best_acc:
            best_acc, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{tag}] {ep:>2}/{epochs} | train {ta:6.1%} loss {tl:.3f} | "
              f"val {va:6.1%} loss {vl:.3f} | {time.time() - t0:4.0f}s")
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{tag}] restored best weights from epoch {best_epoch} (val {best_acc:.1%})")
    return hist


@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities and true labels for a whole loader."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(probs), torch.cat(ys)


@torch.no_grad()
def extract_features(backbone, loader, device):
    """Run a headless backbone over a loader: returns (N, D) numpy features and labels."""
    backbone.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).flatten(1).cpu())
        ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()
