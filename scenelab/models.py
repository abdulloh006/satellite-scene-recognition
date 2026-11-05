"""Models: a small CNN trained from scratch, and pretrained backbones for transfer learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SatNet(nn.Module):
    """Small CNN for 64x64 satellite tiles.

    Four stages of Conv3x3 -> BatchNorm -> ReLU -> MaxPool, then global average
    pooling instead of a big flatten + dense layer. GAP keeps the classifier tiny
    (256*10 weights) and makes the network work for any input size.
    """

    def __init__(self, num_classes=10, widths=(32, 64, 128, 256), dropout=0.3, batchnorm=True):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers.append(nn.Conv2d(c_in, w, kernel_size=3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(w))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, num_classes),  # logits; CrossEntropyLoss applies softmax
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def receptive_fields(sequential: nn.Sequential):
    """Receptive field after every Conv/Pool layer, read from the layers themselves.

    r_out = r_in + (k - 1) * jump,  jump_out = jump_in * stride
    """
    r, jump, rows = 1, 1, []
    for name, layer in sequential.named_children():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            r += (k - 1) * jump
            jump *= s
            rows.append((f"{name}:{layer.__class__.__name__}", r, jump))
    return rows


def conv_out_size(h, k, p=0, s=1, d=1):
    """Output size of a conv/pool layer: floor((h + 2p - d(k-1) - 1) / s) + 1."""
    return (h + 2 * p - d * (k - 1) - 1) // s + 1


def build_resnet18(num_classes=10, pretrained=True, dropout=0.0):
    """ImageNet ResNet-18 with every layer frozen and a fresh trainable head."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)  # ~45 MB download the first time
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True
    return module


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017): which pixels pushed the model to its answer?

    Takes the activations of `layer`, weights each channel by the mean gradient of
    the class score, sums the channels and keeps the positive part.
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._hook = layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)  # makes gradients flow even through frozen layers
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        self.model.zero_grad()
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        flat = cam.flatten(1)
        lo, hi = flat.min(1)[0].view(-1, 1, 1, 1), flat.max(1)[0].view(-1, 1, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam[:, 0].detach().cpu(), class_idx.detach().cpu()

    def remove(self):
        self._hook.remove()
