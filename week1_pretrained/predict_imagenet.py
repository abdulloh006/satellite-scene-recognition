"""Week 1: classify an image with ImageNet-pretrained models and compare them.

Usage (from the repo root):
    python week1_pretrained/predict_imagenet.py --image samples/eurosat_river.png
    python week1_pretrained/predict_imagenet.py --image samples/eurosat_highway.png --models resnet18 efficientnet_b0 --topk 5

Try both an ordinary photo and a satellite tile. ImageNet has no "farmland seen
from space" class, so on tiles the predictions are guesses at the nearest
ImageNet object. That gap is the reason Weeks 3-4 train on EuroSAT.
"""
import argparse
import time

import torch
import torchvision.models as tvm
from PIL import Image

# name -> (constructor, weights enum). Labels come from the weights' metadata,
# so no label file has to be downloaded.
MODELS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def classify(image: Image.Image, name: str, topk: int):
    build, weights = MODELS[name]
    model = build(weights=weights).eval()
    preprocess = weights.transforms()  # the exact resize/crop/normalise the model was trained with
    labels = weights.meta["categories"]
    x = preprocess(image).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000
    top = probs.topk(topk)
    n_params = sum(p.numel() for p in model.parameters())
    return [(labels[i], s.item()) for s, i in zip(top.values, top.indices)], ms, n_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image}  ({image.width}x{image.height})\n")
    for name in args.models:
        preds, ms, n_params = classify(image, name, args.topk)
        print(f"{name}  ({n_params / 1e6:.1f}M params, {ms:.0f} ms on CPU)")
        for label, p in preds:
            print(f"   {label:<28}{p:7.2%}")
        print()


if __name__ == "__main__":
    main()
