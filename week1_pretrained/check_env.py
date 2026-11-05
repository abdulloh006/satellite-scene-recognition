"""Week 1: check that every library the project needs is installed."""
import importlib
import platform

PACKAGES = ["torch", "torchvision", "cv2", "skimage", "sklearn", "matplotlib", "numpy", "PIL"]

print(f"Python {platform.python_version()} on {platform.system()}")
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<12} {getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"  {name:<12} MISSING  ->  pip install -r requirements.txt")

try:
    import torch
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("GPU: Apple MPS")
    else:
        print("GPU: none (CPU only; use Colab with a T4 GPU for Weeks 3-4)")
except ImportError:
    pass
