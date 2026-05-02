from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed yet. PyTorch cu128 will be installed later.")
        return

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
