"""GPU/CPU forward-pass smoke test. No dataset or pretrained weights required."""
import argparse

import torch

from .model import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside this Python environment.")
    if args.roi_size % 32:
        raise SystemExit("--roi-size must be divisible by 32 for this configuration.")

    model = make_model(args.feature_size, args.checkpoint).to(args.device).eval()
    x = torch.randn(1, 1, args.roi_size, args.roi_size, args.roi_size, device=args.device)
    with torch.inference_mode():
        y = model(x)
    print(f"Input: {tuple(x.shape)} | Output: {tuple(y.shape)}")
    assert y.shape == (1, 3, args.roi_size, args.roi_size, args.roi_size)
    if args.device == "cuda":
        print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("Forward pass OK; no training has been performed.")


if __name__ == "__main__":
    main()
