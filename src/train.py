"""Small-budget Swin UNETR training with volume-level validation.

Example:
  python -m src.train --data-dir msd_pancreas --steps 100 --train-count 20 --val-count 3
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from .metrics import CLASS_NAMES, case_metrics
from .visualize import save_overlay

from .dataset import make_datasets
from .model import make_model


def evaluate(model, loader, roi_size, device, overlay_path=None):
    """Mean per-case Dice, excluding cases with empty ground truth."""
    model.eval()
    per_case = []
    first_overlay = overlay_path
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                output = sliding_window_inference(
                    image, (roi_size,) * 3, 1, model, overlap=0.25
                )
            pred = output.argmax(dim=1)
            for i in range(pred.shape[0]):
                scores = case_metrics(pred[i], label[i, 0])
                per_case.append(scores)
                for name in CLASS_NAMES:
                    print(f"  {name}: predicted={scores[name + '_pred_voxels']} "
                          f"true={scores[name + '_true_voxels']}", flush=True)
                if first_overlay is not None:
                    save_overlay(image[i, 0].cpu().numpy(),
                                 label[i, 0].cpu().numpy(),
                                 pred[i].cpu().numpy(), first_overlay)
                    first_overlay = None
    result = {}
    for name in CLASS_NAMES:
        valid = [row[name + "_dice"] for row in per_case
                 if not np.isnan(row[name + "_dice"])]
        result[name + "_dice"] = float(np.mean(valid)) if valid else float("nan")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/smoke_training")
    parser.add_argument("--checkpoint", default=None,
                        help="Only supply a compatible full-model state_dict.")
    args = parser.parse_args()
    if args.steps < 1 or args.val_every < 1 or args.roi_size % 32:
        parser.error("steps and val-every must be positive; roi-size divisible by 32")
    if not torch.cuda.is_available():
        parser.error("CUDA not available; this experiment is intended for the GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    train_ds, val_ds, train_cases, val_cases = make_datasets(
        args.data_dir, args.train_count, args.val_count, args.roi_size, args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=0, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = make_model(args.feature_size, args.checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler("cuda")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.json").open("w") as f:
        json.dump({"arguments": vars(args),
                   "train_files": [Path(c["image"]).name for c in train_cases],
                   "val_files": [Path(c["image"]).name for c in val_cases],
                   "torch": torch.__version__, "cuda": torch.version.cuda}, f, indent=2)
    metrics_file = output / "metrics.csv"
    iterator = iter(train_loader)
    with metrics_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "train_loss", "pancreas_dice", "tumor_dice", "whole_pancreas_dice"
        ])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            model.train()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            image = batch["image"].to(device)
            label = batch["label"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(image)
                loss = loss_fn(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "pancreas_dice": "", "tumor_dice": "", "whole_pancreas_dice": ""}
            if step % args.val_every == 0 or step == args.steps:
                scores = evaluate(model, val_loader, args.roi_size, device,
                                  output / f"overlay_step_{step}.png")
                row.update(scores)
                print(f"Step {step}: loss={row['train_loss']:.4f} "
                      f"pancreas Dice={scores['pancreas_dice']:.3f} "
                      f"tumor Dice={scores['tumor_dice']:.3f} "
                      f"whole Dice={scores['whole_pancreas_dice']:.3f}", flush=True)
                torch.save(model.state_dict(), output / "last_model.pt")
            elif step == 1 or step % 10 == 0:
                print(f"Step {step}: loss={row['train_loss']:.4f}", flush=True)
            writer.writerow(row)
            f.flush()
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
