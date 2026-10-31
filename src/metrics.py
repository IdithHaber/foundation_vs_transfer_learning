"""Voxel-level Dice metrics for mutually exclusive MSD pancreas labels.

Undefined Dice (empty ground truth) is NaN and excluded from class means.
"""
import torch

CLASS_NAMES = ("pancreas", "tumor", "whole_pancreas")


def class_masks(labels):
    """Return boolean masks for pancreas tissue, tumor, and their union."""
    labels = torch.as_tensor(labels)
    return {
        "pancreas": labels == 1,
        "tumor": labels == 2,
        "whole_pancreas": (labels == 1) | (labels == 2),
    }


def dice_binary(pred_mask, true_mask):
    """Return Dice; NaN if truth is empty (including both-empty case)."""
    pred = torch.as_tensor(pred_mask, dtype=torch.bool)
    truth = torch.as_tensor(true_mask, dtype=torch.bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and ground truth must have identical shapes")
    true_count = int(truth.sum())
    if true_count == 0:
        return float("nan")
    pred_count = int(pred.sum())
    intersection = int((pred & truth).sum())
    return 2.0 * intersection / (pred_count + true_count)


def case_metrics(pred_labels, true_labels):
    """Compute per-case Dice and voxel counts; accepts one 3D label volume."""
    pred = torch.as_tensor(pred_labels)
    truth = torch.as_tensor(true_labels)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and ground truth must have identical shapes")
    pm, tm = class_masks(pred), class_masks(truth)
    result = {}
    for name in CLASS_NAMES:
        result[f"{name}_dice"] = dice_binary(pm[name], tm[name])
        result[f"{name}_pred_voxels"] = int(pm[name].sum())
        result[f"{name}_true_voxels"] = int(tm[name].sum())
    return result
