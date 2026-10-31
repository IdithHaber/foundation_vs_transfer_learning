import math

import torch

from src.metrics import case_metrics, dice_binary


def test_perfect_masks():
    labels = torch.tensor([[[0, 1, 2]]])
    result = case_metrics(labels, labels)
    assert all(result[f"{name}_dice"] == 1.0
               for name in ("pancreas", "tumor", "whole_pancreas"))


def test_tumor_pancreas_confusion_preserves_whole_organ():
    truth = torch.tensor([[[0, 1, 2]]])
    pred = torch.tensor([[[0, 2, 1]]])
    result = case_metrics(pred, truth)
    assert result["pancreas_dice"] == 0.0
    assert result["tumor_dice"] == 0.0
    assert result["whole_pancreas_dice"] == 1.0


def test_disjoint_masks():
    assert dice_binary(torch.tensor([1, 0]), torch.tensor([0, 1])) == 0.0


def test_empty_ground_truth_is_undefined():
    assert math.isnan(dice_binary(torch.zeros(3), torch.zeros(3)))
    assert math.isnan(dice_binary(torch.ones(3), torch.zeros(3)))


def test_counts():
    truth = torch.tensor([[[0, 1, 2, 2]]])
    pred = torch.tensor([[[0, 0, 1, 2]]])
    result = case_metrics(pred, truth)
    assert result["whole_pancreas_true_voxels"] == 3
    assert result["whole_pancreas_pred_voxels"] == 2


def test_shape_mismatch():
    try:
        case_metrics(torch.zeros((2, 2)), torch.zeros((2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected shape mismatch to raise")
