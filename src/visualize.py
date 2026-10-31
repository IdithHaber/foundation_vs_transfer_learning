"""Save a representative axial CT slice with ground-truth/predicted overlays."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_overlay(image, truth, prediction, path):
    """Inputs are 3D arrays in matching spatial orientation."""
    image = np.asarray(image)
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if image.shape != truth.shape or image.shape != prediction.shape:
        raise ValueError("Image, truth, and prediction must have same shape")
    # Prefer a slice containing the most tumor voxels, else pancreas.
    score = (truth == 2).sum(axis=(0, 1))
    if score.max() == 0:
        score = (truth > 0).sum(axis=(0, 1))
    z = int(score.argmax()) if score.max() else image.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, label in zip(
        axes, ("CT", "Ground truth", "Prediction"),
        (None, truth, prediction)
    ):
        ax.imshow(image[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        if label is not None:
            overlay = np.ma.masked_where(label[:, :, z].T == 0, label[:, :, z].T)
            ax.imshow(overlay, cmap="autumn", vmin=1, vmax=2,
                      origin="lower", alpha=0.6, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Axial slice {z} | pancreas=1, tumor=2")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
