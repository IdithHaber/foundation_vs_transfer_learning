"""MSD pancreas training pairs and MONAI preprocessing."""
import json
import random
from pathlib import Path

from monai.data import Dataset
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd,
    RandCropByLabelClassesd, ScaleIntensityRanged, SpatialPadd, Spacingd,
)


def split_cases(data_dir: str, train_count: int = 20, val_count: int = 3, seed: int = 42):
    root = Path(data_dir)
    with (root / "dataset.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    cases = []
    for entry in manifest["training"]:
        image = root / entry["image"]
        label = root / entry["label"]
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image or label: {image}, {label}")
        cases.append({"image": str(image), "label": str(label)})
    if train_count < 1 or val_count < 1 or train_count + val_count > len(cases):
        raise ValueError("Need positive train/val counts within manifest training entries.")
    cases.sort(key=lambda c: c["image"])
    random.Random(seed).shuffle(cases)
    return cases[:train_count], cases[train_count:train_count + val_count]


def transforms(roi_size=64, training=True):
    keys = ["image", "label"]
    pipeline = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=240,
                             b_min=0, b_max=1, clip=True),
        EnsureTyped(keys=keys, dtype=None),
    ]
    if training:
        pipeline.extend([
            SpatialPadd(keys=keys, spatial_size=(roi_size,) * 3),
            RandCropByLabelClassesd(
                keys=keys, label_key="label",
                spatial_size=(roi_size,) * 3, num_classes=3,
                ratios=[1, 2, 4], num_samples=1,
                allow_smaller=False,
            ),
        ])
    pipeline.append(EnsureTyped(keys=keys, track_meta=not training))
    return Compose(pipeline)


def make_datasets(data_dir, train_count=20, val_count=3, roi_size=64, seed=42):
    train_cases, val_cases = split_cases(data_dir, train_count, val_count, seed)
    return (
        Dataset(train_cases, transform=transforms(roi_size, training=True)),
        Dataset(val_cases, transform=transforms(roi_size, training=False)),
        train_cases,
        val_cases,
    )
