"""Validate MSD dataset.json training pairs; report unlisted files separately."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir
    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    pairs = manifest.get("training", [])
    if not pairs:
        raise SystemExit("No training pairs in dataset.json")
    print(f"Manifest numTraining: {manifest.get('numTraining')}")
    print(f"Manifest training entries: {len(pairs)}")
    print(f"Manifest numTest: {manifest.get('numTest')}")

    images = []
    labels = []
    missing = []
    for entry in pairs:
        image = root / entry["image"]
        label = root / entry["label"]
        images.append(image)
        labels.append(label)
        for p in (image, label):
            if not p.is_file():
                missing.append(str(p))
    if len(set(images)) != len(images) or len(set(labels)) != len(labels):
        raise SystemExit("Duplicate training paths in dataset.json")

    actual_images = set((root / "imagesTr").glob("*.nii.gz"))
    actual_labels = set((root / "labelsTr").glob("*.nii.gz"))
    extra_images = sorted(actual_images - set(images))
    extra_labels = sorted(actual_labels - set(labels))
    print(f"Files in imagesTr: {len(actual_images)}")
    print(f"Files in labelsTr: {len(actual_labels)}")
    print(f"Unlisted training images: {len(extra_images)}")
    print(f"Unlisted training labels: {len(extra_labels)}")
    for p in extra_images:
        print(f"  extra image: {p.name}")
    for p in extra_labels:
        print(f"  extra label: {p.name}")
    if missing:
        for p in missing:
            print(f"  missing: {p}")
        raise SystemExit(f"{len(missing)} referenced files are missing")
    if manifest.get("numTraining") != len(pairs):
        raise SystemExit("numTraining does not match manifest training list")
    print("All manifest training pairs exist. Extra files are ignored.")


if __name__ == "__main__":
    main()
