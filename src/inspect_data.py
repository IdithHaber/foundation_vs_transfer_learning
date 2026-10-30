"""Check that the local MSD Task07_Pancreas data are present and paired."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    images = args.data_dir / "imagesTr"
    labels = args.data_dir / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"Expected imagesTr/ and labelsTr/ in {args.data_dir}")
    image_files = sorted(images.glob("*.nii.gz"))
    label_files = sorted(labels.glob("*.nii.gz"))
    image_names = {p.name for p in image_files}
    label_names = {p.name for p in label_files}
    print(f"Training CT volumes: {len(image_files)}")
    print(f"Segmentation labels: {len(label_files)}")
    print(f"Missing labels: {len(image_names - label_names)}")
    print(f"Missing images: {len(label_names - image_names)}")
    if not image_files or image_names != label_names:
        raise SystemExit("Dataset is empty or image/label filenames do not match.")
    print("Dataset file pairs verified (voxel contents not yet inspected).")


if __name__ == "__main__":
    main()
