# Foundation vs. Transfer Learning: 3D Pancreas CT

A reproducible, small-GPU experiment comparing **Swin UNETR trained from scratch** with **the same architecture initialized from self-supervised pretrained weights** on Medical Segmentation Decathlon Task07 Pancreas.

## Scope

- 3D CT segmentation: background (0), pancreas (1), tumor (2).
- Hardware target: NVIDIA RTX 4080 (16 GB) under Windows/WSL2.
- First milestone: inspect data, instantiate the network, and complete a smoke test.
- Next milestones: training, fixed-split evaluation, Dice scores for pancreas and tumor, and a label-efficiency comparison.

**Important:** Swin UNETR is a segmentation architecture, not itself a foundation model. The comparison tests transfer of self-supervised encoder representations. A pretrained experiment requires a *compatible* Swin UNETR checkpoint; without one, do not call a run pretrained.

## Setup (inside WSL2)

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build compatible with your driver:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -c "import torch,monai; print(torch.__version__,monai.__version__,torch.cuda.is_available())"
```

## Data

Download MSD Task07_Pancreas separately (dataset not included in Git). Point to the folder containing `imagesTr/` and `labelsTr/`.

```bash
python -m src.inspect_data --data-dir /path/to/Task07_Pancreas
python -m src.smoke_test --device cuda --feature-size 24 --roi-size 64
```

The smoke test uses a small **randomly initialized** model by default. It does not report segmentation accuracy. A `feature_size=48` architecture may be needed for compatibility with specific released pretrained checkpoints; verify the original checkpoint architecture and key mapping before loading weights.

## Experimental safeguards

Keep patients/volumes disjoint between train and validation; apply CT intensity normalization consistently; use identical splits, optimizer budgets, and evaluation for both initializations. Report tumor Dice separately from pancreas Dice. Record checkpoint provenance, package versions, random seeds, and any missing/unexpected keys when loading pretrained weights.

## Status

Initial scaffolding only. **No training or validation results are claimed yet.**
