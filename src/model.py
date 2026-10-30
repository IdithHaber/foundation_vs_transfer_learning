"""Swin UNETR factory; do not silently substitute random weights for pretraining."""
import torch
from monai.networks.nets import SwinUNETR


def make_model(feature_size: int = 24, checkpoint: str | None = None):
    model = SwinUNETR(
        in_channels=1,
        out_channels=3,
        feature_size=feature_size,
        use_checkpoint=True,
    )
    if checkpoint is not None:
        # A checkpoint must match the architecture and naming conventions.
        # This strict load intentionally rejects incompatible weights.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    return model
