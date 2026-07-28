"""Loads the custom SimpleSAM2Seg checkpoint."""

import os

import torch

from sam_vla.perception.sam2_custom_head import NUM_CLASSES, SimpleSAM2Seg, build_sam2_backbone

SAM_CHECKPOINT_PATH = "sam/sam/sam_v2_dataset/training_output/checkpoints/best_model.pth"


def load_sam_model(checkpoint_path: str = SAM_CHECKPOINT_PATH, device: str = "cuda") -> object:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Custom SimpleSAM2Seg weights required, none found at '{checkpoint_path}'. "
        )

    sam2_backbone = build_sam2_backbone(device)
    model = SimpleSAM2Seg(sam2_backbone, NUM_CLASSES).to(device)

    # best_model.pth is a training checkpoint dict, not a bare state dict.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model
