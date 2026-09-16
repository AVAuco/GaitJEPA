# (c) MJMJ/2026
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .encoder import DeepGaitV2JEPAEncoder

ENCODER_PREFIX = "online_encoder."


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_encoder_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract encoder parameters and buffers from a checkpoint dictionary.

    Supports:
        1. Full training checkpoint containing 'online_encoder.<key>' keys.
        2. Filtered inference state dict where keys are already direct encoder parameter names.
    """
    if "state_dict" in state_dict:
        raw_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        raw_dict = state_dict["model"]
    else:
        raw_dict = state_dict

    has_prefixed_keys = any(k.startswith(ENCODER_PREFIX) for k in raw_dict.keys())

    if has_prefixed_keys:
        extracted = {
            k[len(ENCODER_PREFIX):]: v
            for k, v in raw_dict.items()
            if k.startswith(ENCODER_PREFIX)
        }
    else:
        # Check if keys match expected encoder keys
        valid_prefixes = ("layer0.", "layer1.", "layer2.", "layer3.", "layer4.", "part_proj.", "fcs.")
        extracted = {
            k: v for k, v in raw_dict.items()
            if any(k.startswith(p) for p in valid_prefixes)
        }

    return extracted


def load_pretrained_encoder(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    **encoder_kwargs,
) -> DeepGaitV2JEPAEncoder:
    """
    Load a pretrained DeepGaitV2JEPAEncoder from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint (either full pretrain or exported inference weights).
        map_location: Device to load the tensors onto (default: "cpu").
        strict: Whether to strictly enforce that the keys in state_dict match model keys.
        **encoder_kwargs: Optional overrides for DeepGaitV2JEPAEncoder initialization.

    Returns:
        DeepGaitV2JEPAEncoder instance with loaded weights, in eval mode.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # Attempt safe weights_only load first; fallback to full load if structure requires it
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        loaded = torch.load(path, map_location=map_location, weights_only=False)

    encoder_sd = extract_encoder_state_dict(loaded)
    if not encoder_sd:
        raise ValueError(
            f"No encoder weights found in checkpoint {path}. Ensure the checkpoint contains "
            f"'{ENCODER_PREFIX}*' or direct encoder layers ('layer0.', 'fcs.', etc.)."
        )

    model = DeepGaitV2JEPAEncoder(**encoder_kwargs)
    load_result = model.load_state_dict(encoder_sd, strict=strict)

    if not strict and (load_result.missing_keys or load_result.unexpected_keys):
        import warnings
        warnings.warn(
            f"Loaded with discrepancies: missing {len(load_result.missing_keys)} keys, "
            f"unexpected {len(load_result.unexpected_keys)} keys."
        )

    model.to(map_location)
    model.eval()
    return model
