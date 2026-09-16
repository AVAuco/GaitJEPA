# (c) MJMJ/2026
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def sample_sequence_indices(
    total_frames: int,
    target_length: int = 30,
    stride: int = 1,
    start_idx: int = 0,
    mode: str = "even",
) -> List[int]:
    """
    Sample frame indices from a sequence of silhouettes.

    Args:
        total_frames: Total number of frames available in the video/sequence.
        target_length: Desired number of sampled frames (e.g., 30 for pretraining clip length).
        stride: Stride between frames when using consecutive sampling.
        start_idx: Starting frame index for consecutive sampling.
        mode: Sampling strategy:
            - 'even': Evenly spaced positions across the sequence. If sequence is shorter than target_length,
                      indices wrap cyclically.
            - 'consecutive': Select consecutive frames with given stride.

    Returns:
        List of integer frame indices.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    if total_frames == target_length:
        return list(range(total_frames))

    if mode == "even":
        if total_frames < target_length:
            indices = list(range(total_frames))
            pad_count = target_length - total_frames
            indices.extend([i % total_frames for i in range(pad_count)])
            return indices

        # Evenly spaced sampling across [0, total_frames - 1]
        step = (total_frames - 1) / float(target_length - 1) if target_length > 1 else 0.0
        return [int(round(i * step)) for i in range(target_length)]

    elif mode == "consecutive":
        indices = []
        for i in range(target_length):
            idx = start_idx + i * stride
            indices.append(idx % total_frames)
        return indices

    else:
        raise ValueError(f"Unsupported sampling mode '{mode}'. Choose 'even' or 'consecutive'.")


def preprocess_silhouettes(
    silhouettes: Union[torch.Tensor, Any],
    target_resolution: Tuple[int, int] = (64, 44),
    interpolation_mode: str = "nearest",
) -> torch.Tensor:
    """
    Preprocess raw silhouette arrays or tensors to canonical encoder input: [B, T, 1, 64, 44] float32 in [0.0, 1.0].

    Supports:
        - NumPy arrays (uint8 [0, 255] or float [0.0, 1.0]): (T, H, W) or (B, T, H, W).
        - PyTorch tensors: (T, H, W), (B, T, H, W), (B, T, 1, H, W), or (B, 1, T, H, W).

    Args:
        silhouettes: Input silhouette data.
        target_resolution: Desired spatial resolution (Height, Width). Default: (64, 44).
        interpolation_mode: Interpolation mode for F.interpolate (default: 'nearest').

    Returns:
        torch.FloatTensor of shape [B, T, 1, 64, 44] with values in range [0.0, 1.0].
    """
    # Convert from numpy if needed
    if not isinstance(silhouettes, torch.Tensor):
        if hasattr(silhouettes, "__array__"):
            import numpy as np
            silhouettes = torch.from_numpy(np.asarray(silhouettes))
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(silhouettes)}")

    # Ensure float32 and scale [0, 1] if values appear to be in uint8 range [0, 255]
    if silhouettes.dtype == torch.uint8 or silhouettes.max() > 1.0:
        tensor = silhouettes.float() / 255.0
    else:
        tensor = silhouettes.float()

    # Normalize shapes to [B, T, 1, H, W]
    if tensor.dim() == 2:
        # Single frame [H, W]
        tensor = tensor.view(1, 1, 1, tensor.shape[0], tensor.shape[1])
    elif tensor.dim() == 3:
        # [T, H, W]
        tensor = tensor.unsqueeze(0).unsqueeze(2)
    elif tensor.dim() == 4:
        if tensor.shape[1] == 1:
            # [T, 1, H, W] -> unbatched
            tensor = tensor.unsqueeze(0)
        else:
            # [B, T, H, W]
            tensor = tensor.unsqueeze(2)
    elif tensor.dim() == 5:
        if tensor.shape[1] == 1 and tensor.shape[2] != 1:
            # [B, 1, T, H, W] -> transpose to [B, T, 1, H, W]
            tensor = tensor.transpose(1, 2)
    else:
        raise ValueError(f"Unsupported silhouette tensor dimensionality: {tensor.dim()} (shape: {tensor.shape})")

    b, t, c, h, w = tensor.shape
    target_h, target_w = target_resolution

    if (h, w) != (target_h, target_w):
        # Flatten B and T to resize frames via 4D F.interpolate
        reshaped = tensor.view(b * t, c, h, w)
        resized = F.interpolate(reshaped, size=(target_h, target_w), mode=interpolation_mode)
        tensor = resized.view(b, t, c, target_h, target_w)

    return tensor.contiguous()
