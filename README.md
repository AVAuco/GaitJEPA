# GaitJEPA: Self-Supervised Gait Representation Learning (IJCB 2026)

Official minimal inference release of the pretrained **GaitJEPA** encoder for silhouette-based gait representation extraction.

**Repository**: [https://github.com/AVAuco/gaitjepa](https://github.com/AVAuco/gaitjepa)

## Overview and Scope

This package provides the pretrained gait encoder described in the IJCB 2026 paper. It is designed to:
1. Load the pretrained encoder weights with minimal dependencies.
2. Extract spatial, part-based, and spatiotemporal representations from silhouette sequences.
3. Serve as a standardized feature extractor or backbone initialization for downstream tasks (e.g., identity recognition, sex estimation, clinical gait analysis).

> **Important Scope Clarification**:
> The published checkpoint is a **self-supervised pretrained initialization**, not a ready-to-use identity or sex classifier. Feature extraction alone does not reproduce published benchmark tables; downstream performance results also depend on the respective downstream adaptation protocol (linear probing, fine-tuning, metric learning head, etc.).
> Training engines, dataset-specific downloaders, and full benchmark suites are deliberately excluded from this minimal inference distribution to keep dependencies minimal and portable.

---

## Installation

The core inference engine requires **only PyTorch** (`torch >= 2.0.0`):

```bash
# Minimal installation (inference only)
pip install .

# Optional dependencies for reading images (Pillow) or numpy arrays
pip install ".[data]"

# For running test suites
pip install ".[test]"
```

---

## Quickstart

```python
import torch
from gaitjepa import load_pretrained_encoder, preprocess_silhouettes

# 1. Load pretrained encoder (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = load_pretrained_encoder("weights/gaitjepa_ijcb2026_encoder.pth", map_location=device)

# 2. Prepare silhouettes [B, T, 1, 64, 44] float32 in range [0.0, 1.0]
# Example with synthetic silhouettes:
dummy_sils = torch.randint(0, 256, (1, 30, 64, 44), dtype=torch.uint8)
x = preprocess_silhouettes(dummy_sils).to(device)

# 3. Extract representations
with torch.no_grad():
    # Global normalized embedding vector [B, 256]
    global_emb = encoder.forward_features(x, pool=True)

    # Part embeddings [B, 256, 16] (16 horizontal parts)
    part_emb = encoder.forward_features(x, pool=False)

    # Spatiotemporal part tokens [B, T, 16, 256]
    tokens = encoder.forward_features(x, pool="unflattened_tokens")

print("Global embedding shape:", global_emb.shape)  # [1, 256]
print("Part embeddings shape: ", part_emb.shape)    # [1, 256, 16]
print("Token representations:  ", tokens.shape)      # [1, 30, 16, 256]
```

---

## Input Contract

To ensure representations match the pretraining distribution:
- **Tensor Shape**: `[B, T, 1, 64, 44]` (`Batch, Frames, Channels, Height, Width`).
- **Value Range**: Float32 values normalized to `[0.0, 1.0]` (binary or grayscale foreground silhouettes). If passing `uint8` arrays `[0, 255]`, `preprocess_silhouettes` will divide by 255.0 automatically.
- **Cropping & Centering**: Standard gait preprocessing (centered bounding box, aspect ratio preserved with horizontal padding to 64x44) must be applied prior to inference.
- **Temporal Length**: Pretraining utilized clips of $T=30$ frames. The convolutional P3D architecture natively supports variable sequence lengths ($T \ge 8$), but downstream comparability is highest around $T=30$ frames.
- **Interpolation**: When resizing frames to $(64, 44)$, nearest-neighbor interpolation is recommended for binary silhouettes to avoid boundary blurring.

---

## Output Representations

| Representation | Method Call / Key | Shape | Description |
|---|---|---|---|
| **Global Embedding** | `forward_features(x, pool=True)` or `forward(x)['global_embedding']` | `[B, 256]` | L2-normalized representation pooled across parts and time. |
| **Part Embeddings** | `forward_features(x, pool=False)` or `forward(x)['pooled_part_embedding']` | `[B, 256, 16]` | Temporal max-pooled representation per horizontal part bin. |
| **Spatiotemporal Part Tokens** | `forward_features(x, pool="unflattened_tokens")` or `forward(x)['sequence_part_tokens']` | `[B, T, 16, 256]` | Part tokens across all individual time steps. |
| **Flattened Part Tokens** | `forward_tokens(x)` or `forward_features(x, pool="tokens")` | `[B, T * 16, 256]` | Temporal tokens flattened along the sequence dimension. |

---

## Provenance & Checkpoint Metadata

Full audit records and provenance are stored in `provenance/checkpoint.json`:
- **Source checkpoint**: Pretrained on GaitLU-1M (`epoch_0050.pth`).
- **Original Checkpoint SHA-256**: `76824736c5fddb41cd47dec30bc2dad69e6c5cc5b5dd272f9640699b40d62f49` (146.1 MB).
- **Derived Inference Checkpoint SHA-256**: `2b90c9adc7167bd7f9f3d238b7df6b6f52182f0fac6d2cd91ec484874d13d665` (31.2 MB).
- **Original Config Files**: Archived unchanged under `provenance/pretraining/`.

> **Note on Training Schedule**:
> The configuration saved in `epoch_0050.pth` reflects a run configured for 500 epochs with `frames_skip_num: 4` (1 frame sampled every 5 raw frames). The paper narrative outlines 50 epochs on consecutive frames. The historical YAML files in `provenance/pretraining/` reflect the exact files as preserved in the research repository.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@inproceedings{gaitjepa2026ijcb,
  title={GaitJEPA: How far can we go with JEPA on binary silhouettes for gait recognition?},
  author={Marin-Jimenez, Manuel J. and Jimenez-Velasco, Isabel and Munoz-Salinas, Rafael},
  booktitle={IEEE International Joint Conference on Biometrics (IJCB)},
  year={2026}
}
```
