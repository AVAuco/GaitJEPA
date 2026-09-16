#!/usr/bin/env python3
# (c) MJMJ/2026
"""
Example script demonstrating representation extraction with GaitJEPA.

Usage:
    python examples/extract_embeddings.py
    python examples/extract_embeddings.py --checkpoint weights/gaitjepa_ijcb2026_encoder.pth --device cuda
"""

import argparse
from pathlib import Path

import torch

# Allows running directly from package root without installing
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaitjepa import load_pretrained_encoder, preprocess_silhouettes


def create_synthetic_gait_cycle(batch_size: int = 1, seq_len: int = 30) -> torch.Tensor:
    """Generate synthetic binary silhouette ellipses simulating a periodic walking motion."""
    h, w = 64, 44
    t_coords = torch.linspace(0, 2 * 3.14159, seq_len)
    frames = []

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")

    for t in t_coords:
        # Oscillating leg and torso mask
        center_x = 0.1 * torch.sin(t)
        scale_y = 0.7 + 0.05 * torch.cos(2 * t)
        scale_x = 0.35 + 0.05 * torch.sin(t).abs()

        mask = (((xx - center_x) / scale_x) ** 2 + (yy / scale_y) ** 2) <= 1.0
        frames.append(mask.float())

    seq = torch.stack(frames, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, T, H, W]
    return seq


def main():
    parser = argparse.ArgumentParser(description="Extract representations using GaitJEPA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (defaults to release weights or research checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save extracted embeddings (.pt).",
    )
    args = parser.parse_args()

    # Determine default checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        local_weights = Path(__file__).resolve().parent.parent / "weights" / "gaitjepa_ijcb2026_encoder.pth"
        research_weights = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "outputs"
            / "checkpoints"
            / "gaitjepa_pretrain_36_deepgaitv2_hybrid_gaitlu1m_30f"
            / "epoch_0050.pth"
        )
        if local_weights.exists():
            ckpt_path = str(local_weights)
        elif research_weights.exists():
            ckpt_path = str(research_weights)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found. Please provide --checkpoint pointing to the pretrained weights."
            )

    print(f"Loading encoder from: {ckpt_path}")
    print(f"Using device: {args.device}")
    encoder = load_pretrained_encoder(ckpt_path, map_location=args.device)

    # Generate synthetic silhouette batch [B=2, T=30, H=64, W=44]
    print("\nGenerating synthetic silhouette sequence (2 samples, 30 frames, 64x44)...")
    raw_sils = create_synthetic_gait_cycle(batch_size=2, seq_len=30)
    x = preprocess_silhouettes(raw_sils).to(args.device)
    print(f"Preprocessed input tensor: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.2f}, {x.max():.2f}]")

    print("\nExtracting gait representations...")
    with torch.no_grad():
        # 1. Global embedding [B, 256]
        global_emb = encoder.forward_features(x, pool=True)

        # 2. Part embeddings [B, 256, 16]
        part_emb = encoder.forward_features(x, pool=False)

        # 3. Spatiotemporal tokens [B, T, 16, 256]
        spatiotemporal_tokens = encoder.forward_features(x, pool="unflattened_tokens")

        # 4. Flattened tokens [B, T * 16, 256]
        flattened_tokens = encoder.forward_tokens(x)

    print("\n" + "=" * 50)
    print("REPRESENTATION SUMMARY:")
    print(f"  - Global Embedding:           shape={tuple(global_emb.shape)}")
    print(f"                                L2-norm={global_emb.norm(dim=-1).cpu().tolist()}")
    print(f"  - Part Embeddings:            shape={tuple(part_emb.shape)}")
    print(f"  - Spatiotemporal Part Tokens: shape={tuple(spatiotemporal_tokens.shape)}")
    print(f"  - Flattened Part Tokens:      shape={tuple(flattened_tokens.shape)}")
    print("=" * 50)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_embedding": global_emb.cpu(),
                "part_embeddings": part_emb.cpu(),
                "sequence_part_tokens": spatiotemporal_tokens.cpu(),
                "flattened_tokens": flattened_tokens.cpu(),
            },
            out_path,
        )
        print(f"\nEmbeddings saved to {out_path}")


if __name__ == "__main__":
    main()
