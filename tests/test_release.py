# (c) MJMJ/2026
import math
from pathlib import Path

import pytest
import torch

from gaitjepa import (
    DeepGaitV2JEPAEncoder,
    load_pretrained_encoder,
    preprocess_silhouettes,
    sample_sequence_indices,
)


def test_encoder_default_initialization():
    encoder = DeepGaitV2JEPAEncoder()
    assert encoder.backbone_mode == "p3d"
    assert encoder.embed_dim == 256
    assert encoder.part_dim == 256
    assert encoder.num_parts == 16
    assert encoder.temporal_pool == "max"

    total_params = sum(p.numel() for p in encoder.parameters())
    assert total_params > 8_000_000  # ~8.17M parameters


@pytest.mark.parametrize("batch_size,seq_len", [(1, 16), (2, 30), (3, 45)])
def test_output_shapes_and_norm(batch_size, seq_len):
    encoder = DeepGaitV2JEPAEncoder()
    encoder.eval()

    x = torch.rand(batch_size, seq_len, 1, 64, 44)
    with torch.no_grad():
        out = encoder(x)
        global_emb = out["global_embedding"]
        part_emb = out["pooled_part_embedding"]
        tokens = out["sequence_part_tokens"]
        flat_tokens = encoder.forward_tokens(x)

    # Check shapes
    assert global_emb.shape == (batch_size, 256)
    assert part_emb.shape == (batch_size, 256, 16)
    assert tokens.shape == (batch_size, seq_len, 16, 256)
    assert flat_tokens.shape == (batch_size, seq_len * 16, 256)

    # Check L2 normalization of global embedding
    norms = global_emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_preprocessing():
    # Test uint8 scaling
    uint8_data = torch.randint(0, 256, (2, 20, 64, 44), dtype=torch.uint8)
    preprocessed = preprocess_silhouettes(uint8_data)
    assert preprocessed.shape == (2, 20, 1, 64, 44)
    assert preprocessed.dtype == torch.float32
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0

    # Test spatial resizing
    odd_sized = torch.rand(1, 10, 128, 88)
    resized = preprocess_silhouettes(odd_sized, target_resolution=(64, 44))
    assert resized.shape == (1, 10, 1, 64, 44)


def test_sampling_indices():
    # Test even sampling when sequence is longer
    indices = sample_sequence_indices(total_frames=100, target_length=30, mode="even")
    assert len(indices) == 30
    assert indices[0] == 0
    assert indices[-1] == 99
    assert sorted(indices) == indices

    # Test cyclic wrap when sequence is shorter
    short_indices = sample_sequence_indices(total_frames=10, target_length=30, mode="even")
    assert len(short_indices) == 30
    assert short_indices[:10] == list(range(10))
    assert short_indices[10:20] == list(range(10))


def test_strict_weight_loading():
    weights_path = Path(__file__).resolve().parent.parent / "weights" / "gaitjepa_ijcb2026_encoder.pth"
    if not weights_path.exists():
        pytest.skip(f"Weights not found at {weights_path}")

    encoder = load_pretrained_encoder(weights_path, map_location="cpu", strict=True)
    assert isinstance(encoder, DeepGaitV2JEPAEncoder)
    assert not encoder.training


def test_numerical_parity_with_source_model():
    research_ckpt = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "outputs"
        / "checkpoints"
        / "gaitjepa_pretrain_36_deepgaitv2_hybrid_gaitlu1m_30f"
        / "epoch_0050.pth"
    )
    if not research_ckpt.exists():
        pytest.skip(f"Research checkpoint not found at {research_ckpt}")

    try:
        research_pkg = str(research_ckpt.parents[3] / "src" / "gaitjepa")
        import gaitjepa
        if research_pkg not in gaitjepa.__path__:
            gaitjepa.__path__.append(research_pkg)
        from gaitjepa.models.gait_deepgaitv2_hybrid import DeepGaitV2Hybrid
    except Exception as e:
        pytest.skip(f"Could not import original DeepGaitV2Hybrid from research src: {e}")

    raw_ckpt = torch.load(research_ckpt, map_location="cpu", weights_only=False)
    orig_model = DeepGaitV2Hybrid(raw_ckpt["cfg"])
    orig_model.load_state_dict(raw_ckpt["state_dict"], strict=False)
    orig_model.eval()

    rel_weights = Path(__file__).resolve().parent.parent / "weights" / "gaitjepa_ijcb2026_encoder.pth"
    rel_model = load_pretrained_encoder(rel_weights, map_location="cpu", strict=True)
    rel_model.eval()

    torch.manual_seed(1234)
    x = torch.rand(2, 30, 1, 64, 44)

    with torch.no_grad():
        orig_global = orig_model.forward_features(x, pool=True)
        rel_global = rel_model.forward_features(x, pool=True)

        orig_parts = orig_model.forward_features(x, pool=False)
        rel_parts = rel_model.forward_features(x, pool=False)

        orig_tokens = orig_model.forward_tokens(x)
        rel_tokens = rel_model.forward_tokens(x)

    assert torch.allclose(orig_global, rel_global, atol=1e-6)
    assert torch.allclose(orig_parts, rel_parts, atol=1e-6)
    assert torch.allclose(orig_tokens, rel_tokens, atol=1e-6)
