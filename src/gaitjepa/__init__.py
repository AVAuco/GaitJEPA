# (c) MJMJ/2026
"""
GaitJEPA: How far can we go with JEPA on binary silhouettes for gait recognition? (IJCB 2026).
"""

__version__ = "0.1.0"

from .checkpoint import compute_sha256, load_pretrained_encoder
from .encoder import DeepGaitV2JEPAEncoder
from .preprocessing import preprocess_silhouettes, sample_sequence_indices

__all__ = [
    "DeepGaitV2JEPAEncoder",
    "load_pretrained_encoder",
    "preprocess_silhouettes",
    "sample_sequence_indices",
    "compute_sha256",
    "__version__",
]
