"""Shared fixtures for the test suite.

Tests run without the (private) clinical dataset, so every fixture here builds
synthetic tensors. They cover shape and geometry contracts, config resolution and
coordinate-space conversions - the parts that break silently when the code moves.
"""

from __future__ import annotations

import pytest
import torch

#: Small model dimensions that keep tests fast while remaining architecturally valid.
#: `patch_size` must stay >= 16: the patch CNN downsamples by 2**3, and MONAI's
#: InstanceNorm raises on a 1x1x1 spatial extent even in eval mode.
MODEL_DIMS = {
    "n_landmarks": 4,
    "n_verts": 5,
    "patch_size": 16,
    "poi_feature_l": 16,
    "patch_feature_l": 8,
    "coord_embedding_l": 8,
    "poi_embedding_l": 8,
    "vert_embedding_l": 8,
    "loss_fn": "L1",
    "mlp_dim": 32,
    "num_layers": 1,
    "num_heads": 2,
    "dropout": 0.0,
    "lr": 1e-4,
}


@pytest.fixture
def model_dims() -> dict:
    """Dimensions for constructing a small refinement module."""
    return dict(MODEL_DIMS)


@pytest.fixture
def refinement_batch() -> dict:
    """A synthetic batch shaped like what the coarse stage hands the refiner."""
    generator = torch.Generator().manual_seed(1)
    batch_size, n_landmarks = 2, MODEL_DIMS["n_landmarks"]
    return {
        "coarse_preds": torch.rand(batch_size, n_landmarks, 3, generator=generator) * 8 + 16,
        "coarse_features": torch.randn(batch_size, n_landmarks, MODEL_DIMS["poi_feature_l"], generator=generator),
        "poi_list_idx": torch.randint(0, n_landmarks, (batch_size, n_landmarks), generator=generator),
        "vert_list_idx": torch.randint(0, MODEL_DIMS["n_verts"], (batch_size, n_landmarks), generator=generator),
        "input": torch.randn(batch_size, 1, 48, 48, 48, generator=generator),
        "target": torch.rand(batch_size, n_landmarks, 3, generator=generator) * 8 + 16,
        "loss_mask": torch.ones(batch_size, n_landmarks, dtype=torch.bool),
        "target_indices": torch.arange(n_landmarks).expand(batch_size, n_landmarks),
        "zoom": torch.tensor([[1.0, 1.0, 1.0], [0.8, 0.8, 1.5]]),
    }
