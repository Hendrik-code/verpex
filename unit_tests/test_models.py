"""Forward-pass shape contracts for the network building blocks."""

from __future__ import annotations

import pytest
import torch

from verpex.geometry.heatmaps import SoftArgmax3D
from verpex.models.patch_extraction import PatchExtractor
from verpex.models.poi_transformer import FlexiblePoiTransformer, PoiTransformer


def test_soft_argmax_recovers_the_peak_of_a_one_hot_heatmap():
    """A one-hot heatmap must decode to that voxel's coordinates.

    SoftArgmax3D takes a coordinate expectation over an already-normalised heatmap -
    it does not apply a softmax itself - so the input must sum to 1.
    """
    heatmap = torch.zeros(1, 1, 8, 8, 8)
    heatmap[0, 0, 2, 3, 4] = 1.0
    coords = SoftArgmax3D()(heatmap)
    assert torch.allclose(coords.squeeze(), torch.tensor([2.0, 3.0, 4.0]), atol=1e-4)


def test_soft_argmax_averages_over_a_spread_distribution():
    """Two equally weighted voxels decode to their midpoint."""
    heatmap = torch.zeros(1, 1, 8, 8, 8)
    heatmap[0, 0, 2, 0, 0] = 0.5
    heatmap[0, 0, 4, 0, 0] = 0.5
    coords = SoftArgmax3D()(heatmap)
    assert coords.squeeze()[0].item() == pytest.approx(3.0, abs=1e-4)


def test_flexible_transformer_matches_the_original_when_all_embeddings_are_on():
    """FlexiblePoiTransformer generalises PoiTransformer; with everything enabled
    the two must be numerically identical. This is what makes the single unified
    PatchTransformer a faithful replacement for the old ablation subclasses.
    """
    kwargs = {
        "poi_feature_l": 24,
        "coord_embedding_l": 8,
        "poi_embedding_l": 8,
        "vert_embedding_l": 8,
        "mlp_dim": 32,
        "num_layers": 1,
        "num_heads": 2,
        "n_landmarks": 4,
        "n_verts": 5,
        "dropout_rate": 0.0,
    }
    torch.manual_seed(0)
    original = PoiTransformer(**kwargs).eval()
    flexible = FlexiblePoiTransformer(**kwargs).eval()
    flexible.load_state_dict(original.state_dict())

    generator = torch.Generator().manual_seed(3)
    coords = torch.rand(2, 4, 3, generator=generator) * 8 + 16
    features = torch.randn(2, 4, 24, generator=generator)
    poi_idx = torch.randint(0, 4, (2, 4), generator=generator)
    vert_idx = torch.randint(0, 5, (2, 4), generator=generator)

    with torch.no_grad():
        expected = original(coords, poi_idx, vert_idx, features)
        actual = flexible(coarse_preds=coords, poi_indices=poi_idx, vertebra=vert_idx, poi_features=features)
    assert torch.equal(expected, actual)


def test_flexible_transformer_requires_at_least_one_input():
    with pytest.raises(ValueError, match="At least one component"):
        FlexiblePoiTransformer(
            poi_feature_l=0,
            coord_embedding_l=None,
            poi_embedding_l=None,
            vert_embedding_l=None,
            mlp_dim=8,
            num_layers=1,
            num_heads=1,
            n_landmarks=2,
        )


def test_patch_extractor_returns_one_feature_vector_per_landmark():
    from monai.networks.nets.regressor import Regressor

    extractor = PatchExtractor(
        patch_size=16,
        feature_extraction_model=Regressor(
            in_shape=(1, 16, 16, 16), out_shape=(8,), channels=(8, 16, 32), strides=(2, 2, 2), kernel_size=3
        ),
    ).eval()
    volume = torch.randn(2, 1, 48, 48, 48)
    coords = torch.rand(2, 4, 3) * 8 + 16
    with torch.no_grad():
        features = extractor(volume, coords)
    assert features.shape == (2, 4, 8)


def test_patch_centres_are_rounded_not_truncated():
    """A .long() cast biased every patch up to a voxel toward the origin."""
    from monai.networks.nets.regressor import Regressor

    extractor = PatchExtractor(
        patch_size=16,
        feature_extraction_model=Regressor(
            in_shape=(1, 16, 16, 16), out_shape=(8,), channels=(8, 16, 32), strides=(2, 2, 2), kernel_size=3
        ),
    ).eval()
    volume = torch.zeros(1, 1, 48, 48, 48)
    volume[0, 0, 25, 25, 25] = 1.0  # a single hot voxel at 25

    # 24.6 rounds to 25, so the hot voxel sits at the patch centre; truncation would
    # have put the centre at 24 and shifted it off-centre.
    patches = extractor.extract_patches(volume, torch.tensor([[[24.6, 24.6, 24.6]]]))
    centre = patches.shape[-1] // 2
    assert patches[0, 0, 0, centre, centre, centre] == 1.0
