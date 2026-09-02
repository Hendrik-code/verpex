"""The coarse feature-extraction modules must agree on units.

All four report distances and compute losses in millimetres. The two sparse
backbones previously worked in raw voxels, so their numbers were not comparable
with the dense ones or with the refinement stage logged beside them.
"""

from __future__ import annotations

import pytest
import torch

from verpex.modules.feature_extraction import FEATURE_EXTRACTION_MODULES

#: The sparse backbones need spconv and a CUDA toolchain; skipped where unavailable.
DENSE_MODULES = ["SADenseNet", "HeatmapFeatureDenseNet"]

N_LANDMARKS = 4


def make_module(name):
    return FEATURE_EXTRACTION_MODULES[name](
        in_channels=1,
        n_landmarks=N_LANDMARKS,
        loss_fn="L1",
        feature_l=16,
        init_features=8,
        growth_rate=4,
        block_config=[2, 2],
        bn_size=2,
        dropout_prob=0.0,
        lr=1e-3,
    )


def make_batch(zoom_value=1.0):
    generator = torch.Generator().manual_seed(0)
    return {
        "target": torch.rand(2, N_LANDMARKS, 3, generator=generator) * 8 + 16,
        "coarse_preds": torch.rand(2, N_LANDMARKS, 3, generator=generator) * 8 + 16,
        "loss_mask": torch.ones(2, N_LANDMARKS, dtype=torch.bool),
        "surface": None,
        "zoom": torch.full((2, 3), zoom_value),
        "input": torch.randn(2, 1, 16, 16, 16, generator=generator),
        "target_indices": torch.arange(N_LANDMARKS).expand(2, N_LANDMARKS),
    }


@pytest.mark.parametrize("name", DENSE_MODULES)
def test_loss_scales_with_voxel_spacing(name):
    """Doubling the spacing must double the loss - i.e. it is computed in mm."""
    module = make_module(name)
    at_1mm = module.calculate_loss(make_batch(1.0))
    at_2mm = module.calculate_loss(make_batch(2.0))
    assert at_2mm.item() == pytest.approx(2.0 * at_1mm.item(), rel=1e-5)


@pytest.mark.parametrize("name", DENSE_MODULES)
def test_reported_distances_scale_with_voxel_spacing(name):
    module = make_module(name)
    at_1mm = module.calculate_metrics(make_batch(1.0), "val")["coarse_mean_distance_val"]
    at_2mm = module.calculate_metrics(make_batch(2.0), "val")["coarse_mean_distance_val"]
    assert at_2mm.item() == pytest.approx(2.0 * at_1mm.item(), rel=1e-5)


@pytest.mark.parametrize("name", DENSE_MODULES)
def test_loss_works_without_a_surface_when_projection_is_off(name):
    """`surface` is read unconditionally now, so a None surface must be fine."""
    assert torch.isfinite(make_module(name).calculate_loss(make_batch()))


@pytest.mark.parametrize("name", DENSE_MODULES)
def test_the_modules_agree_with_each_other(name):
    """Given identical inputs, unit-consistent modules give the same scale of loss."""
    reference = make_module("SADenseNet").calculate_loss(make_batch())
    other = make_module(name).calculate_loss(make_batch())
    assert other.item() == pytest.approx(reference.item(), rel=0.5)
