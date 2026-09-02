"""Per-vertebra loss weighting in the neighbour-aware prediction module.

The neighbour datasets concatenate one equally sized landmark block per vertebra,
block 0 being the current one. `PoiNeighborPredictionModule` weights that block
against its neighbours. The weighting was previously gated on an `n_vertebrae` batch
key that no dataset produced, so it silently did nothing.
"""

from __future__ import annotations

import pytest
import torch

from verpex.loss.loss_modules import masked_weighted_mean
from verpex.modules.poi_module import PoiNeighborPredictionModule

N_PER_VERTEBRA = 4
N_VERTEBRAE = 3
N_LANDMARKS = N_PER_VERTEBRA * N_VERTEBRAE

COARSE_CONFIG = {
    "type": "SADenseNet",
    "params": {
        "in_channels": 1,
        "n_landmarks": N_LANDMARKS,
        "loss_fn": "L1",
        "feature_l": 16,
        "init_features": 8,
        "growth_rate": 4,
        "block_config": [2, 2],
        "bn_size": 2,
        "dropout_prob": 0.0,
        "lr": 1e-3,
    },
}


def make_module(current_weight=1.0, neighbor_weight=0.2):
    return PoiNeighborPredictionModule(
        coarse_config=COARSE_CONFIG,
        refinement_config={"type": "Identity", "params": {}},
        current_weight=current_weight,
        neighbor_weight=neighbor_weight,
    )


@pytest.fixture
def neighbor_batch():
    """A batch shaped like what SpineNeighborDataset produces."""
    generator = torch.Generator().manual_seed(0)
    return {
        "target": torch.rand(2, N_LANDMARKS, 3, generator=generator) * 8 + 16,
        "coarse_preds": torch.rand(2, N_LANDMARKS, 3, generator=generator) * 8 + 16,
        "refined_preds": torch.rand(2, N_LANDMARKS, 3, generator=generator) * 8 + 16,
        "loss_mask": torch.ones(2, N_LANDMARKS, dtype=torch.bool),
        "surface": None,
        "zoom": torch.ones(2, 3),
        "input": torch.randn(2, 1, 16, 16, 16, generator=generator),
        "n_pois_per_vertebra": N_PER_VERTEBRA,
        "n_vertebrae": N_VERTEBRAE,
    }


def test_the_weights_actually_change_the_loss(neighbor_batch):
    """The headline check: this is what silently did nothing before."""
    emphasise_current = make_module(1.0, 0.0)._calculate_multi_vertebrae_loss(neighbor_batch)
    treat_equally = make_module(1.0, 1.0)._calculate_multi_vertebrae_loss(neighbor_batch)
    assert not torch.isclose(emphasise_current, treat_equally)


def test_equal_weights_reproduce_the_flat_loss_exactly(neighbor_batch):
    """Weighting everything the same must not perturb the unweighted result."""
    module = make_module(1.0, 1.0)
    assert torch.equal(
        module._calculate_multi_vertebrae_loss(neighbor_batch),
        module._calculate_standard_loss(neighbor_batch),
    )


def test_zero_neighbour_weight_equals_the_current_block_alone(neighbor_batch):
    """A neighbour weight of 0 must be exactly the loss over block 0."""
    module = make_module(1.0, 0.0)
    current_only = neighbor_batch["loss_mask"].clone()
    current_only[:, N_PER_VERTEBRA:] = False
    assert torch.allclose(
        module._calculate_multi_vertebrae_loss(neighbor_batch),
        module._calculate_standard_loss({**neighbor_batch, "loss_mask": current_only}),
        atol=1e-6,
    )


def test_a_missing_neighbour_does_not_produce_nan(neighbor_batch):
    """Vertebrae at either end of the spine have an all-masked neighbour block.

    A per-block mean over an empty selection returns NaN, which would poison the
    whole batch's loss - and missing neighbours are routine, not an edge case.
    """
    mask = neighbor_batch["loss_mask"].clone()
    mask[:, 2 * N_PER_VERTEBRA :] = False
    loss = make_module()._calculate_multi_vertebrae_loss({**neighbor_batch, "loss_mask": mask})
    assert torch.isfinite(loss)


def test_a_batch_with_nothing_valid_gives_zero_not_nan(neighbor_batch):
    mask = torch.zeros_like(neighbor_batch["loss_mask"])
    loss = make_module()._calculate_multi_vertebrae_loss({**neighbor_batch, "loss_mask": mask})
    assert torch.isfinite(loss) and loss == 0.0


def test_a_single_vertebra_batch_falls_back_to_the_flat_loss(neighbor_batch):
    """Batches without the per-vertebra layout must be unaffected."""
    plain = {k: v for k, v in neighbor_batch.items() if k not in ("n_vertebrae", "n_pois_per_vertebra")}
    module = make_module()
    assert torch.equal(module._calculate_multi_vertebrae_loss(plain), module._calculate_standard_loss(plain))


def test_an_inconsistent_block_layout_is_rejected(neighbor_batch):
    """Declared blocks must account for exactly the landmarks present."""
    with pytest.raises(ValueError, match="but carries"):
        make_module()._landmark_weights({**neighbor_batch, "n_pois_per_vertebra": N_PER_VERTEBRA + 1})


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        make_module(1.0, -0.5)


def test_weights_mark_the_first_block_as_current(neighbor_batch):
    weights = make_module(1.0, 0.25)._landmark_weights(neighbor_batch)
    assert weights.shape == (2, N_LANDMARKS)
    assert (weights[:, :N_PER_VERTEBRA] == 1.0).all()
    assert (weights[:, N_PER_VERTEBRA:] == 0.25).all()


def test_per_vertebra_metrics_are_reported_in_millimetres(neighbor_batch):
    """They were unreachable before, and computed raw voxel distances."""
    module = make_module()
    at_1mm = module._calculate_multi_vertebrae_metrics(neighbor_batch, "val")
    at_2mm = module._calculate_multi_vertebrae_metrics({**neighbor_batch, "zoom": torch.full((2, 3), 2.0)}, "val")
    assert at_2mm["current_vertebra_mean_distance_val"] == pytest.approx(2.0 * at_1mm["current_vertebra_mean_distance_val"], rel=1e-5)


class TestMaskedWeightedMean:
    """The reduction every loss now shares."""

    def test_unweighted_matches_a_plain_masked_mean(self):
        values = torch.randn(3, 5, 3)
        mask = torch.rand(3, 5) > 0.4
        assert torch.equal(masked_weighted_mean(values, mask), values[mask].mean())

    def test_uniform_weights_match_the_unweighted_mean(self):
        values = torch.randn(3, 5)
        mask = torch.ones(3, 5, dtype=torch.bool)
        assert masked_weighted_mean(values, mask, torch.ones(3, 5)) == pytest.approx(values.mean().item())

    def test_zero_weight_entries_are_excluded(self):
        values = torch.tensor([[1.0, 100.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        weights = torch.tensor([[1.0, 0.0]])
        assert masked_weighted_mean(values, mask, weights) == pytest.approx(1.0)

    def test_an_empty_selection_is_zero_not_nan(self):
        values = torch.randn(2, 4)
        assert masked_weighted_mean(values, torch.zeros(2, 4, dtype=torch.bool)) == 0.0

    def test_all_weights_zero_is_zero_not_nan(self):
        values = torch.randn(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        assert masked_weighted_mean(values, mask, torch.zeros(2, 4)) == 0.0

    def test_the_result_stays_differentiable_when_empty(self):
        """An empty batch must not detach the graph, or the step would fail."""
        values = torch.randn(2, 4, requires_grad=True)
        masked_weighted_mean(values, torch.zeros(2, 4, dtype=torch.bool)).backward()
        assert values.grad is not None
