"""Contracts for the unified refinement module.

`PatchTransformer` absorbed seven ablation subclasses that differed only in which
inputs they feed the transformer. These tests pin down that the ablation *names*
still build the same architectures, and that the flags actually change what the
module consumes.
"""

from __future__ import annotations

import pytest
import torch

from verpex.modules.refinement import REFINEMENT_MODULES

ABLATIONS = [name for name in REFINEMENT_MODULES if name != "Identity"]


@pytest.mark.parametrize("name", ABLATIONS)
def test_variant_builds_and_produces_refined_predictions(name, model_dims, refinement_batch):
    module = REFINEMENT_MODULES[name](**model_dims).eval()
    with torch.no_grad():
        out = module(refinement_batch)
    refined = out["refined_preds"]
    assert refined.shape == refinement_batch["coarse_preds"].shape
    assert torch.isfinite(refined).all()


@pytest.mark.parametrize("name", ABLATIONS)
def test_variant_computes_a_finite_loss(name, model_dims, refinement_batch):
    module = REFINEMENT_MODULES[name](**model_dims).eval()
    with torch.no_grad():
        batch = module(refinement_batch)
        loss = module.calculate_loss(dict(batch, surface=None))
    assert torch.isfinite(torch.as_tensor(loss)).all()


def test_refined_predictions_keep_sub_voxel_precision(model_dims, refinement_batch):
    """Coarse predictions must not be truncated to integers.

    The ablation variants used to cast coarse predictions with `.long()`, which
    discarded the fractional part and capped refinement at whole-voxel accuracy.
    """
    module = REFINEMENT_MODULES["NoVertPatchTransformer"](**model_dims).eval()
    with torch.no_grad():
        out = module(refinement_batch)
    offsets = out["refined_preds"] - out["offsets"]
    fractional = offsets - offsets.floor()
    assert fractional.abs().sum() > 0, "coarse predictions were truncated to integers"


def test_loss_scales_with_voxel_spacing(model_dims, refinement_batch):
    """The loss is computed in millimetres, so anisotropic spacing must matter.

    Three variants used to skip the zoom multiplication and optimise in voxels,
    making their numbers incomparable with the main model's.
    """
    module = REFINEMENT_MODULES["NoPoiPatchTransformer"](**model_dims).eval()
    with torch.no_grad():
        batch = module(refinement_batch)
        isotropic = module.calculate_loss(dict(batch, surface=None, zoom=torch.ones(2, 3)))
        anisotropic = module.calculate_loss(dict(batch, surface=None, zoom=torch.full((2, 3), 3.0)))
    assert not torch.isclose(torch.as_tensor(isotropic), torch.as_tensor(anisotropic))


def test_disabling_embeddings_shrinks_the_model(model_dims):
    """The use_* flags must actually remove parameters, not just gate them."""
    full = REFINEMENT_MODULES["PatchTransformer"](**model_dims)
    no_poi = REFINEMENT_MODULES["NoPoiPatchTransformer"](**model_dims)
    no_both = REFINEMENT_MODULES["NoPoiVertPatchTransformer"](**model_dims)

    def n(module):
        return sum(p.numel() for p in module.parameters())

    assert n(no_poi) < n(full)
    assert n(no_both) < n(no_poi)


def test_no_coarse_pred_variant_ignores_coarse_predictions(model_dims, refinement_batch):
    """NoCoarsePredTransformer regresses coordinates instead of refining them."""
    module = REFINEMENT_MODULES["NoCoarsePredTransformer"](**model_dims).eval()
    with torch.no_grad():
        baseline = module(dict(refinement_batch))["refined_preds"]
        shifted = module(dict(refinement_batch, coarse_preds=refinement_batch["coarse_preds"] + 5))["refined_preds"]
    assert torch.allclose(baseline, shifted)


def test_a_variant_without_any_feature_source_is_rejected(model_dims):
    """Catch the empty-token config at construction, not mid-training.

    With both feature sources off the transformer receives an empty feature list,
    which used to surface as an IndexError on the first forward pass.
    """
    with pytest.raises(ValueError, match="no per-landmark features"):
        REFINEMENT_MODULES["PatchTransformer"](**model_dims, use_coarse_features=False, use_patches=False)


def test_a_config_cannot_contradict_the_variant_it_named(model_dims):
    """`NoPoiPatchTransformer` means "no POI embedding"; re-enabling it is a mistake.

    Silently honouring the kwarg would build a model that does not match the type
    string recorded alongside its checkpoint.
    """
    with pytest.raises(ValueError, match="already fixes use_poi_embedding"):
        REFINEMENT_MODULES["NoPoiPatchTransformer"](**model_dims, use_poi_embedding=True)


def test_variants_still_accept_ordinary_parameters(model_dims):
    """The conflict check must not block parameters the variant does not fix."""
    module = REFINEMENT_MODULES["NoPoiPatchTransformer"](**{**model_dims, "dropout": 0.5})
    assert module is not None


def test_warmup_does_not_need_lightning_to_supply_an_epoch(model_dims, refinement_batch):
    """`warmup_epochs > 0` must work.

    PatchTransformer is a plain nn.Module, so Lightning gives it no `current_epoch`;
    the warmup branch used to raise AttributeError on the first training step. The
    parent module now propagates the epoch, and the default keeps a standalone
    module usable.
    """
    module = REFINEMENT_MODULES["PatchTransformer"](**model_dims, warmup_epochs=5)
    module.train()
    out = module(refinement_batch)
    assert torch.isfinite(out["refined_preds"]).all()


def test_current_epoch_stays_out_of_the_state_dict(model_dims):
    """It is bookkeeping, not a parameter - it must not change checkpoint contents."""
    module = REFINEMENT_MODULES["PatchTransformer"](**model_dims, warmup_epochs=5)
    assert not any("current_epoch" in key for key in module.state_dict())


def test_no_coarse_pred_requires_coarse_features(model_dims):
    """Without a coarse stage the coarse features are the only feature source."""
    with pytest.raises(ValueError, match="no other feature source"):
        REFINEMENT_MODULES["PatchTransformer"](**model_dims, use_coarse_pred=False, use_patches=False, use_coarse_features=False)
