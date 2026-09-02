"""Loss functions: masking, resolution by name, and compound weighting."""

from __future__ import annotations

import pytest
import torch

from verpex.loss.loss_modules import CompoundLoss, L1LossMasked, get_loss_fn


@pytest.mark.parametrize("name", ["L1", "L2", "WingLoss", "SD"])
def test_every_registered_loss_resolves(name):
    assert get_loss_fn(name) is not None


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="Unknown loss function"):
        get_loss_fn("NotALoss")


def test_a_list_builds_an_evenly_weighted_compound():
    loss = get_loss_fn(["L1", "L2"])
    assert isinstance(loss, CompoundLoss)
    assert loss.weights == [0.5, 0.5]


def test_masked_entries_do_not_affect_the_loss():
    """Landmarks excluded by the loss mask must not contribute."""
    loss_fn = L1LossMasked()
    pred = torch.zeros(1, 3, 3)
    target = torch.zeros(1, 3, 3)
    target[0, 2] = 1000.0  # a wildly wrong landmark ...
    mask = torch.tensor([[True, True, False]])  # ... that is masked out
    assert loss_fn(pred, target, mask) == pytest.approx(0.0)


def test_unmasked_error_is_reported():
    loss_fn = L1LossMasked()
    pred = torch.zeros(1, 2, 3)
    target = torch.ones(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert loss_fn(pred, target, mask) == pytest.approx(1.0)


def test_perfect_prediction_has_zero_loss():
    for name in ("L1", "L2"):
        loss_fn = get_loss_fn(name)
        coords = torch.rand(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        assert loss_fn(coords, coords.clone(), mask) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ["L1", "L2"])
def test_masked_losses_work_without_a_mask(name):
    """`mask` is optional, so omitting it must not raise.

    Both masked losses previously assigned the reduced value only inside
    `if mask is not None`, then returned it unconditionally - an UnboundLocalError
    on any unmasked call.
    """
    loss_fn = get_loss_fn(name)
    value = loss_fn(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
    assert value == pytest.approx(1.0)


def test_compound_weights_that_sum_to_one_in_decimal_are_accepted():
    """`sum([0.7, 0.2, 0.1])` is 0.9999999999999999, which an exact check rejected."""
    loss = CompoundLoss([L1LossMasked()] * 3, weights=[0.7, 0.2, 0.1])
    assert loss.weights == [0.7, 0.2, 0.1]


def test_compound_weights_that_really_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        CompoundLoss([L1LossMasked()] * 2, weights=[0.5, 0.9])


def test_default_compound_weights_sum_to_one():
    """They previously defaulted to 1.0 each, contradicting the check on explicit weights."""
    loss = CompoundLoss([L1LossMasked()] * 4)
    assert sum(loss.weights) == pytest.approx(1.0)
