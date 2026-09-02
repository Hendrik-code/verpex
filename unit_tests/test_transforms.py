"""Landmark-aware augmentation transforms.

A flip must move the image and the landmark coordinates together, and it must remap
left/right landmark pairs so a "left pedicle" stays a left pedicle after the volume is
mirrored.
"""

from __future__ import annotations

import pytest
import torch

from verpex.data.transforms import LandMarksRandHorizontalFlipNeighbor

#: 81/82 are a left/right pair; the rest are midline landmarks that map to themselves.
FLIP_PAIRS = {81: 82, 82: 81, 90: 90, 91: 91}


def landmark_ids(n_per_vertebra: int) -> list[int]:
    """Return `n_per_vertebra` landmark ids, starting with one left/right pair."""
    return [81, 82, 90, 91, *range(200, 200 + n_per_vertebra - 4)]


def make_batch(n_per_vertebra: int, n_vertebrae: int = 3) -> tuple[dict, dict]:
    """Build a sample whose landmark rows encode (block index, landmark id).

    Columns 1 and 2 are untouched by the flip, so they can be read afterwards to see
    exactly where each landmark ended up.
    """
    ids = landmark_ids(n_per_vertebra)
    flip_pairs = dict(FLIP_PAIRS)
    flip_pairs.update({i: i for i in range(200, 200 + n_per_vertebra - 4)})
    batch = {
        "target_indices": torch.tensor(ids * n_vertebrae),
        "target": torch.tensor([[0.0, float(block), float(poi)] for block in range(n_vertebrae) for poi in ids]),
        "input": torch.zeros(1, 10, 10, 10),
    }
    return batch, flip_pairs


# 35 was the value hardcoded before; 45 is what include_com=True produces.
@pytest.mark.parametrize("n_per_vertebra", [4, 6, 35, 45])
def test_flip_never_mixes_landmarks_between_vertebrae(n_per_vertebra):
    """Each vertebra's landmarks must be remapped within its own block.

    The block size used to be hardcoded to 35, so at any other landmark count the
    per-block remapping misaligned - silently, since it still produced a list of the
    right length. At 45 landmarks per vertebra, 20 of 135 landmarks were duplicated,
    20 were dropped, and 20 were attributed to the wrong vertebra.
    """
    batch, flip_pairs = make_batch(n_per_vertebra)
    out = LandMarksRandHorizontalFlipNeighbor(prob=1.0, flip_pairs=flip_pairs)(batch)
    blocks = out["target"][:, 1].tolist()
    assert blocks == [float(block) for block in range(3) for _ in range(n_per_vertebra)]


@pytest.mark.parametrize("n_per_vertebra", [4, 6, 35, 45])
def test_flip_is_a_true_permutation(n_per_vertebra):
    """Every landmark must survive exactly once - none duplicated, none dropped."""
    batch, flip_pairs = make_batch(n_per_vertebra)
    expected = sorted(batch["target"][:, 2].tolist())
    out = LandMarksRandHorizontalFlipNeighbor(prob=1.0, flip_pairs=flip_pairs)(batch)
    assert sorted(out["target"][:, 2].tolist()) == expected


@pytest.mark.parametrize("n_per_vertebra", [4, 45])
def test_left_right_pairs_swap_in_every_block(n_per_vertebra):
    batch, flip_pairs = make_batch(n_per_vertebra)
    out = LandMarksRandHorizontalFlipNeighbor(prob=1.0, flip_pairs=flip_pairs)(batch)
    ids_out = out["target"][:, 2].tolist()
    for block in range(3):
        start = block * n_per_vertebra
        assert (ids_out[start], ids_out[start + 1]) == (82.0, 81.0)


def test_landmark_count_that_does_not_divide_evenly_is_rejected():
    """Better to fail than to silently misalign, which is what used to happen."""
    batch = {
        "target_indices": torch.tensor([81, 82, 90, 91, 81]),
        "target": torch.zeros(5, 3),
        "input": torch.zeros(1, 10, 10, 10),
    }
    with pytest.raises(ValueError, match="not divisible"):
        LandMarksRandHorizontalFlipNeighbor(prob=1.0, flip_pairs=FLIP_PAIRS)(batch)


def test_probability_zero_leaves_the_sample_untouched():
    batch, flip_pairs = make_batch(4)
    before = batch["target"].clone()
    out = LandMarksRandHorizontalFlipNeighbor(prob=0.0, flip_pairs=flip_pairs)(batch)
    assert torch.equal(out["target"], before)


def test_flipping_twice_restores_the_original():
    """The flip is its own inverse, for both the image and the landmarks."""
    batch, flip_pairs = make_batch(4)
    before_target = batch["target"].clone()
    before_input = batch["input"].clone()
    flip = LandMarksRandHorizontalFlipNeighbor(prob=1.0, flip_pairs=flip_pairs)
    out = flip(flip(batch))
    assert torch.equal(out["target"], before_target)
    assert torch.equal(out["input"], before_input)
