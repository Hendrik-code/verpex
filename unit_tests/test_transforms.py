"""Landmark-aware augmentation transforms.

A flip must move the image and the landmark coordinates together, and it must remap
left/right landmark pairs so a "left pedicle" stays a left pedicle after the volume is
mirrored.
"""

from __future__ import annotations

import pytest
import torch

from verpex.data.transforms import (
    LandMarksRandHorizontalFlipNeighbor,
    LandmarksRandLabelDropout,
    LandmarksRandLimitedFov,
    RandLabelCollapse,
    create_transforms,
)
from verpex.registry import UnknownTypeError

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


# ---------------------------------------------------------------------------
# Limited field of view, label dropout, label collapse, and the registry.
# ---------------------------------------------------------------------------

FOV_SHAPE = (16, 20, 24)


def make_fov_sample(shape=FOV_SHAPE, label=3.0):
    """Build a sample whose labelled bar hugs the high face of axis 2.

    Channel 0 is the label map, channel 1 a surface-like copy. The landmarks run along
    the bar, one per slice, so it is obvious which of them a cut removes.
    """
    volume = torch.zeros(2, *shape)
    volume[0, 4:12, 6:14, 10 : shape[2]] = label
    volume[1] = (volume[0] > 0).float()
    target = torch.tensor([[8.0, 10.0, float(z)] for z in range(10, shape[2])])
    return {
        "input": volume,
        "target": target,
        "target_indices": torch.arange(len(target)),
        "loss_mask": torch.ones(len(target), dtype=torch.bool),
        "label_channel": 0,
        "landmark_labels": torch.full((len(target),), int(label)),
        "sample_index": 0,
    }


def clone_sample(sample, **overrides):
    """Deep-enough copy so a transform's in-place writes cannot leak between cases."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}
    out.update(overrides)
    return out


def cut_axis2_at_both_ends(**kwargs):
    """A fully determined transform: it removes 20% of each end of axis 2 and nothing else.

    Both ends rather than one because which single end gets cut is itself a random draw,
    and these tests want the cut fixed, not the seed.
    """
    params = {
        "prob": (0.0, 0.0, 1.0),
        "cut_range": ((0.0, 0.0), (0.0, 0.0), (0.2, 0.2)),
        "both_sides_prob": 1.0,
        "min_visible_landmarks": 1,
    }
    params.update(kwargs)
    return LandmarksRandLimitedFov(**params)


def test_limited_fov_keeps_the_spatial_shape():
    """The model's input size is fixed, so the transform may zero but never resize."""
    sample = make_fov_sample()
    out = cut_axis2_at_both_ends()(clone_sample(sample))
    assert out["input"].shape == sample["input"].shape


def test_limited_fov_zeroes_the_slab_it_removes():
    """After the cut and the shift, no labelled voxel may survive from the removed slab."""
    sample = make_fov_sample()
    out = cut_axis2_at_both_ends()(clone_sample(sample))
    kept = int((out["input"][0] > 0).sum())
    # 20% of 24 is 4.8 -> 5 slices removed from a 14-slice bar.
    assert kept == int((sample["input"][0] > 0).sum()) * 9 // 14


def test_limited_fov_does_not_wrap_content_around_the_volume_edge():
    """A circular shift would put anatomy back at the face it was pushed off.

    ``torch.roll`` is the obvious way to translate a volume and it is wrong here: the
    landmark coordinate lands back *inside* the volume, on the wrong structure, so no
    bounds check catches it. The re-centring shift is driven by the label channel, and it
    moves the uncut axes too - which is where content actually can wrap.

    Here the label sits high on axis 0, so re-centring shifts by -4 along it, and a marker
    fills the four slices that a wrap would deposit at the opposite face.
    """
    volume = torch.zeros(2, *FOV_SHAPE)
    volume[0, 10:15, 8:12, 8:16] = 3.0
    volume[1, 0:4] = 7.0
    sample = {
        "input": volume,
        "target": torch.tensor([[12.0, 10.0, 12.0]]),
        "loss_mask": torch.ones(1, dtype=torch.bool),
        "label_channel": 0,
    }
    out = cut_axis2_at_both_ends()(sample)
    assert float(out["input"][1][-4:].sum()) == 0.0


def test_limited_fov_moves_the_volume_and_the_landmarks_by_the_same_delta():
    """A sign error in the shift would leave the landmarks off their own anatomy."""
    out = cut_axis2_at_both_ends()(clone_sample(make_fov_sample()))
    kept = out["target"][out["loss_mask"]].round().long()
    on_label = out["input"][0][kept[:, 0], kept[:, 1], kept[:, 2]]
    assert bool((on_label > 0).all())


def test_limited_fov_centres_the_surviving_label_on_the_volume_centre():
    """Re-centring must reproduce what the offline cutout does to a truncated scan."""
    out = cut_axis2_at_both_ends()(clone_sample(make_fov_sample()))
    indices = torch.nonzero(out["input"][0] > 0)
    centre = ((indices.amin(0) + indices.amax(0)) // 2).tolist()
    assert centre == [size // 2 for size in FOV_SHAPE]


def test_limited_fov_masks_landmarks_that_fall_outside_the_kept_slab():
    """Supervising a landmark whose anatomy was cut away teaches hallucination."""
    out = cut_axis2_at_both_ends()(clone_sample(make_fov_sample()))
    assert int(out["loss_mask"].sum()) == 9


def test_limited_fov_intersects_an_existing_loss_mask_rather_than_replacing_it():
    """A landmark the dataset already excluded must not come back supervised."""
    sample = make_fov_sample()
    sample["loss_mask"][0] = False
    out = cut_axis2_at_both_ends()(clone_sample(sample))
    assert not bool(out["loss_mask"][0])


def test_limited_fov_writes_a_loss_mask_when_the_sample_has_none():
    """Verpex's own datasets build the mask after the transform, so it has to be carried."""
    sample = make_fov_sample()
    del sample["loss_mask"]
    out = cut_axis2_at_both_ends()(sample)
    assert out["loss_mask"].dtype == torch.bool
    assert int(out["loss_mask"].sum()) == 9


def test_limited_fov_is_a_noop_when_no_axis_is_cut():
    """Re-centring unconditionally would undo the affine's deliberate off-centre shift."""
    sample = make_fov_sample()
    out = LandmarksRandLimitedFov(prob=(0.0, 0.0, 0.0))(clone_sample(sample))
    assert torch.equal(out["input"], sample["input"])
    assert torch.equal(out["target"], sample["target"])


def test_limited_fov_is_a_noop_when_too_few_landmarks_would_survive():
    """A sample whose supervision the cut would wipe out is worth nothing; leave it whole."""
    sample = make_fov_sample()
    out = cut_axis2_at_both_ends(min_visible_landmarks=999)(clone_sample(sample))
    assert torch.equal(out["input"], sample["input"])
    assert torch.equal(out["loss_mask"], sample["loss_mask"])


def test_limited_fov_reads_the_label_channel_the_sample_names():
    """The stack order depends on input_data_type, so a hardcoded -1 is silently wrong.

    Here channel 1 holds a bar offset from channel 0's. Re-centring on the named channel
    must put *that* bar on the volume centre.
    """
    sample = make_fov_sample()
    sample["input"][1] = 0.0
    sample["input"][1, 2:6, 2:6, 0:8] = 5.0
    out = cut_axis2_at_both_ends(label_channel=0)(clone_sample(sample, label_channel=1))
    indices = torch.nonzero(out["input"][1] > 0)
    assert ((indices.amin(0) + indices.amax(0)) // 2).tolist() == [size // 2 for size in FOV_SHAPE]


def test_limited_fov_with_a_seed_gives_the_same_cut_for_the_same_sample():
    """An evaluation metric is only comparable across epochs if the cut is repeatable."""
    transform = LandmarksRandLimitedFov(seed=11, min_visible_landmarks=1)
    torch.manual_seed(1)
    first = transform(clone_sample(make_fov_sample()))
    torch.manual_seed(999)
    second = transform(clone_sample(make_fov_sample()))
    assert torch.equal(first["input"], second["input"])
    assert torch.equal(first["loss_mask"], second["loss_mask"])


def test_limited_fov_with_a_seed_gives_a_different_cut_for_a_different_sample():
    """Every scan getting the identical cut would measure one truncation, not a range."""
    transform = LandmarksRandLimitedFov(seed=11, min_visible_landmarks=1)
    outs = [transform(clone_sample(make_fov_sample(), sample_index=i)) for i in range(8)]
    assert any(not torch.equal(outs[0]["input"], other["input"]) for other in outs[1:])


def test_limited_fov_without_a_seed_varies_across_calls():
    """Training wants a fresh cut every epoch, which is what the global RNG gives."""
    transform = LandmarksRandLimitedFov(min_visible_landmarks=1)
    torch.manual_seed(0)
    outs = [transform(clone_sample(make_fov_sample())) for _ in range(8)]
    assert any(not torch.equal(outs[0]["input"], other["input"]) for other in outs[1:])


@pytest.mark.parametrize("bad", [{"prob": (0.5, 0.5)}, {"prob": (2.0, 0, 0)}, {"cut_range": ((0.6, 0.7),) * 3}, {"min_keep_fraction": 0}])
def test_limited_fov_rejects_a_malformed_range(bad):
    """A bad range would otherwise degrade to a silent no-op for a whole training run."""
    with pytest.raises(ValueError):
        LandmarksRandLimitedFov(**bad)


def test_label_dropout_removes_the_label_from_every_channel():
    """A field of view that lost a rib lost its surface and its intensities too."""
    sample = make_fov_sample()
    sample["input"][0, 0:3, 0:3, 0:3] = 9.0
    sample["input"][1, 0:3, 0:3, 0:3] = 1.0
    out = LandmarksRandLabelDropout(prob=1.0, n_labels=(1, 1), end_bias=1.0)(clone_sample(sample))
    present = torch.unique(out["input"][0])
    dropped = {3.0, 9.0} - set(present.tolist())
    assert len(dropped) == 1
    value = dropped.pop()
    assert float(out["input"][1][sample["input"][0] == value].sum()) == 0.0


def test_label_dropout_unsupervises_the_landmarks_of_the_label_it_removed():
    """Asking the model to find anatomy absent from its own input teaches hallucination."""
    sample = make_fov_sample()
    sample["input"][0, 0:3, 0:3, 0:3] = 9.0
    out = LandmarksRandLabelDropout(prob=1.0, n_labels=(1, 1), end_bias=1.0)(clone_sample(sample))
    gone = 3.0 not in torch.unique(out["input"][0]).tolist()
    assert int(out["loss_mask"].sum()) == (0 if gone else len(sample["target"]))


def test_label_dropout_never_empties_the_input():
    """A volume with no anatomy left has nothing to learn from."""
    sample = make_fov_sample()
    out = LandmarksRandLabelDropout(prob=1.0, n_labels=(1, 5))(clone_sample(sample))
    assert bool((out["input"][0] > 0).any())


def test_label_collapse_makes_every_label_value_identical():
    """The point is to remove the identity the model would otherwise read off the value."""
    sample = make_fov_sample()
    sample["input"][0, 0:3, 0:3, 0:3] = 9.0
    out = RandLabelCollapse(prob=1.0, channels=[0])({"input": sample["input"].clone()})
    assert sorted(torch.unique(out["input"][0]).tolist()) == [0.0, 1.0]


def test_label_collapse_leaves_the_channels_it_was_not_given():
    """It must not reach a bookkeeping copy of the label map the loss mask comes from."""
    sample = make_fov_sample()
    out = RandLabelCollapse(prob=1.0, channels=[0])({"input": sample["input"].clone()})
    assert torch.equal(out["input"][1], sample["input"][1])


def test_label_collapse_without_channels_is_an_error_not_a_silent_noop():
    """Quietly doing nothing for a whole run is the worst outcome for an ablation."""
    with pytest.raises(KeyError, match="label channels"):
        RandLabelCollapse(prob=1.0)({"input": torch.zeros(1, 4, 4, 4)})


def test_a_single_transform_config_still_builds_one_transform():
    """Every config written before the pipeline became a list must keep working."""
    built = create_transforms({"type": "LandmarksRandLimitedFov"})
    assert len(built) == 1
    assert isinstance(built[0], LandmarksRandLimitedFov)


def test_a_list_of_transform_configs_builds_them_in_order():
    """Order is part of the meaning: the field-of-view cut has to run after the affine."""
    built = create_transforms([{"type": "LandmarksRandLimitedFov"}, {"type": "RandLabelCollapse"}])
    assert [type(t) for t in built] == [LandmarksRandLimitedFov, RandLabelCollapse]


def test_no_transform_config_builds_nothing():
    """The datasets always take a `transforms=` argument, so None must survive."""
    assert create_transforms(None) is None


def test_an_unregistered_transform_type_names_the_registered_ones():
    """The old factory raised a bare ValueError and could reach only two of the classes."""
    with pytest.raises(UnknownTypeError, match="LandmarksRandLimitedFov"):
        create_transforms({"type": "NotATransform"})
