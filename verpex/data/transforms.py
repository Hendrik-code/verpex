"""Data augmentation transforms for landmark-annotated volumes.

The project's own transforms are at the top of this module: they apply an
augmentation to the image *and* carry the landmark coordinates through the same
transform, which the stock MONAI transforms do not do.

Below the marked divider is a vendored copy of MONAI's affine-transform internals,
adapted so the sampled affine can be recovered and applied to the landmarks. It is
kept close to upstream so it stays easy to diff against MONAI; it is excluded from
this project's docstring lint for that reason (see per-file-ignores in pyproject.toml).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.networks.layers import grid_pull
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.functional import affine_func
from monai.transforms.transform import (
    LazyTransform,
    Randomizable,
    RandomizableTransform,
    Transform,
)
from monai.transforms.utils import (
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    resolves_modes,
)
from monai.transforms.utils_pytorch_numpy_unification import linalg_inv, moveaxis
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.enums import TraceKeys, TransformBackends
from monai.utils.type_conversion import convert_data_type, get_equivalent_dtype

from verpex.registry import build


class LandmarksRandAffine:
    def __init__(
        self,
        prob,
        rotate_range,
        shear_range,
        translate_range,
        scale_range,
        device="cpu",
    ):
        self.prob = prob
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.translate_range = translate_range
        self.scale_range = scale_range

        self.image_transform = RandAffine(
            prob=prob,
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            mode="nearest",
            padding_mode="zeros",
            device=device,
        )

    def __call__(self, dd):
        volume = dd["input"]
        landmarks = dd["target"]

        # Apply MonAI's RandAffine to the volume
        transformed_volume, affine_matrix = self.image_transform(volume)

        # Convert landmarks to homogeneous coordinates
        ones = torch.ones(landmarks.shape[0], 1, dtype=landmarks.dtype, device=landmarks.device)
        homogeneous_landmarks = torch.cat([landmarks, ones], dim=1)

        # Apply the affine transformation to the landmarks
        transformed_landmarks = torch.mm(
            homogeneous_landmarks,
            torch.linalg.inv(torch.tensor(affine_matrix, dtype=torch.float).t()),
        )[:, :3]

        dd["input"] = transformed_volume
        dd["target"] = transformed_landmarks

        return dd


class LandMarksRandHorizontalFlip:
    def __init__(self, prob, flip_pairs, device="cpu"):
        self.prob = prob
        self.flip_pairs = flip_pairs

    def __call__(self, dd):
        if torch.rand(1) < self.prob:
            target_indices = dd["target_indices"]

            # Flip the volume horizontally, since the orientation is LAS (Left, Anterior, Superior), this means flipping along dim 1 (dim 0 is channel dim)
            x = torch.flip(dd["input"], dims=[1])
            x_swapped = x.clone()

            # Swap the labels in the seg mask
            for label1, label2 in [(43, 44), (45, 46), (47, 48)]:
                x_swapped[x == label1] = label2
                x_swapped[x == label2] = label1

            dd["input"] = x_swapped

            # Flip the landmarks horizontally
            dd["target"][:, 0] = dd["input"].shape[1] - dd["target"][:, 0]

            # Reorder the landmarks according to the swap indices
            indices_map = {k.item(): v for v, k in enumerate(target_indices)}
            new_positions = [indices_map[self.flip_pairs[k.item()]] for k in target_indices]

            dd["target"] = dd["target"][new_positions]

        return dd


class LandMarksRandHorizontalFlipNeighbor:
    """Random left-right flip for a sample holding several vertebrae's landmarks.

    The neighbour datasets concatenate one block of landmarks per vertebra (current,
    upper neighbour, lower neighbour). A left-right flip has to remap each block
    independently, because ``flip_pairs`` is defined within a single vertebra.

    Args:
        prob: Probability of flipping a given sample.
        flip_pairs: Maps each landmark id to the id it becomes under a flip.
        n_vertebrae: Number of concatenated per-vertebra blocks in ``target_indices``.
        device: Unused; accepted for signature compatibility with the other transforms.
    """

    def __init__(self, prob, flip_pairs, n_vertebrae=3, device="cpu"):
        self.prob = prob
        self.flip_pairs = flip_pairs
        self.n_vertebrae = n_vertebrae

    def __call__(self, dd):
        """Flip ``dd`` in place with probability ``prob`` and return it.

        Raises:
            ValueError: If the landmark count is not divisible into ``n_vertebrae``
                equal blocks, which would silently misalign the remapping.
        """
        if torch.rand(1) < self.prob:
            target_indices = dd["target_indices"]

            n_total = len(target_indices)
            if n_total % self.n_vertebrae:
                raise ValueError(
                    f"Expected {self.n_vertebrae} equally sized per-vertebra landmark blocks, "
                    f"but got {n_total} landmarks, which is not divisible by {self.n_vertebrae}."
                )
            # Block size is derived, not hardcoded: it was fixed at 35, so any other
            # landmark count (include_com adds 10 per vertebra) silently misaligned
            # the per-block remapping below.
            block = n_total // self.n_vertebrae

            # Orientation is LAS, so left-right is dim 1 of the volume (dim 0 is channels).
            x = torch.flip(dd["input"], dims=[1])
            x_swapped = x.clone()

            # Swap the left/right subregion labels in the segmentation mask
            for label1, label2 in [(43, 44), (45, 46), (47, 48)]:
                x_swapped[x == label1] = label2
                x_swapped[x == label2] = label1

            dd["input"] = x_swapped

            # Flip the landmark coordinates along the same axis
            dd["target"][:, 0] = dd["input"].shape[1] - dd["target"][:, 0]

            new_positions = []
            for start in range(0, n_total, block):
                block_slice = slice(start, start + block)
                # Reorder the landmarks within this vertebra's block
                indices_map = {k.item(): v + start for v, k in enumerate(target_indices[block_slice])}
                new_positions += [indices_map[self.flip_pairs[k.item()]] for k in target_indices[block_slice]]

            dd["target"] = dd["target"][new_positions]

        return dd


def _sample_generator(seed, dd, index_key):
    """Return a per-sample RNG, or ``None`` to draw from the global one.

    Seeding from the sample's own index is what makes an evaluation-time augmentation
    repeatable: ``worker_init_fn`` reseeds workers from ``torch.initial_seed()``, which
    Lightning advances every epoch, so the global RNG cannot give a sample the same draw
    twice.
    """
    if seed is None:
        return None
    index = dd.get(index_key, 0)
    if torch.is_tensor(index):
        index = int(index.item())
    generator = torch.Generator()
    generator.manual_seed((int(seed) * 1_000_003 + int(index)) % (2**63 - 1))
    return generator


def _rand(generator=None) -> float:
    """Draw one uniform in ``[0, 1)`` from ``generator``, or from the global RNG."""
    return float(torch.rand(1, generator=generator).item())


def _uniform(low: float, high: float, generator=None) -> float:
    """Draw one uniform in ``[low, high)``."""
    return low + (high - low) * _rand(generator)


def _resolve_channel(dd, key, default):
    """Read a channel index the sample may carry, falling back to a configured default.

    The stack order depends on which channels were requested, so the dataset is the only
    thing that reliably knows where the label map sits; a hardcoded index is silently
    wrong for some ``input_data_type`` values.
    """
    channel = dd.get(key, default)
    if torch.is_tensor(channel):
        channel = int(channel.item())
    return int(channel)


def _shift_volume(volume, delta):
    """Translate ``volume`` by ``delta`` voxels along its spatial axes, padding with zeros.

    Deliberately not ``torch.roll``: rolling is circular, so anatomy cut off one face would
    reappear on the opposite one and the landmark coordinate would land back *inside* the
    volume on the wrong structure - a corruption no bounds check can catch.

    Returns:
        ``(shifted, True)``, or ``(volume, False)`` if a shift would empty the volume.
    """
    shifted = torch.zeros_like(volume)
    src: list = [slice(None)]
    dst: list = [slice(None)]
    for axis, size in enumerate(volume.shape[1:]):
        step = int(delta[axis])
        if abs(step) >= size:
            return volume, False
        src.append(slice(max(0, -step), size - max(0, step)))
        dst.append(slice(max(0, step), size - max(0, -step)))
    shifted[tuple(dst)] = volume[tuple(src)]
    return shifted, True


def _intersect_loss_mask(dd, keep):
    """Fold ``keep`` into ``dd['loss_mask']``, creating the key if the sample has none."""
    carried = dd.get("loss_mask")
    dd["loss_mask"] = keep if carried is None else (carried.bool() & keep)


class LandmarksRandLimitedFov:
    """Truncate the volume the way a limited scanner field of view does, then re-centre.

    A cutout is centred offline on the bounding-box centre of its segmentation, and
    inference runs the same code - so on a scan whose field of view stops mid-thorax the
    box is centred on whatever anatomy is *visible*, and the content sits quite differently
    from any untruncated training sample. This reproduces that: zero one or more
    axis-aligned slabs, recompute the bounding-box centre of what survives, and shift the
    fixed-size box back onto it. Landmarks in the removed slabs are unsupervised.

    The spatial shape never changes, so the model's input size is unaffected.

    Args:
        prob: Per spatial axis, the probability that axis is cut at all. Axis order is the
            volume's own, i.e. ``dd["input"]`` dim ``axis + 1``.
        cut_range: Per axis, the ``(low, high)`` fraction of the axis removed per cut side.
        both_sides_prob: Given an axis is cut, the probability both ends are cut rather
            than one.
        min_keep_fraction: Floor on the surviving extent of an axis, as a fraction.
        min_visible_landmarks: Resample the cut if fewer supervised landmarks than this
            would survive; after ``max_attempts`` the sample is left untouched.
        max_attempts: Bound on that resampling.
        recentre: Whether to shift the surviving content back to the volume centre.
        label_channel: Channel of ``dd["input"]`` holding the instance-label map, used for
            the bounding box. Overridden per sample by ``dd[label_channel_key]``.
        label_channel_key: Sample key carrying that channel index.
        seed: When set, the cut is a pure function of ``seed`` and ``dd[index_key]``, so a
            sample gets the same field of view in every epoch. Leave ``None`` for training.
        index_key: Sample key carrying the dataset index.

    Raises:
        ValueError: If the ranges are malformed, which would otherwise degrade silently.
    """

    def __init__(
        self,
        prob=(0.25, 0.10, 0.60),
        cut_range=((0.05, 0.30), (0.05, 0.20), (0.10, 0.45)),
        both_sides_prob: float = 0.35,
        min_keep_fraction: float = 0.25,
        min_visible_landmarks: int = 16,
        max_attempts: int = 8,
        recentre: bool = True,
        label_channel: int = -1,
        label_channel_key: str = "label_channel",
        seed: int | None = None,
        index_key: str = "sample_index",
    ):
        prob = tuple(float(p) for p in prob)
        cut_range = tuple((float(low), float(high)) for low, high in cut_range)
        if len(prob) != 3 or len(cut_range) != 3:
            raise ValueError(f"prob and cut_range need one entry per spatial axis; got {len(prob)} and {len(cut_range)}.")
        if any(not 0.0 <= p <= 1.0 for p in prob):
            raise ValueError(f"prob entries must lie in [0, 1]; got {prob}.")
        if any(not 0.0 <= low <= high < 0.5 for low, high in cut_range):
            raise ValueError(f"cut_range entries must satisfy 0 <= low <= high < 0.5; got {cut_range}.")
        if not 0.0 <= both_sides_prob <= 1.0:
            raise ValueError(f"both_sides_prob must lie in [0, 1]; got {both_sides_prob}.")
        if not 0.0 < min_keep_fraction <= 1.0:
            raise ValueError(f"min_keep_fraction must lie in (0, 1]; got {min_keep_fraction}.")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1; got {max_attempts}.")

        self.prob = prob
        self.cut_range = cut_range
        self.both_sides_prob = both_sides_prob
        self.min_keep_fraction = min_keep_fraction
        self.min_visible_landmarks = min_visible_landmarks
        self.max_attempts = max_attempts
        self.recentre = recentre
        self.label_channel = label_channel
        self.label_channel_key = label_channel_key
        self.seed = seed
        self.index_key = index_key

    def _clamp(self, frac_low: float, frac_high: float) -> tuple[float, float]:
        """Scale both cut depths down proportionally until enough of the axis survives."""
        total = frac_low + frac_high
        excess = total - (1.0 - self.min_keep_fraction)
        if total <= 0.0 or excess <= 0.0:
            return frac_low, frac_high
        return frac_low - excess * (frac_low / total), frac_high - excess * (frac_high / total)

    def _sample_box(self, shape, generator):
        """Sample the slab to keep, or ``None`` when no axis was cut."""
        low = [0, 0, 0]
        high = list(shape)
        cut_any = False
        for axis in range(3):
            if _rand(generator) >= self.prob[axis]:
                continue
            range_low, range_high = self.cut_range[axis]
            both = _rand(generator) < self.both_sides_prob
            cut_low = both or _rand(generator) < 0.5
            frac_low = _uniform(range_low, range_high, generator) if cut_low else 0.0
            frac_high = _uniform(range_low, range_high, generator) if (both or not cut_low) else 0.0
            frac_low, frac_high = self._clamp(frac_low, frac_high)
            size = shape[axis]
            new_low = round(frac_low * size)
            new_high = size - round(frac_high * size)
            if new_high - new_low < 1 or (new_low == 0 and new_high == size):
                continue
            low[axis], high[axis] = new_low, new_high
            cut_any = True
        return (low, high) if cut_any else None

    @staticmethod
    def _inside(target, low, high):
        """Boolean per landmark: does it lie in the half-open keep box."""
        low_t = torch.as_tensor(low, dtype=target.dtype, device=target.device)
        high_t = torch.as_tensor(high, dtype=target.dtype, device=target.device)
        return torch.all((target >= low_t) & (target < high_t), dim=1)

    def __call__(self, dd: dict) -> dict:
        """Cut, re-centre and re-mask ``dd``; return it untouched when nothing was cut."""
        volume = dd["input"]
        target = dd["target"]
        shape = tuple(volume.shape[1:])
        generator = _sample_generator(self.seed, dd, self.index_key)
        carried = dd.get("loss_mask")

        for _ in range(self.max_attempts):
            box = self._sample_box(shape, generator)
            if box is None:
                # No slab sampled. Returning early matters: re-centring here would undo the
                # affine's deliberate off-centre translation on every such sample.
                return dd
            low, high = box
            visible = self._inside(target, low, high)
            survivors = visible if carried is None else (visible & carried.bool())
            if int(survivors.sum()) >= self.min_visible_landmarks:
                break
        else:
            return dd

        cropped = torch.zeros_like(volume)
        keep = (slice(None), slice(low[0], high[0]), slice(low[1], high[1]), slice(low[2], high[2]))
        cropped[keep] = volume[keep]

        delta = torch.zeros(3, dtype=torch.long)
        if self.recentre:
            channel = _resolve_channel(dd, self.label_channel_key, self.label_channel)
            # round() because an affine resamples in float: a label of 43 comes back as
            # 43.000004 and a bare `> 0` on the raw values would still be right, but the
            # bounding box must not pick up interpolation dust from a zero background.
            indices = torch.nonzero(cropped[channel].round() > 0)
            if indices.numel() == 0:
                return dd  # the cut removed every label; a sample with no anatomy teaches nothing
            centre = (indices.amin(0) + indices.amax(0)) // 2
            delta = torch.as_tensor(shape) // 2 - centre

        shifted, shifted_ok = _shift_volume(cropped, delta)
        if not shifted_ok:
            return dd
        target = target + delta.to(target.dtype)

        limits = torch.as_tensor(shape, dtype=target.dtype, device=target.device) - 1
        visible = visible & torch.all((target >= 0) & (target <= limits), dim=1)

        dd["input"] = shifted
        dd["target"] = target
        _intersect_loss_mask(dd, visible)
        return dd


class LandmarksRandLabelDropout:
    """Delete whole instance labels from the input and unsupervise their landmarks.

    A segmentation that misses a structure entirely is a common upstream failure, and it
    clusters at the ends of an ordered label range - the first and last ribs, the topmost
    and bottommost vertebrae - rather than falling uniformly, which is what ``end_bias``
    encodes.

    Args:
        prob: Probability of dropping anything at all.
        n_labels: Inclusive ``(min, max)`` number of labels to drop. Never all of them.
        contiguous: Drop a run of neighbouring labels rather than a random subset.
        end_bias: Probability that a contiguous run starts at one end of the label range.
        label_channel: Channel holding the instance-label map; overridden per sample by
            ``dd[label_channel_key]``.
        label_channel_key: Sample key carrying that channel index.
        landmark_labels_key: Sample key mapping each landmark to the label it sits on. When
            absent, the volume is still edited but no landmark is unsupervised - so the
            dataset must either supply it or re-derive the mask itself.
        seed: When set, the draw is a pure function of ``seed`` and ``dd[index_key]``.
        index_key: Sample key carrying the dataset index.

    Raises:
        ValueError: If ``prob``, ``end_bias`` or ``n_labels`` are malformed.
    """

    def __init__(
        self,
        prob: float = 0.3,
        n_labels=(1, 3),
        contiguous: bool = True,
        end_bias: float = 0.7,
        label_channel: int = -1,
        label_channel_key: str = "label_channel",
        landmark_labels_key: str = "landmark_labels",
        seed: int | None = None,
        index_key: str = "sample_index",
    ):
        n_labels = (int(n_labels[0]), int(n_labels[1]))
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must lie in [0, 1]; got {prob}.")
        if not 0.0 <= end_bias <= 1.0:
            raise ValueError(f"end_bias must lie in [0, 1]; got {end_bias}.")
        if not 1 <= n_labels[0] <= n_labels[1]:
            raise ValueError(f"n_labels must satisfy 1 <= min <= max; got {n_labels}.")

        self.prob = prob
        self.n_labels = n_labels
        self.contiguous = contiguous
        self.end_bias = end_bias
        self.label_channel = label_channel
        self.label_channel_key = label_channel_key
        self.landmark_labels_key = landmark_labels_key
        self.seed = seed
        self.index_key = index_key

    def _choose(self, present, generator):
        """Pick which of the ``present`` labels to delete, as a list of label values."""
        n_present = int(present.numel())
        # Never delete every label: an input with no anatomy has nothing to learn from.
        high = min(self.n_labels[1], n_present - 1)
        if high < 1:
            return []
        low = min(self.n_labels[0], high)
        count = max(low, min(high, int(_uniform(low, high + 1, generator))))

        if not self.contiguous:
            order = torch.randperm(n_present, generator=generator)
            return present[order[:count]].tolist()

        if _rand(generator) < self.end_bias:
            start = 0 if _rand(generator) < 0.5 else n_present - count
        else:
            start = int(_uniform(0, n_present - count + 1, generator))
        start = max(0, min(start, n_present - count))
        return present[start : start + count].tolist()

    def __call__(self, dd: dict) -> dict:
        """Delete a run of labels from every channel of ``dd['input']``, with probability ``prob``."""
        generator = _sample_generator(self.seed, dd, self.index_key)
        if _rand(generator) >= self.prob:
            return dd

        volume = dd["input"]
        channel = _resolve_channel(dd, self.label_channel_key, self.label_channel)
        labels = volume[channel].round()
        present = torch.unique(labels)
        present = present[present > 0].sort().values
        if present.numel() == 0:
            return dd

        chosen = self._choose(present, generator)
        if not chosen:
            return dd

        chosen_t = torch.as_tensor(chosen, dtype=labels.dtype, device=labels.device)
        drop = torch.isin(labels, chosen_t)
        if not bool(drop.any()):
            return dd
        # Every channel, not just the label map: a field of view that lost a rib lost its
        # surface and its intensities too.
        dd["input"] = volume.masked_fill(drop.unsqueeze(0), 0.0)

        landmark_labels = dd.get(self.landmark_labels_key)
        if landmark_labels is not None:
            gone = torch.isin(landmark_labels, chosen_t.to(landmark_labels.dtype))
            _intersect_loss_mask(dd, ~gone)
        return dd


class RandLabelCollapse:
    """Replace instance labels with one value, so identity cannot be read off the input.

    A label map fed to the network as raw values hands it the instance identity directly.
    Collapsing it to a binary mask forces the model to localise from geometry instead,
    which is what it has to do whenever the upstream segmentation mislabels.

    This edits ``dd["input"]`` only and belongs in an image-only pipeline, run *after* any
    bookkeeping copy of the label map has been split off - collapsing the map a loss mask
    is derived from would unsupervise almost everything.

    Args:
        prob: Probability of collapsing a given sample.
        value: The value every non-zero label becomes.
        channels: Channels of ``dd["input"]`` to collapse. When ``None`` the sample must
            carry ``dd[channels_key]``.
        channels_key: Sample key listing those channels.
        seed: When set, the draw is a pure function of ``seed`` and ``dd[index_key]``.
        index_key: Sample key carrying the dataset index.

    Raises:
        ValueError: If ``prob`` is not a probability.
    """

    def __init__(
        self,
        prob: float = 0.3,
        value: float = 1.0,
        channels=None,
        channels_key: str = "input_label_channels",
        seed: int | None = None,
        index_key: str = "sample_index",
    ):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must lie in [0, 1]; got {prob}.")
        self.prob = prob
        self.value = float(value)
        self.channels = None if channels is None else [int(c) for c in channels]
        self.channels_key = channels_key
        self.seed = seed
        self.index_key = index_key

    def __call__(self, dd: dict) -> dict:
        """Binarise the label channels of ``dd['input']`` with probability ``prob``.

        Raises:
            KeyError: If no channels were configured and the sample names none, which would
                otherwise make the transform a silent no-op for a whole training run.
        """
        generator = _sample_generator(self.seed, dd, self.index_key)
        if _rand(generator) >= self.prob:
            return dd

        channels = self.channels if self.channels is not None else dd.get(self.channels_key)
        if channels is None:
            raise KeyError(
                f"RandLabelCollapse needs the label channels: pass channels=... or have the dataset set dd[{self.channels_key!r}]."
            )
        if torch.is_tensor(channels):
            channels = channels.tolist()

        volume = dd["input"]
        collapsed = volume.clone()
        for channel in channels:
            index = int(channel)
            collapsed[index] = (volume[index] > 0).to(volume.dtype) * self.value
        dd["input"] = collapsed
        return dd


class Compose:
    def __init__(self, transforms):
        # `None` entries are dropped rather than rejected: callers compose an
        # optional pipeline with an optional flip stage and pass whichever exist.
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, dd):
        for transform in self.transforms:
            dd = transform(dd)
        return dd


#: Config ``"type"`` string -> transform class. A project that needs its own variants
#: builds a new dict from this one and passes it to the factories below; see
#: ``rib_poi.data.transforms.RIB_TRANSFORMS``.
TRANSFORMS = {
    "LandmarksRandAffine": LandmarksRandAffine,
    "LandMarksRandHorizontalFlip": LandMarksRandHorizontalFlip,
    "LandMarksRandHorizontalFlipNeighbor": LandMarksRandHorizontalFlipNeighbor,
    "LandmarksRandLimitedFov": LandmarksRandLimitedFov,
    "LandmarksRandLabelDropout": LandmarksRandLabelDropout,
    "RandLabelCollapse": RandLabelCollapse,
}


def create_transform(config, registry=None):
    """Build one transform from a ``{"type", "params"}`` config.

    Args:
        config: The config mapping.
        registry: Name-to-class mapping to resolve against; :data:`TRANSFORMS` by default.

    Returns:
        The constructed transform.

    Raises:
        UnknownTypeError: If the config names a transform that is not registered. The
            previous hand-written factory raised a bare ``ValueError`` and could only
            reach two of the classes in this module.
    """
    return build(TRANSFORMS if registry is None else registry, "transform", config)


def create_transforms(config, registry=None):
    """Build a transform pipeline from ``None``, one config, or a list of configs.

    Args:
        config: ``None``, a single ``{"type", "params"}`` mapping, or a list of them.
        registry: Name-to-class mapping to resolve against; :data:`TRANSFORMS` by default.

    Returns:
        A list of transforms, or ``None`` when there is nothing to build - so the result
        can be passed straight to a dataset's ``transforms=`` argument either way.
    """
    if config is None:
        return None
    configs = [config] if isinstance(config, dict) else list(config)
    return [create_transform(entry, registry) for entry in configs] or None


"""
Adapted from MONAI
"""

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

RandRange = Optional[Union[Sequence[Union[tuple[float, float], float]], float]]


# ---------------------------------------------------------------------------
# Vendored from MONAI (Apache-2.0), adapted to expose the sampled affine.
# https://github.com/Project-MONAI/MONAI  -  keep close to upstream.
# ---------------------------------------------------------------------------


class Resample(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        norm_coords: bool = True,
        device: torch.device | None = None,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ) -> None:
        """Computes output image using values from `img`, locations from `grid` using
        pytorch. supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            norm_coords: whether to normalize the coordinates from `[-(size-1)/2, (size-1)/2]` to
                `[0, size - 1]` (for ``monai/csrc`` implementation) or
                `[-1, 1]` (for torch ``grid_sample`` implementation) to be compatible with the underlying
                resampling API.
            device: device on which the tensor will be allocated.
            align_corners: Defaults to False.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm_coords = norm_coords
        self.device = device
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        grid: torch.Tensor | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        dtype: DtypeLike = None,
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        """Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
                if ``norm_coords`` is True, the grid values must be in `[-(size-1)/2, (size-1)/2]`.
                if ``USE_COMPILED=True`` and ``norm_coords=False``, grid values must be in `[0, size-1]`.
                if ``USE_COMPILED=False`` and ``norm_coords=False``, grid values must be in `[-1, 1]`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                To be compatible with other modules, the output data type is always `float32`.
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

        See Also:
            :py:const:`monai.config.USE_COMPILED`
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if grid is None:
            return img

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        _dtype = dtype or self.dtype or img.dtype
        _align_corners = self.align_corners if align_corners is None else align_corners
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=_dtype, device=_device)
        sr = min(
            len(img_t.peek_pending_shape() if isinstance(img_t, MetaTensor) else img_t.shape[1:]),
            3,
        )
        backend, _interp_mode, _padding_mode, _ = resolves_modes(
            self.mode if mode is None else mode,
            self.padding_mode if padding_mode is None else padding_mode,
            backend=None,
            use_compiled=USE_COMPILED,
        )

        if USE_COMPILED or backend == TransformBackends.NUMPY:
            grid_t, *_ = convert_to_dst_type(grid[:sr], img_t, dtype=grid.dtype, wrap_sequence=True)
            if isinstance(grid, torch.Tensor) and grid_t.data_ptr() == grid.data_ptr():
                grid_t = grid_t.clone(memory_format=torch.contiguous_format)
            for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                _dim = max(2, dim)
                t = (_dim - 1) / 2.0
                if self.norm_coords:
                    grid_t[i] = ((_dim - 1) / _dim) * grid_t[i] + t if _align_corners else grid_t[i] + t
                elif _align_corners:
                    grid_t[i] = ((_dim - 1) / _dim) * (grid_t[i] + 0.5)
            if USE_COMPILED and backend == TransformBackends.TORCH:  # compiled is using torch backend param name
                grid_t = moveaxis(grid_t, 0, -1)  # type: ignore
                out = grid_pull(
                    img_t.unsqueeze(0),
                    grid_t.unsqueeze(0).to(img_t),
                    bound=_padding_mode,
                    extrapolate=True,
                    interpolation=_interp_mode,
                )[0]
            elif backend == TransformBackends.NUMPY:
                is_cuda = img_t.is_cuda
                img_np = (convert_to_cupy if is_cuda else convert_to_numpy)(img_t, wrap_sequence=True)
                grid_np, *_ = convert_to_dst_type(grid_t, img_np, dtype=grid_t.dtype, wrap_sequence=True)
                _map_coord = (cupy_ndi if is_cuda else np_ndi).map_coordinates
                out = (cupy if is_cuda else np).stack([_map_coord(c, grid_np, order=_interp_mode, mode=_padding_mode) for c in img_np])
                out = convert_to_dst_type(out, img_t)[0]
        else:
            grid_t = moveaxis(grid[list(range(sr - 1, -1, -1))], 0, -1)  # type: ignore
            grid_t = convert_to_dst_type(grid_t, img_t, wrap_sequence=True)[0].unsqueeze(0)
            if isinstance(grid, torch.Tensor) and grid_t.data_ptr() == grid.data_ptr():
                grid_t = grid_t.clone(memory_format=torch.contiguous_format)
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[sr + 1 : 0 : -1]):
                    grid_t[0, ..., i] *= 2.0 / max(2, dim)
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0),
                grid_t,
                mode=_interp_mode,
                padding_mode=_padding_mode,
                align_corners=None if _align_corners == TraceKeys.NONE else _align_corners,  # type: ignore
            )[0]
        out_val, *_ = convert_to_dst_type(out, dst=img, dtype=np.float32)
        return out_val


class AffineGrid(LazyTransform):
    """Affine transforms on the coordinates.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
            Defaults to no rotation.
        shear_params: shearing factors for affine matrix, take a 3D affine as example::

            [
                [1.0, params[0], params[1], 0.0],
                [params[2], 1.0, params[3], 0.0],
                [params[4], params[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

            a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
        translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
            pixel/voxel relative to the center of the input image. Defaults to no translation.
        scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
            a tuple of 3 floats for 3D. Defaults to `1.0`.
        dtype: data type for the grid computation. Defaults to ``float32``.
            If ``None``, use the data type of input data (if `grid` is provided).
        device: device on which the tensor will be allocated, if a new grid is generated.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        align_corners: bool = False,
        affine: NdarrayOrTensor | None = None,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        _dtype = get_equivalent_dtype(dtype, torch.Tensor)
        self.dtype = _dtype if _dtype in (torch.float16, torch.float64, None) else torch.float32
        self.align_corners = align_corners
        self.affine = affine

    def __call__(
        self,
        spatial_size: Sequence[int] | None = None,
        grid: torch.Tensor | None = None,
        lazy: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """The grid can be initialized with a `spatial_size` parameter, or provided
        directly as `grid`. Therefore, either `spatial_size` or `grid` must be provided.
        When initialising from `spatial_size`, the backend "torch" will be used.

        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.
        """
        lazy_ = self.lazy if lazy is None else lazy
        if not lazy_:
            if grid is None:  # create grid from spatial_size
                if spatial_size is None:
                    raise ValueError("Incompatible values: grid=None and spatial_size=None.")
                grid_ = create_grid(spatial_size, device=self.device, backend="torch", dtype=self.dtype)
            else:
                grid_ = grid
            _dtype = self.dtype or grid_.dtype
            grid_: torch.Tensor = convert_to_tensor(grid_, dtype=_dtype, track_meta=get_track_meta())  # type: ignore
            _device = grid_.device  # type: ignore
            spatial_dims = len(grid_.shape) - 1
        else:
            _device = self.device
            spatial_dims = len(spatial_size)  # type: ignore
        _b = TransformBackends.TORCH
        affine: torch.Tensor
        if self.affine is None:
            affine = torch.eye(spatial_dims + 1, device=_device)
            if self.rotate_params:
                affine @= create_rotate(spatial_dims, self.rotate_params, device=_device, backend=_b)
            if self.shear_params:
                affine @= create_shear(spatial_dims, self.shear_params, device=_device, backend=_b)
            if self.translate_params:
                affine @= create_translate(spatial_dims, self.translate_params, device=_device, backend=_b)
            if self.scale_params:
                affine @= create_scale(spatial_dims, self.scale_params, device=_device, backend=_b)
        else:
            affine = self.affine  # type: ignore
        affine = to_affine_nd(spatial_dims, affine)
        if lazy_:
            return None, affine

        affine = convert_to_tensor(affine, device=grid_.device, dtype=grid_.dtype, track_meta=False)  # type: ignore
        if self.align_corners:
            sc = create_scale(
                spatial_dims,
                [max(d, 2) / (max(d, 2) - 1) for d in grid_.shape[1:]],
                device=_device,
                backend=_b,
            )
            sc = convert_to_dst_type(sc, affine)[0]
            grid_ = ((affine @ sc) @ grid_.view((grid_.shape[0], -1))).view([-1] + list(grid_.shape[1:]))
        else:
            grid_ = (affine @ grid_.view((grid_.shape[0], -1))).view([-1] + list(grid_.shape[1:]))
        return grid_, affine


class RandAffineGrid(Randomizable, LazyTransform):
    """Generate randomised affine grid.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling
    topic<lazy_resampling>` for more information.
    """

    backend = AffineGrid.backend

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        lazy: bool = False,
    ) -> None:
        """Args:
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxels to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            device: device to store the output grid data.
            dtype: data type for the grid computation. Defaults to ``np.float32``.
                If ``None``, use the data type of input data (if `grid` is provided).
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See Also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`

        """
        LazyTransform.__init__(self, lazy=lazy)
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: list[float] | None = None
        self.shear_params: list[float] | None = None
        self.translate_params: list[float] | None = None
        self.scale_params: list[float] | None = None

        self.device = device
        self.dtype = dtype
        self.affine: torch.Tensor | None = torch.eye(4, dtype=torch.float64)

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError(f"If giving range as [min,max], should have 2 elements per dim, got {f}.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Any | None = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self,
        spatial_size: Sequence[int] | None = None,
        grid: NdarrayOrTensor | None = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            randomize: boolean as to whether the grid parameters governing the grid should be randomized.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        if randomize:
            self.randomize()
        lazy_ = self.lazy if lazy is None else lazy
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
            dtype=self.dtype,
            lazy=lazy_,
        )
        if lazy_:  # return the affine only, don't construct the grid
            self.affine = affine_grid(spatial_size, grid)[1]  # type: ignore
            return None  # type: ignore
        _grid: torch.Tensor
        _grid, self.affine = affine_grid(spatial_size, grid)  # type: ignore
        return _grid

    def get_transformation_matrix(self) -> torch.Tensor | None:
        """Get the most recently applied transformation matrix."""
        return self.affine


class Affine(InvertibleTransform, LazyTransform):
    """Transform ``img`` given the affine parameters.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = list(set(AffineGrid.backend) & set(Resample.backend))

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        affine: NdarrayOrTensor | None = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        normalized: bool = False,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        align_corners: bool = False,
        image_only: bool = False,
        lazy: bool = False,
    ) -> None:
        """The affine transformations are applied in rotate, shear, translate, scale
        order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.
            affine: If applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            normalized: indicating whether the provided `affine` is defined to include a normalization
                transform converting the coordinates from `[-(size-1)/2, (size-1)/2]` (defined in ``create_grid``) to
                `[0, size - 1]` or `[-1, 1]` in order to be compatible with the underlying resampling API.
                If `normalized=False`, additional coordinate normalization will be applied before resampling.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            align_corners: Defaults to False.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            image_only: if True return only the image volume, otherwise return (image, affine).
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            dtype=dtype,
            align_corners=align_corners,
            device=device,
            lazy=lazy,
        )
        self.image_only = image_only
        self.norm_coord = not normalized
        self.resampler = Resample(
            norm_coords=self.norm_coord,
            device=device,
            dtype=dtype,
            align_corners=align_corners,
        )
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self.affine_grid.lazy = val
        self._lazy = val

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        lazy: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, NdarrayOrTensor]:
        """Args:
        img: shape must be (num_channels, H, W[, D]),
        spatial_size: output image spatial size.
            if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
            the transform will use the spatial size of `img`.
            if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
            if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``self.padding_mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        lazy: a flag to indicate whether this transform should execute lazily or not
            during this call. Setting this to False or True overrides the ``lazy`` flag set
            during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img_size)
        lazy_ = self.lazy if lazy is None else lazy
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        grid, affine = self.affine_grid(spatial_size=sp_size, lazy=lazy_)

        return affine_func(  # type: ignore
            img,
            affine,
            grid,
            self.resampler,
            sp_size,
            _mode,
            _padding_mode,
            True,
            self.image_only,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )

    @classmethod
    def compute_w_affine(cls, spatial_rank, mat, img_size, sp_size):
        r = int(spatial_rank)
        mat = to_affine_nd(r, mat)
        shift_1 = create_translate(r, [float(d - 1) / 2 for d in img_size[:r]])
        shift_2 = create_translate(r, [-float(d - 1) / 2 for d in sp_size[:r]])
        mat = shift_1 @ convert_data_type(mat, np.ndarray)[0] @ shift_2
        return mat

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]

        affine_grid = AffineGrid(affine=inv_affine, align_corners=align_corners)
        grid, _ = affine_grid(orig_size)
        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode, align_corners=align_corners)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        affine = convert_data_type(out.peek_pending_affine(), torch.Tensor)[0]
        xform, *_ = convert_to_dst_type(
            Affine.compute_w_affine(len(affine) - 1, inv_affine, data.shape[1:], orig_size),
            affine,
        )
        out.affine @= xform
        return out


class RandAffine(RandomizableTransform, InvertibleTransform, LazyTransform):
    """Random affine transform.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Affine.backend

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        lazy: bool = False,
    ) -> None:
        """Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``bilinear``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``reflection``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See Also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
            lazy=lazy,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache(lazy)
        self.mode = mode
        self.padding_mode: str = padding_mode

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.rand_affine_grid.lazy = val

    def _init_identity_cache(self, lazy: bool):
        """Create cache of the identity grid if cache_grid=True and spatial_size is
        known.
        """
        if lazy:
            return None
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn("cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'.")
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return create_grid(spatial_size=_sp_size, device=self.rand_affine_grid.device, backend="torch")

    def get_identity_grid(self, spatial_size: Sequence[int], lazy: bool):
        """Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        if lazy:
            return None
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(spatial_size, [2] * ndim):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return (
            create_grid(
                spatial_size=spatial_size,
                device=self.rand_affine_grid.device,
                backend="torch",
            )
            if self._cached_grid is None
            else self._cached_grid
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffine:
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
        grid=None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """Args:
        img: shape must be (num_channels, H, W[, D]),
        spatial_size: output image spatial size.
            if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
            the transform will use the spatial size of `img`.
            if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
            if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``self.padding_mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        randomize: whether to execute `randomize()` function first, default to True.
        grid: precomputed grid to be used (mainly to accelerate `RandAffined`).
        lazy: a flag to indicate whether this transform should execute lazily or not
            during this call. Setting this to False or True overrides the ``lazy`` flag set
            during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize()
        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and device
        ori_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, ori_size)
        do_resampling = self._do_transform or (sp_size != ensure_tuple(ori_size))
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        lazy_ = self.lazy if lazy is None else lazy
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if lazy_:
            if self._do_transform:
                if grid is None:
                    self.rand_affine_grid(sp_size, randomize=randomize, lazy=True)
                affine = self.rand_affine_grid.get_transformation_matrix()
            else:
                affine = convert_to_dst_type(torch.eye(len(sp_size) + 1), img, dtype=self.rand_affine_grid.dtype)[0]
        else:
            if grid is None:
                grid = self.get_identity_grid(sp_size, lazy_)
                if self._do_transform:
                    grid = self.rand_affine_grid(grid=grid, randomize=randomize, lazy=lazy_)
            affine = self.rand_affine_grid.get_transformation_matrix()
        return affine_func(  # type: ignore
            img,
            affine,
            grid,
            self.resampler,
            sp_size,
            _mode,
            _padding_mode,
            do_resampling,
            False,  # Return the affine matrix for usage with landmarks
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # if transform was not performed nothing to do.
        if not transform[TraceKeys.EXTRA_INFO]["do_resampling"]:
            return data
        orig_size = transform[TraceKeys.ORIG_SIZE]
        orig_size = fall_back_tuple(orig_size, data.shape[1:])
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]
        affine_grid = AffineGrid(affine=inv_affine)
        grid, _ = affine_grid(orig_size)

        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        affine = convert_data_type(out.peek_pending_affine(), torch.Tensor)[0]
        xform, *_ = convert_to_dst_type(
            Affine.compute_w_affine(len(affine) - 1, inv_affine, data.shape[1:], orig_size),
            affine,
        )
        out.affine @= xform
        return out
