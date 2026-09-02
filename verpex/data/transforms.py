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


class Compose:
    def __init__(self, transforms):
        # `None` entries are dropped rather than rejected: callers compose an
        # optional pipeline with an optional flip stage and pass whichever exist.
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, dd):
        for transform in self.transforms:
            dd = transform(dd)
        return dd


def create_transform(config):
    transform_type = config["type"]

    if transform_type == "LandmarksRandAffine":
        return LandmarksRandAffine(**config["params"])
    elif transform_type == "LandMarksRandHorizontalFlip":
        return LandMarksRandHorizontalFlip(**config["params"])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


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
