import ast
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

if TYPE_CHECKING:
    from TPTBox import POI


def batch_to_device(batch: dict, device) -> dict:
    """Move all tensors in a batch dict to the given device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class SpaceMeta(NamedTuple):
    """Spatial metadata for one coordinate space (preprocessed cutout or original patient)."""

    zoom: tuple
    origin: tuple
    shape: tuple
    rotation: np.ndarray
    orientation: tuple


def extract_space_meta(batch: dict, prefix: str) -> SpaceMeta:
    """Extract zoom/origin/shape/rotation/orientation from a collated inference batch.

    prefix is either 'preprocessed' or 'original'.
    Each spatial field is stored as a list-of-batch-tensors [dim0_batch, dim1_batch, dim2_batch],
    so we read index [dim][0].item() to get the scalar for the first (and only) batch item.
    """
    return SpaceMeta(
        zoom=(
            batch[f"{prefix}_zoom"][0][0].item(),
            batch[f"{prefix}_zoom"][1][0].item(),
            batch[f"{prefix}_zoom"][2][0].item(),
        ),
        origin=(
            batch[f"{prefix}_origin"][0][0].item(),
            batch[f"{prefix}_origin"][1][0].item(),
            batch[f"{prefix}_origin"][2][0].item(),
        ),
        shape=(
            batch[f"{prefix}_shape"][0][0].item(),
            batch[f"{prefix}_shape"][1][0].item(),
            batch[f"{prefix}_shape"][2][0].item(),
        ),
        rotation=batch[f"{prefix}_rotation"][0].detach().cpu().numpy(),
        orientation=ast.literal_eval(batch[f"{prefix}_orientation"][0]),
    )


def revert_poi_to_original_space(poi, cutout_offset, orig_meta: SpaceMeta) -> "POI":
    """Transform a POI from preprocessed cutout space back to original patient space.

    Steps: shift by cutout offset → rescale to original zoom → set shape → reorient.
    Modifies poi in-place and returns it.

    The offset must be applied before the rescale. preprocess_segmentation_masks derives
    it from a crop taken after the volume was rescaled to the model's zoom, so it counts
    voxels in that preprocessed grid, not in the original one. Rescaling first would add
    preprocessed-grid voxels to original-grid coordinates, displacing every POI by
    (zoom_ratio - 1) * offset - which is zero only when the two spacings agree, as they do
    for 1mm-iso data like verse, and is tens of millimetres along the spine at 0.8mm.
    """
    new_centroids = {}
    for v, p_idx, c in poi.centroids.items():
        new_coords = np.array(c) + cutout_offset
        new_centroids[(v, p_idx)] = (new_coords[0], new_coords[1], new_coords[2])
    poi.centroids = new_centroids
    poi.rescale_(orig_meta.zoom, verbose=False)
    poi.shape = orig_meta.shape
    poi.reorient_(orig_meta.orientation, verbose=False)
    return poi
