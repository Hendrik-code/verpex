"""Coordinate-space conversions between the cutout grid and patient space.

`revert_poi_to_original_space` is the step that has silently corrupted results
before: the cutout offset is counted in the *preprocessed* grid, so it has to be
added before the rescale. Getting the order wrong is invisible at 1 mm isotropic
spacing and is tens of millimetres off on anisotropic data - exactly the kind of
bug that survives a visual check on VerSe and ruins everything else.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from TPTBox import POI

from verpex.geometry.spaces import SpaceMeta, batch_to_device, extract_space_meta, revert_poi_to_original_space


def make_poi(centroids, zoom=(1.0, 1.0, 1.0), shape=(32, 32, 32)):
    """Build a real TPTBox POI on a 1 mm cutout grid.

    Using the real class rather than a stub keeps these tests honest about
    TPTBox's actual rescale semantics, which is where the bug lived.
    """
    return POI(centroids=dict(centroids), orientation=("L", "A", "S"), zoom=zoom, shape=shape)


ORIGINAL = SpaceMeta(
    zoom=(2.0, 2.0, 2.0),
    origin=(0.0, 0.0, 0.0),
    shape=(64, 64, 64),
    rotation=np.eye(3),
    orientation=("L", "A", "S"),
)


def test_offset_is_applied_before_rescale():
    """The offset counts preprocessed-grid voxels, so it must be added first.

    With an offset of 10 and a zoom of 2, adding first then rescaling gives
    (0 + 10) / 2 = 5. Rescaling first would give 0 / 2 + 10 = 10 - the classic
    off-by-(zoom_ratio - 1) * offset error.
    """
    poi = make_poi({(1, 81): (0.0, 0.0, 0.0)})
    revert_poi_to_original_space(poi, np.array([10.0, 10.0, 10.0]), ORIGINAL)
    assert poi.centroids[1, 81] == pytest.approx((5.0, 5.0, 5.0))


def test_isotropic_1mm_is_the_case_that_hides_the_bug():
    """At 1 mm the two orderings agree, which is why this went unnoticed on VerSe."""
    meta = ORIGINAL._replace(zoom=(1.0, 1.0, 1.0))
    poi = make_poi({(1, 81): (3.0, 4.0, 5.0)})
    revert_poi_to_original_space(poi, np.array([10.0, 10.0, 10.0]), meta)
    assert poi.centroids[1, 81] == pytest.approx((13.0, 14.0, 15.0))


def test_anisotropic_spacing_is_applied_per_axis():
    meta = ORIGINAL._replace(zoom=(0.8, 1.0, 2.5))
    poi = make_poi({(1, 81): (0.0, 0.0, 0.0)})
    revert_poi_to_original_space(poi, np.array([8.0, 8.0, 8.0]), meta)
    assert poi.centroids[1, 81] == pytest.approx((10.0, 8.0, 3.2))


def test_shape_and_orientation_are_restored():
    poi = make_poi({(1, 81): (1.0, 1.0, 1.0)})
    revert_poi_to_original_space(poi, np.array([0.0, 0.0, 0.0]), ORIGINAL)
    assert poi.shape == ORIGINAL.shape
    assert poi.orientation == ORIGINAL.orientation


def test_every_landmark_is_transformed():
    poi = make_poi({(1, 81): (0.0, 0.0, 0.0), (1, 82): (2.0, 2.0, 2.0), (2, 81): (4.0, 4.0, 4.0)})
    revert_poi_to_original_space(poi, np.array([2.0, 2.0, 2.0]), ORIGINAL)
    assert {(v, p) for v, p, _ in poi.centroids.items()} == {(1, 81), (1, 82), (2, 81)}
    assert poi.centroids[2, 81] == pytest.approx((3.0, 3.0, 3.0))


def test_extract_space_meta_reads_the_collated_batch_layout():
    """Spatial fields arrive as [dim0_batch, dim1_batch, dim2_batch]."""
    batch = {
        "original_zoom": [torch.tensor([0.8]), torch.tensor([1.0]), torch.tensor([2.5])],
        "original_origin": [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
        "original_shape": [torch.tensor([64]), torch.tensor([65]), torch.tensor([66])],
        "original_rotation": [torch.eye(3)],
        "original_orientation": ["('L', 'A', 'S')"],
    }
    meta = extract_space_meta(batch, "original")
    assert meta.zoom == pytest.approx((0.8, 1.0, 2.5))
    assert meta.shape == (64, 65, 66)
    assert meta.orientation == ("L", "A", "S")


def test_batch_to_device_leaves_non_tensors_alone():
    batch = {"a": torch.zeros(2), "b": "a string", "c": 5}
    moved = batch_to_device(batch, torch.device("cpu"))
    assert moved["b"] == "a string"
    assert moved["c"] == 5
    assert moved["a"].device.type == "cpu"
