"""Data module construction: defaults that differ between the two dataset layouts."""

from __future__ import annotations

import pytest

from verpex.modules.data_modules import (
    DATA_MODULES,
    NEIGHBOR_DATASETS,
    SINGLE_VERTEBRA_DATASETS,
    SPINE_DATASET,
    SPINE_NEIGHBOR_DATASET,
    POIDataModule,
    SpineDataModule,
    SpineNeighborDataModule,
)

#: __init__ only stores configuration; nothing is read until setup(), so a
#: non-existent master_df is fine here.
BASE = {"master_df": "nonexistent.csv", "train_subjects": [], "val_subjects": [], "test_subjects": []}


def test_neighbour_module_defaults_to_no_flip():
    """It previously inherited 0.5 and flipped silently.

    The neighbour module printed a warning about flip_prob and then did nothing: the
    line that would have disabled flipping was commented out.
    """
    assert SpineNeighborDataModule(**BASE).flip_prob == 0.0


def test_single_vertebra_module_keeps_the_base_default():
    assert SpineDataModule(**BASE).flip_prob == 0.5


@pytest.mark.parametrize("flip_prob", [0.0, 0.25, 0.5])
def test_an_explicit_flip_prob_is_honoured(flip_prob):
    """A config's explicit setting must win over the module's default."""
    assert SpineNeighborDataModule(**BASE, flip_prob=flip_prob).flip_prob == flip_prob


@pytest.mark.parametrize("name", sorted(DATA_MODULES))
def test_every_registered_data_module_constructs(name):
    assert DATA_MODULES[name](**BASE) is not None


def test_the_neighbour_flag_is_derived_from_the_saved_dataset_value():
    """infer.py decides whether to use neighbours from the saved dataset value."""
    assert SPINE_NEIGHBOR_DATASET in NEIGHBOR_DATASETS
    assert SPINE_DATASET not in NEIGHBOR_DATASETS
    assert SPINE_DATASET in SINGLE_VERTEBRA_DATASETS


def test_build_cutouts_requires_a_poi_source():
    """`poi_source` replaced a hardcoded cohort name, so it has to be supplied.

    There is no dataset-independent default: the value is the BIDS `source-` entity
    and has to match the filenames on disk.
    """
    for cls in (SpineDataModule, SpineNeighborDataModule):
        with pytest.raises(ValueError, match="poi_source"):
            cls(**BASE).build_cutouts(bids_surgery_info=None, save_path="/tmp/unused")


def _capture_base_build_cutouts(monkeypatch):
    """Stub out the base `build_cutouts` and return what the subclass passes up.

    The real base reads BIDS data, so this intercepts the call one level below the
    subclass logic under test.
    """
    captured = {}

    def record(self, bids_surgery_info, save_path, get_files=None, rescale_zoom=None, poi_source=None):
        captured["get_files"] = get_files

    monkeypatch.setattr(POIDataModule, "build_cutouts", record)
    return captured


def test_poi_source_is_bound_into_the_getter_handed_upwards(monkeypatch):
    """Given a source, the subclass builds the POI getter and passes it to the base."""
    captured = _capture_base_build_cutouts(monkeypatch)

    SpineDataModule(**BASE).build_cutouts(None, "/tmp/unused", poi_source="annotator-a")

    # The given source reaches get_spine_poi, rather than a default chosen for us.
    assert captured["get_files"].keywords["get_poi"].keywords == {"source": "annotator-a"}


def test_an_explicit_get_files_wins_over_poi_source(monkeypatch):
    """Supplying a getter is the documented alternative to naming a source."""
    sentinel = object()
    captured = _capture_base_build_cutouts(monkeypatch)

    SpineDataModule(**BASE).build_cutouts(None, "/tmp/unused", get_files=sentinel)

    assert captured["get_files"] is sentinel
