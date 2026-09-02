"""Resolving cutout directories recorded in master_df.csv."""

from __future__ import annotations

import pytest

from verpex import paths
from verpex.data.dataset import resolve_cutout_dir


@pytest.fixture(autouse=True)
def _cutout_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VERPEX_CUTOUT_ROOT", str(tmp_path / "cutouts"))
    paths.reset_cache()
    yield tmp_path / "cutouts"
    paths.reset_cache()


def test_relative_entry_is_resolved_against_the_cutout_root(_cutout_root):
    """prepare_data writes relative paths so the CSV is portable across machines."""
    assert resolve_cutout_dir("sub-01/20") == str(_cutout_root / "sub-01" / "20")


def test_absolute_entry_is_left_alone():
    """Older master_df.csv files stored absolute paths; those must keep working."""
    assert resolve_cutout_dir("/data/cutouts/sub-01/20") == "/data/cutouts/sub-01/20"


def test_legacy_relative_entries_resolve_under_the_configured_root(_cutout_root):
    """A master_df.csv from before the relative-path change still works.

    Older files stored `file_dir` as `dataset/data_preprocessing/.../<sub>/<vert>`,
    which the dataset patched by substituting an absolute prefix. Joining such an
    entry onto `cutout_root` reproduces that exactly, provided `cutout_root` is set
    to the directory the old substitution produced.
    """
    legacy = "dataset/data_preprocessing/cutout-folder/cutouts/sub-01/20"
    assert resolve_cutout_dir(legacy) == str(_cutout_root / legacy)
