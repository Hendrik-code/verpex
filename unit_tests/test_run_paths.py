"""Locating a training run's files from its checkpoint path.

`verpex-infer` used to take a name from an enum of hardcoded checkpoint locations.
It now takes checkpoint paths, and everything the enum supplied - the data module
configuration and a label for the output derivative - is derived from the path.
"""

from __future__ import annotations

import pytest

from verpex.cli.infer import resolve_run_paths


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A training run laid out the way Lightning writes one."""
    monkeypatch.setenv("VERPEX_MODEL_ROOT", str(tmp_path))
    version_dir = tmp_path / "surface_cc3" / "version_2"
    (version_dir / "checkpoints").mkdir(parents=True)
    (version_dir / "data_module_params.json").write_text("{}", encoding="utf-8")
    checkpoint = version_dir / "checkpoints" / "sad-pt-epoch=104.ckpt"
    checkpoint.write_text("", encoding="utf-8")
    return checkpoint


def test_an_absolute_checkpoint_path_is_used_as_given(run):
    model_path, dm_path, _ = resolve_run_paths(str(run))

    assert model_path == str(run)
    assert dm_path.endswith("data_module_params.json")


def test_a_relative_path_resolves_against_model_root(run):
    """Relative paths are the point: `model_root` is machine-specific configuration."""
    relative = "surface_cc3/version_2/checkpoints/sad-pt-epoch=104.ckpt"

    assert resolve_run_paths(relative)[0] == str(run)


def test_the_ckpt_suffix_is_optional(run):
    without_suffix = "surface_cc3/version_2/checkpoints/sad-pt-epoch=104"

    assert resolve_run_paths(without_suffix)[0] == str(run)


def test_the_label_identifies_the_run_for_the_output_derivative(run):
    """The label names the output folder, so it must distinguish runs and versions."""
    assert resolve_run_paths(str(run))[2] == "surface_cc3-version_2"


def test_a_missing_checkpoint_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("VERPEX_MODEL_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_run_paths("does/not/exist.ckpt")


def test_a_run_without_a_saved_data_module_config_is_reported(tmp_path, monkeypatch):
    """Without it there is no way to rebuild the dataset the model was trained on."""
    monkeypatch.setenv("VERPEX_MODEL_ROOT", str(tmp_path))
    checkpoint = tmp_path / "orphan" / "version_0" / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"data_module_params\.json"):
        resolve_run_paths(str(checkpoint))
