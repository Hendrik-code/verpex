"""The shipped example configs must stay valid as the code changes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verpex.modules.data_modules import DATA_MODULES
from verpex.modules.feature_extraction import FEATURE_EXTRACTION_MODULES
from verpex.modules.poi_module import PREDICTION_MODULES
from verpex.modules.refinement import REFINEMENT_MODULES
from verpex.registry import build, resolve
from verpex.training_utils import CALLBACKS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "verpex" / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.json"))

#: Guards against a leak reappearing in a committed config.
FORBIDDEN = ("/DATA/", "/home/", "/media/", "/mnt/", "WS-")


def test_at_least_one_example_config_is_shipped():
    assert CONFIGS, f"no example configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_has_the_expected_top_level_keys(path):
    config = json.loads(path.read_text())
    assert {"path", "name", "module_config", "data_module_config"} <= set(config)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_type_string_resolves(path):
    config = json.loads(path.read_text())
    params = config["module_config"].get("params", {})
    resolve(PREDICTION_MODULES, "prediction module", config["module_config"]["type"])
    resolve(FEATURE_EXTRACTION_MODULES, "coarse module", params["coarse_config"]["type"])
    resolve(REFINEMENT_MODULES, "refinement module", params["refinement_config"]["type"])
    resolve(DATA_MODULES, "data module", config["data_module_config"]["type"])
    for callback in config.get("callbacks_config", []):
        resolve(CALLBACKS, "callback", callback["type"])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_builds_a_model(path):
    """The strongest check: the config actually constructs its model."""
    config = json.loads(path.read_text())
    module = build(PREDICTION_MODULES, "prediction module", config["module_config"])
    assert sum(p.numel() for p in module.parameters()) > 0


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_contains_no_machine_paths_or_subject_ids(path):
    text = path.read_text()
    for needle in FORBIDDEN:
        assert needle not in text, f"{path.name} contains {needle!r}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_subject_splits_are_empty_placeholders(path):
    """Example configs must not ship a real cohort's subject list."""
    params = json.loads(path.read_text())["data_module_config"]["params"]
    for key in ("train_subjects", "val_subjects", "test_subjects"):
        assert params.get(key) == [], f"{path.name}: {key} should be an empty placeholder"
