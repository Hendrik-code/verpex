"""The registries must resolve exactly the type strings existing configs use."""

from __future__ import annotations

import pytest

from verpex.modules.data_modules import DATA_MODULES
from verpex.modules.feature_extraction import FEATURE_EXTRACTION_MODULES
from verpex.modules.poi_module import PREDICTION_MODULES
from verpex.modules.refinement import REFINEMENT_MODULES
from verpex.registry import UnknownTypeError, build, resolve
from verpex.training_utils import CALLBACKS

#: Every type string that appears in a historical experiment config. These must keep
#: resolving, or previously working configs break.
HISTORICAL_TYPES = {
    "prediction module": (PREDICTION_MODULES, ["PoiPredictionModule", "PoiNeighborPredictionModule"]),
    "feature extraction module": (FEATURE_EXTRACTION_MODULES, ["SADenseNet", "HeatmapFeatureDenseNet", "SMDenseNet", "SMSADenseNet"]),
    "refinement module": (
        REFINEMENT_MODULES,
        [
            "Identity",
            "PatchTransformer",
            "NoPoiPatchTransformer",
            "NoVertPatchTransformer",
            "NoPoiVertPatchTransformer",
            "NoPoiFeaturePatchTransformer",
            "NoPoiVertFeaturePatchTransformer",
            "NoCoarsePredTransformer",
            "FeatureTransformer",
        ],
    ),
    "data module": (DATA_MODULES, ["SpineDataModule", "SpineNeighborDataModule"]),
    "callback": (CALLBACKS, ["ModelCheckpoint", "EarlyStopping"]),
}


@pytest.mark.parametrize(
    ("kind", "registry", "name"),
    [(kind, registry, name) for kind, (registry, names) in HISTORICAL_TYPES.items() for name in names],
)
def test_historical_config_types_resolve(kind, registry, name):
    assert resolve(registry, kind, name) is not None


def test_unknown_type_names_the_alternatives():
    with pytest.raises(UnknownTypeError) as excinfo:
        resolve(REFINEMENT_MODULES, "refinement module", "NoSuchTransformer")
    message = str(excinfo.value)
    assert "NoSuchTransformer" in message
    # The whole point of the registry is that the error lists what is valid.
    assert "PatchTransformer" in message


def test_build_rejects_config_without_type():
    with pytest.raises(KeyError, match="type"):
        build(REFINEMENT_MODULES, "refinement module", {"params": {}})


def test_build_passes_params_through():
    module = build(REFINEMENT_MODULES, "refinement module", {"type": "Identity", "params": {}})
    assert module is not None


@pytest.mark.parametrize("config", [{"type": "Identity"}, {"type": "Identity", "params": None}, {"type": "Identity", "params": {}}])
def test_params_may_be_omitted_or_null(config):
    """`"params": null` is easy to write by hand and must not fail obscurely."""
    assert build(REFINEMENT_MODULES, "refinement module", config) is not None


def test_non_mapping_params_are_rejected_clearly():
    with pytest.raises(TypeError, match="must be a mapping"):
        build(REFINEMENT_MODULES, "refinement module", {"type": "Identity", "params": [1, 2]})
