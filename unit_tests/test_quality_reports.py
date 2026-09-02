"""Reading QA reports that exclude vertebrae with known annotation errors."""

from __future__ import annotations

import pandas as pd
import pytest
from TPTBox import Location, Vertebra_Instance

from verpex.data.quality_reports import (
    convert_agg_report_to_reported_bool_dict,
    is_poi_reported,
    is_vert_reported,
    load_agg_report_df,
    normalize_report_key,
)


@pytest.fixture
def report():
    return pd.DataFrame(
        [
            {"subject_name": "sub-a", "vertebra": 20, "location": 81, "severity": 1.0},
            {"subject_name": "sub-a", "vertebra": 20, "location": 82, "severity": 0.5},
            {"subject_name": "sub-b", "vertebra": 21, "location": 81, "severity": 2.0},
            # Cervical vertebra, below the default vertebra_from cut-off.
            {"subject_name": "sub-b", "vertebra": 3, "location": 81, "severity": 9.0},
        ]
    )


def test_enum_and_int_keys_agree():
    """Location is a plain Enum, so Location(81) != 81 and they hash differently."""
    assert normalize_report_key(20, 81) == normalize_report_key(Vertebra_Instance(20), Location(81))


def test_reported_vertebra_is_found(report):
    index = convert_agg_report_to_reported_bool_dict(report)
    assert is_vert_reported(index, "sub-a", 20)
    assert not is_vert_reported(index, "sub-a", 21)


def test_unknown_subject_is_not_reported(report):
    index = convert_agg_report_to_reported_bool_dict(report)
    assert not is_vert_reported(index, "sub-missing", 20)
    assert not is_poi_reported(index, "sub-missing", 20, 81)


def test_specific_poi_lookup(report):
    index = convert_agg_report_to_reported_bool_dict(report)
    assert is_poi_reported(index, "sub-a", 20, 81)
    assert not is_poi_reported(index, "sub-a", 20, 99)


def test_severity_threshold_filters_rows(report):
    index = convert_agg_report_to_reported_bool_dict(report, severity_threshold=0.9)
    assert is_poi_reported(index, "sub-a", 20, 81)
    assert not is_poi_reported(index, "sub-a", 20, 82)


def test_vertebrae_below_the_cutoff_are_ignored(report):
    index = convert_agg_report_to_reported_bool_dict(report)
    assert not is_vert_reported(index, "sub-b", 3)
    assert is_vert_reported(index, "sub-b", 21)


def test_missing_report_raises_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_agg_report_df(tmp_path / "nope.xlsx")


def test_report_missing_a_column_is_rejected(tmp_path):
    path = tmp_path / "report.xlsx"
    pd.DataFrame([{"subject_name": "sub-a", "vertebra": 20}]).to_excel(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_agg_report_df(path)
