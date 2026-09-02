"""Evaluation metrics: the aggregators behind the reported numbers.

Everything here is pure DataFrame in, DataFrame out, so no dataset, checkpoint or
GPU is needed. Errors are in millimetres throughout - the unit these functions are
documented in, and the one a voxel/mm mix-up would silently break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from TPTBox.core.poi import POI

from verpex.evaluation.metrics import (
    _sibling_path,
    _write_prediction_files,
    calculate_metrics,
    combine_centroids,
    compute_overall_metrics,
    compute_poi_wise_metrics,
    compute_poi_wise_metrics_proj,
    compute_sub_wise_metrics,
    compute_vert_wise_metrics,
    filter_high_refined_error_pois,
    filter_high_refined_proj_error_pois,
    load_and_filter_csv,
    np_to_ctd,
)

METRIC_COLUMNS = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]


@pytest.fixture
def prediction_df():
    """Two subjects x two vertebrae x two landmarks, with errors chosen by hand.

    The refined_error column is ``[1, 2, 3, 4, 5, 6, 7, 8]``, so every aggregate
    below has an exact expected value rather than a tolerance.
    """
    return pd.DataFrame(
        {
            "subject": ["sub-01"] * 4 + ["sub-02"] * 4,
            "vertebra": [20, 20, 21, 21, 20, 20, 21, 21],
            "poi_idx": [81, 82, 81, 82, 81, 82, 81, 82],
            "refined_error": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "coarse_error": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "refined_proj_error": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "coarse_proj_error": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


# --- calculate_metrics -----------------------------------------------------


def test_calculate_metrics_returns_mean_median_mse_accuracy_max():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    mean, median, mse, accuracy, max_error = calculate_metrics(errors, threshold=2.0)

    assert mean == pytest.approx(2.5)
    assert median == pytest.approx(2.5)
    assert mse == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert max_error == pytest.approx(4.0)
    # accuracy is the fraction strictly below the threshold: only 1.0 qualifies.
    assert accuracy == pytest.approx(0.25)


def test_accuracy_threshold_is_strict():
    """A landmark exactly at the threshold does not count as accurate."""
    _, _, _, accuracy, _ = calculate_metrics(np.array([2.0, 2.0]), threshold=2.0)
    assert accuracy == pytest.approx(0.0)


# --- grouped aggregation ---------------------------------------------------


def test_overall_metrics_cover_every_error_type(prediction_df):
    metrics = compute_overall_metrics(prediction_df)

    assert list(metrics.index) == ["coarse_error", "refined_error", "coarse_proj_error", "refined_proj_error"]
    assert metrics.loc["refined_error", "Mean Error"] == pytest.approx(4.5)
    assert metrics.loc["refined_error", "Max Error"] == pytest.approx(8.0)
    # The coarse column is twice the refined one, so its mean must be too.
    assert metrics.loc["coarse_error", "Mean Error"] == pytest.approx(9.0)


def test_poi_wise_metrics_group_by_landmark(prediction_df):
    metrics = compute_poi_wise_metrics(prediction_df)

    assert list(metrics.index) == [81, 82]
    assert list(metrics.columns) == METRIC_COLUMNS
    # Landmark 81 sees errors 1, 3, 5, 7; landmark 82 sees 2, 4, 6, 8.
    assert metrics.loc[81, "Mean Error"] == pytest.approx(4.0)
    assert metrics.loc[82, "Mean Error"] == pytest.approx(5.0)


def test_vert_wise_metrics_group_by_vertebra(prediction_df):
    metrics = compute_vert_wise_metrics(prediction_df)

    assert list(metrics.index) == [20, 21]
    # Vertebra 20 sees 1, 2, 5, 6; vertebra 21 sees 3, 4, 7, 8.
    assert metrics.loc[20, "Mean Error"] == pytest.approx(3.5)
    assert metrics.loc[21, "Mean Error"] == pytest.approx(5.5)


def test_sub_wise_metrics_group_by_subject(prediction_df):
    metrics = compute_sub_wise_metrics(prediction_df)

    assert list(metrics.index) == ["sub-01", "sub-02"]
    assert metrics.loc["sub-01", "Mean Error"] == pytest.approx(2.5)
    assert metrics.loc["sub-02", "Mean Error"] == pytest.approx(6.5)


def test_proj_variants_read_the_projected_column(prediction_df):
    """The _proj aggregators must not silently fall back to the unprojected error."""
    plain = compute_poi_wise_metrics(prediction_df)
    projected = compute_poi_wise_metrics_proj(prediction_df)

    # refined_proj_error is exactly half of refined_error in the fixture.
    assert projected.loc[81, "Mean Error"] == pytest.approx(plain.loc[81, "Mean Error"] / 2)
    assert projected.loc[82, "Mean Error"] == pytest.approx(plain.loc[82, "Mean Error"] / 2)


# --- outlier filtering -----------------------------------------------------


def test_high_error_filter_is_strictly_greater_than_the_threshold(prediction_df):
    outliers = filter_high_refined_error_pois(prediction_df, threshold=6.0)

    assert list(outliers["refined_error"]) == [7.0, 8.0]
    assert list(outliers.columns) == ["subject", "vertebra", "poi_idx", "refined_error"]
    # The index is reset, so callers can position-index the result.
    assert list(outliers.index) == [0, 1]


def test_high_error_filter_can_return_nothing(prediction_df):
    assert filter_high_refined_error_pois(prediction_df, threshold=100.0).empty


def test_proj_filter_reads_the_projected_column(prediction_df):
    outliers = filter_high_refined_proj_error_pois(prediction_df, threshold=3.0)
    assert list(outliers["refined_proj_error"]) == [3.5, 4.0]


# --- load_and_filter_csv ---------------------------------------------------


def test_vertebra_body_landmarks_are_dropped():
    df = pd.DataFrame({"poi_idx": [40, 41, 45, 50, 51, 81], "refined_error": [1.0] * 6})

    filtered = load_and_filter_csv(df)

    # 41-50 go; everything on either side stays.
    assert sorted(filtered["poi_idx"]) == [40, 51, 81]


def test_filtering_keeps_a_frame_with_no_body_landmarks_intact():
    df = pd.DataFrame({"poi_idx": [81, 82, 83], "refined_error": [1.0, 2.0, 3.0]})
    assert len(load_and_filter_csv(df)) == 3


# --- combine_centroids -----------------------------------------------------


SHAPE = (128, 128, 96)
ZOOM = (1.0, 1.0, 1.0)
ORIENTATION = ("L", "A", "S")


def _entry(subject="sub-01", centroids=None):
    """One per-vertebra inference result, as `run_predictions` emits them.

    ``centroids`` is a POI, not a plain dict: `combine_centroids` iterates it as
    ``(vertebra, landmark, coords)`` triples.
    """
    if centroids is None:
        centroids = {(20, 81): (1.0, 2.0, 3.0)}
    return {
        "subject": subject,
        "original_shape": SHAPE,
        "original_zoom": ZOOM,
        "original_orientation": ORIENTATION,
        "original_rotation": np.eye(3),
        "original_origin": (0.0, 0.0, 0.0),
        "centroids": POI(
            centroids=centroids,
            orientation=ORIENTATION,
            zoom=ZOOM,
            shape=SHAPE,
            origin=(0.0, 0.0, 0.0),
            rotation=np.eye(3),
        ),
    }


def test_combine_centroids_merges_entries_from_one_subject():
    subject, poi = combine_centroids(
        [
            _entry(centroids={(20, 81): (1.0, 2.0, 3.0)}),
            _entry(centroids={(21, 81): (4.0, 5.0, 6.0)}),
        ]
    )

    assert subject == "sub-01"
    assert (20, 81) in poi
    assert (21, 81) in poi


def test_combine_centroids_rejects_an_empty_list():
    """Reachable when safe_collate skipped every batch for a subject."""
    with pytest.raises(ValueError, match="at least one entry"):
        combine_centroids([])


def test_combine_centroids_rejects_mismatched_subjects():
    with pytest.raises(AssertionError, match="Subjects do not match"):
        combine_centroids([_entry(subject="sub-01"), _entry(subject="sub-02")])


# --- np_to_ctd -------------------------------------------------------------


def test_np_to_ctd_round_trips_coordinates():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    poi = np_to_ctd(coords, vertebra=20, origin=(0, 0, 0), rotation=np.eye(3), orientation=("L", "A", "S"))

    assert poi[20, 0] == pytest.approx((1.0, 2.0, 3.0))
    assert poi[20, 1] == pytest.approx((4.0, 5.0, 6.0))


def test_np_to_ctd_applies_the_padding_offset():
    coords = np.array([[10.0, 10.0, 10.0]])

    poi = np_to_ctd(coords, vertebra=20, origin=(0, 0, 0), rotation=np.eye(3), orientation=("L", "A", "S"), offset=(1.0, 2.0, 3.0))

    assert poi[20, 0] == pytest.approx((9.0, 8.0, 7.0))


def test_np_to_ctd_labels_landmarks_from_idx_list():
    coords = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    poi = np_to_ctd(coords, vertebra=20, origin=(0, 0, 0), rotation=np.eye(3), orientation=("L", "A", "S"), idx_list=[81, 82])

    assert (20, 81) in poi
    assert (20, 82) in poi


def test_np_to_ctd_requires_an_orientation():
    with pytest.raises(ValueError, match="orientation"):
        np_to_ctd(np.zeros((1, 3)), vertebra=20, origin=(0, 0, 0), rotation=np.eye(3))


# --- output paths ----------------------------------------------------------


def test_sibling_paths_do_not_collide_for_a_custom_file_ending():
    """Regression: the GT used to overwrite the prediction.

    The companion paths were derived by replacing ``"_pred.json"``, a no-op for any
    other ``poi_file_ending``, so ``gt_save_path`` came out equal to the prediction
    path and clobbered the file written moments earlier.
    """
    for ending in ("_pred.json", "_custom.json", ".json"):
        prediction = f"/out/sub-01_20{ending}"
        companions = [_sibling_path(prediction, s) for s in ("_gt.json", "_pred_global.json", "_gt_global.json", "_seg.nii.gz")]

        assert prediction not in companions, f"{ending}: a companion path collides with the prediction"
        assert len(set(companions)) == len(companions), f"{ending}: two companions share a path"


def test_sibling_path_keeps_the_historical_names_for_the_default_ending():
    assert _sibling_path("/out/sub-01_20_pred.json", "_gt.json") == "/out/sub-01_20_gt.json"
    assert _sibling_path("/out/sub-01_20_pred.json", "_seg.nii.gz") == "/out/sub-01_20_seg.nii.gz"


def test_write_prediction_files_reports_a_missing_prediction(tmp_path, monkeypatch):
    """A prediction that fails to land is reported, not silently treated as written."""

    class _Pois:
        def save(self, path, verbose=False):
            pass

    assert _write_prediction_files(_Pois(), str(tmp_path / "poi.json"), str(tmp_path / "missing_pred.json"), [20]) is False
