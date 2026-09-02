"""Reading QA reports that mark vertebrae as having annotation errors.

`prepare_data` can be pointed at an aggregated quality-assurance report (an Excel
file produced by the annotation-review tooling) so that vertebrae with known POI
annotation errors are left out of the generated training cutouts.

Only the read side of that workflow lives here; the tooling that *produces* the
reports is not part of this package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from TPTBox import Location, Vertebra_Instance

#: Column names expected in an aggregated report.
_REQUIRED_COLUMNS = ("subject_name", "vertebra", "location", "severity")

#: Vertebrae below this label are cervical and are not reviewed by the report workflow.
DEFAULT_VERTEBRA_FROM = 8

ReportDict = dict[str, dict[tuple[int, int], bool]]


def normalize_report_key(vert: int | Vertebra_Instance, location: int | Location) -> tuple[int, int]:
    """Return ``(vertebra, location)`` as plain ints.

    ``Location`` is a plain ``Enum``, not an ``IntEnum``, so ``Location(117) != 117``
    and the two hash differently. Every report lookup goes through here, so callers
    may pass either an enum member or an int.

    Args:
        vert: Vertebra label, as an int or a ``Vertebra_Instance``.
        location: POI location id, as an int or a ``Location``.

    Returns:
        The pair as plain ints, safe to use as a dict key.
    """
    v = int(vert.value) if isinstance(vert, Vertebra_Instance) else int(vert)
    s = int(location.value) if isinstance(location, Location) else int(location)
    return v, s


def load_agg_report_df(path: str | Path) -> pd.DataFrame:
    """Load an aggregated POI report from an Excel file.

    Args:
        path: Path to the ``.xlsx`` report.

    Returns:
        The report as a DataFrame.

    Raises:
        FileNotFoundError: If the report does not exist. The original implementation
            logged and returned ``None`` here, which turned a missing report into a
            confusing ``NoneType`` failure much later; failing at the read is clearer.
        ValueError: If the report is missing a required column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated POI report not found: {path}")
    df = pd.read_excel(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Report {path} is missing required column(s): {', '.join(missing)}")
    return df


def convert_agg_report_to_reported_bool_dict(
    df: pd.DataFrame,
    severity_threshold: float = 0.0,
    vertebra_from: int = DEFAULT_VERTEBRA_FROM,
) -> ReportDict:
    """Index a report by subject for fast lookup.

    Args:
        df: An aggregated report, as returned by :func:`load_agg_report_df`.
        severity_threshold: Only rows at or above this severity are counted as reported.
        vertebra_from: Vertebrae with a label below this are skipped.

    Returns:
        ``{subject: {(vertebra, location): True}}`` for every reported POI. Look
        entries up with :func:`is_poi_reported` or :func:`is_vert_reported` rather
        than indexing directly, so key normalisation is applied.
    """
    report_dict: ReportDict = {}
    for _, row in df.iterrows():
        if row["vertebra"] < vertebra_from:
            continue
        if row["severity"] < severity_threshold:
            continue
        subject = report_dict.setdefault(row["subject_name"], {})
        subject[normalize_report_key(row["vertebra"], row["location"])] = True
    return report_dict


def is_poi_reported(report_dict: ReportDict, subject_ct_id: str, vert, location) -> bool:
    """Return whether one specific POI is flagged for this subject."""
    subject = report_dict.get(subject_ct_id)
    if subject is None:
        return False
    return subject.get(normalize_report_key(vert, location), False)


def is_vert_reported(report_dict: ReportDict, subject_ct_id: str, vert) -> bool:
    """Return whether any POI on this vertebra is flagged for this subject."""
    subject = report_dict.get(subject_ct_id)
    if subject is None:
        return False
    target, _ = normalize_report_key(vert, 0)
    return any(v == target for v, _location in subject)
