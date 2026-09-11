"""Gower distance for mixed numeric and categorical data.

Euclidean distance on this dataset is the wrong tool. Brand and city have no
numeric meaning, one-hot encoding them makes every pair of distinct brands
equidistant while inflating dimensionality, and the raw magnitudes of
``cpu_mark`` (up to 57,591) against ``RAM_SIZE`` (up to 128) mean the unscaled
space is effectively one-dimensional.

Gower handles each column in its own terms:

* **numeric** - absolute difference divided by the column's range, so every
  feature contributes on a 0-1 scale regardless of its units
* **categorical** - 0 when the values match, 1 when they do not

and averages across columns. Implemented here rather than pulled in as a
dependency: it is forty lines of arithmetic, and the packaged versions
materialise a full n x n matrix with no way to sample.

Memory
------
A full pairwise matrix is O(n²). At 16,255 listings that is 2.1 GB in float64,
so :func:`gower_matrix` refuses inputs above ``max_rows`` rather than exhausting
memory. Use :func:`sample_for_distance` to take a stratified subsample first -
distance-based diagnostics do not need every row, and the segmentation itself
runs on the full dataset through UMAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Above this many rows a full pairwise matrix stops being reasonable.
DEFAULT_MAX_ROWS = 6_000


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def _numeric_ranges(values: np.ndarray) -> np.ndarray:
    """Per-column range, with zero-range columns neutralised.

    A constant column contributes nothing to the distance. Leaving its range at
    zero would divide by zero; setting it to 1 makes every difference 0, which
    is the correct contribution.
    """
    spans = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    spans[~np.isfinite(spans) | (spans == 0)] = 1.0
    return spans


def gower_vector(
    row: pd.Series,
    frame: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Gower distance from one row to every row of ``frame``.

    Returns an array of length ``len(frame)`` in [0, 1].
    """
    numeric, categorical = _split_columns(frame)
    weight = np.array(
        [(weights or {}).get(c, 1.0) for c in [*numeric, *categorical]], dtype=float
    )

    parts = []
    if numeric:
        values = frame[numeric].to_numpy(dtype=float)
        spans = _numeric_ranges(values)
        target = row[numeric].to_numpy(dtype=float)
        diff = np.abs(values - target) / spans
        parts.append(np.nan_to_num(diff, nan=1.0))
    if categorical:
        left = frame[categorical].astype("string").to_numpy()
        target = row[categorical].astype("string").to_numpy()
        parts.append((left != target).astype(float))

    stacked = np.hstack(parts)
    return (stacked * weight).sum(axis=1) / weight.sum()


def gower_matrix(
    df: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> np.ndarray:
    """Full pairwise Gower distance matrix.

    Raises
    ------
    ValueError
        If ``df`` has more than ``max_rows`` rows. A 16,255-row matrix is 2.1 GB;
        failing loudly beats being killed by the OOM reaper halfway through a
        notebook. Subsample with :func:`sample_for_distance` first.
    """
    if len(df) > max_rows:
        gigabytes = len(df) ** 2 * 8 / 1024**3
        raise ValueError(
            f"refusing to build a {len(df)}x{len(df)} distance matrix "
            f"({gigabytes:.1f} GB). Subsample to <= {max_rows} rows with "
            "sample_for_distance(), or raise max_rows deliberately."
        )

    numeric, categorical = _split_columns(df)
    weight = np.array(
        [(weights or {}).get(c, 1.0) for c in [*numeric, *categorical]], dtype=float
    )
    n = len(df)
    total = np.zeros((n, n), dtype=float)

    if numeric:
        values = df[numeric].to_numpy(dtype=float)
        spans = _numeric_ranges(values)
        for index, column in enumerate(numeric):
            column_values = values[:, index]
            diff = np.abs(column_values[:, None] - column_values[None, :]) / spans[index]
            # A missing value is maximally distant from everything, including
            # another missing value: "unknown" is not evidence of similarity.
            missing = np.isnan(column_values)
            diff[missing, :] = 1.0
            diff[:, missing] = 1.0
            total += diff * weight[index]

    offset = len(numeric)
    for index, column in enumerate(categorical):
        codes = df[column].astype("string").fillna("__NA__").to_numpy()
        total += (codes[:, None] != codes[None, :]).astype(float) * weight[offset + index]

    return total / weight.sum()


def sample_for_distance(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_MAX_ROWS,
    stratify_on: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample for distance-matrix work, preserving the shape of ``stratify_on``.

    Distance-based diagnostics (silhouette on Gower, hierarchical linkage) do not
    need every row, but they do need a sample that looks like the market -
    a uniform sample under-represents the thin premium tail that the original
    clustering mistook for a segment.
    """
    if len(df) <= n:
        return df.copy()

    if stratify_on is None or stratify_on not in df.columns:
        return df.sample(n, random_state=random_state)

    strata = pd.qcut(df[stratify_on], q=10, labels=False, duplicates="drop")
    per_stratum = max(1, n // strata.nunique())
    return (
        df.groupby(strata, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=random_state))
        .reset_index(drop=True)
    )


__all__ = [
    "DEFAULT_MAX_ROWS",
    "gower_matrix",
    "gower_vector",
    "sample_for_distance",
]
