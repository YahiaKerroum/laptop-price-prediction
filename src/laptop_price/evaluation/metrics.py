"""Regression metrics, reported in dinars rather than log space.

The original notebooks reported R², RMSE and MAE only. For a price product the
metric users actually understand is "typically within X%", so median absolute
percentage error is reported alongside - and it is the number for the slide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def regression_metrics(
    y_true: object,
    y_pred: object,
    *,
    label: str = "model",
) -> dict[str, float | str]:
    """Compute the full metric panel in DZD space.

    Both inputs must already be in dinars - invert any log transform before
    calling, otherwise R² is computed on a different scale than it is reported.

    Returns
    -------
    dict
        ``R2``, ``MAE``, ``RMSE``, ``MAPE`` and ``MedAPE`` (both as percentages),
        plus ``within_10pct`` / ``within_20pct`` coverage and the sample count.
    """
    truth = _as_array(y_true)
    prediction = _as_array(y_pred)

    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[finite], prediction[finite]
    if truth.size == 0:
        raise ValueError("no finite (y_true, y_pred) pairs to score")

    # Guard against a zero denominator; prices are positive by construction, but
    # a bad upstream stage should not produce a silent inf here.
    nonzero = truth != 0
    ape = np.full(truth.shape, np.nan)
    ape[nonzero] = np.abs(prediction[nonzero] - truth[nonzero]) / np.abs(truth[nonzero]) * 100

    return {
        "model": label,
        "n": int(truth.size),
        "R2": float(r2_score(truth, prediction)),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAPE": float(np.nanmean(ape)),
        "MedAPE": float(np.nanmedian(ape)),
        "within_10pct": float(np.nanmean(ape <= 10) * 100),
        "within_20pct": float(np.nanmean(ape <= 20) * 100),
    }


def segment_report(
    y_true: object,
    y_pred: object,
    segments: pd.Series,
    *,
    min_count: int = 30,
) -> pd.DataFrame:
    """Per-segment error table.

    Aggregate R² hides that a model is usually poor on premium and rare machines.
    Pass ``segments`` as price decile, brand, city, condition or listing year.
    """
    frame = pd.DataFrame(
        {
            "y_true": _as_array(y_true),
            "y_pred": _as_array(y_pred),
            "segment": np.asarray(segments).ravel(),
        }
    ).dropna()

    rows = []
    for name, group in frame.groupby("segment", observed=True):
        if len(group) < min_count:
            continue
        metrics = regression_metrics(group["y_true"], group["y_pred"], label=str(name))
        metrics["segment"] = name
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["segment", "n", "R2", "MAE", "MedAPE"])

    report = pd.DataFrame(rows).drop(columns=["model"])
    ordered = ["segment", "n", "R2", "MAE", "RMSE", "MAPE", "MedAPE"]
    return report[ordered].sort_values("MedAPE", ascending=False).reset_index(drop=True)


def metrics_frame(records: list[dict[str, float | str]]) -> pd.DataFrame:
    """Assemble metric dicts into a sorted comparison table."""
    frame = pd.DataFrame(records)
    return frame.sort_values("MedAPE").reset_index(drop=True)


def interval_coverage(
    y_true: object,
    lower: object,
    upper: object,
) -> dict[str, float]:
    """Coverage and width of a prediction interval.

    Reported as first-class metrics because the product output is a *range*
    (identical laptops in this dataset sell 2-9x apart, so a point estimate
    overstates what the model knows).
    """
    truth, low, high = _as_array(y_true), _as_array(lower), _as_array(upper)
    inside = (truth >= low) & (truth <= high)
    width = high - low
    return {
        "coverage_pct": float(np.mean(inside) * 100),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "median_relative_width": float(
            np.median(width / np.where(truth == 0, np.nan, truth)) * 100
        ),
    }


__all__ = [
    "interval_coverage",
    "metrics_frame",
    "regression_metrics",
    "segment_report",
]
