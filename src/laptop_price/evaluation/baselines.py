"""Baselines the model has to beat before its complexity is earned.

The original reports compared XGBoost only against linear regression. That sets
the bar too low: a pricing model that cannot beat "the median of identical
listings" is not doing anything a GROUP BY could not.

Three baselines
---------------
``global_median``
    Predict the same number for every laptop. The floor.
``spec_median``
    Predict the median price of listings with the same spec signature, falling
    back to the global median for unseen configurations. This is the strong one -
    the audit measured that predicting the perfect per-config mean would give
    R² ~= 0.954 in log space, so this baseline is close to the noise ceiling for
    configurations we have seen before.
``component_cost``
    The rule-based CPU + GPU + RAM + storage sum. Interesting because it is the
    same heuristic that resolves unit-ambiguous prices, so this answers "does the
    learned model beat the hand-written price table?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from laptop_price.config import CONFIG
from laptop_price.evaluation.metrics import regression_metrics


def global_median_baseline(y_train: pd.Series, n_test: int) -> np.ndarray:
    """Predict the training median everywhere."""
    return np.full(n_test, float(np.median(y_train)))


def spec_median_baseline(
    y_train: pd.Series,
    groups_train: pd.Series,
    groups_test: pd.Series,
) -> np.ndarray:
    """Predict the training median price of each spec signature.

    Unseen configurations fall back to the global training median.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_train, dtype=float), "g": np.asarray(groups_train).astype(str)}
    )
    lookup = frame.groupby("g")["y"].median()
    fallback = float(frame["y"].median())
    keys = pd.Series(np.asarray(groups_test).astype(str))
    return keys.map(lookup).fillna(fallback).to_numpy(dtype=float)


def component_cost_baseline(component_cost: pd.Series, fallback: float) -> np.ndarray:
    """Use the rule-based build cost directly as the price prediction.

    Clipped to the plausible price band. The heuristic is unbounded - a scraped
    "12,800GB SSD" produces a build cost in the hundreds of millions - and a
    single such row drives R² to about -48,000, which says nothing about the
    baseline's actual quality. Any real pricing rule would clamp its output, so
    the baseline does too.
    """
    values = pd.to_numeric(component_cost, errors="coerce").fillna(fallback)
    return values.clip(CONFIG.price.min_dzd, CONFIG.price.max_dzd).to_numpy(dtype=float)


def baseline_table(
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    groups_train: pd.Series | None = None,
    groups_test: pd.Series | None = None,
    component_cost_test: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every available baseline on the test fold.

    Returns a frame with the same columns as
    :func:`laptop_price.evaluation.metrics.regression_metrics`, ready to be
    concatenated above the model rows in the results table.
    """
    rows = [
        regression_metrics(
            y_test,
            global_median_baseline(y_train, len(y_test)),
            label="baseline: global median",
        )
    ]

    if groups_train is not None and groups_test is not None:
        rows.append(
            regression_metrics(
                y_test,
                spec_median_baseline(y_train, groups_train, groups_test),
                label="baseline: median of identical spec",
            )
        )

    if component_cost_test is not None:
        rows.append(
            regression_metrics(
                y_test,
                component_cost_baseline(component_cost_test, float(np.median(y_train))),
                label="baseline: component cost sum",
            )
        )

    return pd.DataFrame(rows)


__all__ = [
    "baseline_table",
    "component_cost_baseline",
    "global_median_baseline",
    "spec_median_baseline",
]
