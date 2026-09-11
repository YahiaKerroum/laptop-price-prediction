"""Honest evaluation: metrics users understand, baselines worth beating, splits that don't flatter."""

from laptop_price.evaluation.baselines import baseline_table
from laptop_price.evaluation.metrics import regression_metrics, segment_report
from laptop_price.evaluation.splits import (
    group_kfold,
    stratified_split,
    time_based_split,
)

__all__ = [
    "baseline_table",
    "group_kfold",
    "regression_metrics",
    "segment_report",
    "stratified_split",
    "time_based_split",
]
