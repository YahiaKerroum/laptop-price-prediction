"""Train the price model and write one self-describing artifact bundle.

Reports every split strategy and every baseline in a single table, so the
headline number cannot be quietly chosen from whichever split flatters most.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from laptop_price.config import CONFIG
from laptop_price.evaluation.baselines import baseline_table
from laptop_price.evaluation.metrics import (
    interval_coverage,
    regression_metrics,
    segment_report,
)
from laptop_price.evaluation.splits import Split, all_splits
from laptop_price.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from laptop_price.models.pipeline import build_pipeline, build_quantile_pipelines
from laptop_price.models.registry import ArtifactBundle, build_metadata

FEATURE_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def _xy(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return matrix[FEATURE_COLUMNS], matrix[TARGET]


def evaluate_on_split(
    matrix: pd.DataFrame,
    split: Split,
    *,
    estimator: Any | None = None,
    label: str = "HistGradientBoosting",
) -> tuple[Any, dict[str, float | str]]:
    """Fit on the split's training rows and score its test rows."""
    features, target = _xy(matrix)

    pipeline = build_pipeline(estimator=estimator)
    pipeline.fit(features.iloc[split.train_idx], target.iloc[split.train_idx])

    predictions = pipeline.predict(features.iloc[split.test_idx])
    metrics = regression_metrics(
        target.iloc[split.test_idx], predictions, label=f"{label} [{split.name}]"
    )
    return pipeline, metrics


def train(
    matrix: pd.DataFrame,
    *,
    fit_quantiles: bool = True,
    compare_models: bool = True,
) -> tuple[ArtifactBundle, pd.DataFrame]:
    """Train, evaluate honestly, and return the bundle plus the results table.

    The returned pipeline is refitted on the *stratified* train+val rows, since
    that is the largest training sample; the metrics table reports all three
    splits so the reader can see the time-based number, which is lower and is the
    one that answers "can we price a listing posted tomorrow".
    """
    features, target = _xy(matrix)
    splits = all_splits(matrix, target)
    records: list[dict[str, Any]] = []

    # --- baselines, scored on the time-based test fold ---------------------
    time_split = next(s for s in splits if s.name == "time_based")
    baseline_rows = baseline_table(
        target.iloc[time_split.train_idx],
        target.iloc[time_split.test_idx],
        groups_train=matrix["spec_signature"].iloc[time_split.train_idx],
        groups_test=matrix["spec_signature"].iloc[time_split.test_idx],
        component_cost_test=matrix["estimated_component_cost"].iloc[time_split.test_idx],
    )
    baseline_rows["model"] = baseline_rows["model"] + " [time_based]"
    records.extend(baseline_rows.to_dict("records"))

    # --- the model, on every split -----------------------------------------
    fitted: dict[str, Any] = {}
    for split in splits:
        pipeline, metrics = evaluate_on_split(matrix, split)
        fitted[split.name] = pipeline
        records.append(metrics)

    # --- comparison models, on the time-based split only --------------------
    if compare_models:
        for label, estimator, scaled in (
            ("Ridge", Ridge(alpha=1.0), True),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=CONFIG.model.random_state,
                ),
                False,
            ),
        ):
            pipeline = build_pipeline(estimator=estimator, scale_numeric=scaled)
            train_features = features.iloc[time_split.train_idx]
            if scaled or isinstance(estimator, RandomForestRegressor):
                # Neither handles NaN; median-impute inside the pipeline.
                pipeline = build_pipeline(estimator=estimator, scale_numeric=True)
            pipeline.fit(train_features, target.iloc[time_split.train_idx])
            records.append(
                regression_metrics(
                    target.iloc[time_split.test_idx],
                    pipeline.predict(features.iloc[time_split.test_idx]),
                    label=f"{label} [time_based]",
                )
            )

    results = pd.DataFrame(records)

    # --- final artifact: refit on train+val of the stratified split ---------
    strat = next(s for s in splits if s.name == "stratified_random")
    final_idx = np.concatenate([strat.train_idx, strat.val_idx])
    final_pipeline = build_pipeline()
    final_pipeline.fit(features.iloc[final_idx], target.iloc[final_idx])

    final_metrics = regression_metrics(
        target.iloc[strat.test_idx],
        final_pipeline.predict(features.iloc[strat.test_idx]),
        label="final",
    )

    # Metrics excluding the rows where the component estimate broke a unit tie.
    clean_mask = ~matrix["price_unit_ambiguous"].to_numpy()[strat.test_idx]
    if clean_mask.sum() > 0:
        clean_idx = strat.test_idx[clean_mask]
        final_metrics_clean = regression_metrics(
            target.iloc[clean_idx],
            final_pipeline.predict(features.iloc[clean_idx]),
            label="final (unit-ambiguous rows excluded)",
        )
    else:  # pragma: no cover - only if every row were ambiguous
        final_metrics_clean = dict(final_metrics)

    # Per-segment error, because an aggregate R2 hides that the model is much
    # worse on premium and rare machines than on the mainstream bulk of the
    # market (roadmap 4d).
    test_actual = target.iloc[strat.test_idx]
    test_predicted = final_pipeline.predict(features.iloc[strat.test_idx])
    segments = {
        "price_decile": pd.qcut(test_actual, 10, labels=False, duplicates="drop"),
        "brand": matrix["brand"].iloc[strat.test_idx],
        "city": matrix["city_grouped"].iloc[strat.test_idx],
        "listing_year": matrix["listing_year"].iloc[strat.test_idx],
        "condition": matrix["spec_Etat"].iloc[strat.test_idx].fillna("not stated"),
    }
    segment_reports = {
        name: segment_report(test_actual, test_predicted, values)
        for name, values in segments.items()
    }

    quantile_pipelines: dict[float, Any] = {}
    interval_stats: dict[str, float] = {}
    if fit_quantiles:
        quantile_pipelines = build_quantile_pipelines()
        for pipeline in quantile_pipelines.values():
            pipeline.fit(features.iloc[final_idx], target.iloc[final_idx])

        quantiles = sorted(quantile_pipelines)
        low = quantile_pipelines[quantiles[0]].predict(features.iloc[strat.test_idx])
        high = quantile_pipelines[quantiles[-1]].predict(features.iloc[strat.test_idx])
        interval_stats = interval_coverage(target.iloc[strat.test_idx], low, high)

    time_metrics = next(
        r for r in records if str(r["model"]).startswith("HistGradientBoosting [time_based]")
    )

    metadata = build_metadata(
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        target=TARGET,
        n_train=len(final_idx),
        n_test=len(strat.test_idx),
        split="stratified_random (artifact); all splits reported in metrics",
        metrics={
            "final": final_metrics,
            "final_excluding_unit_ambiguous": final_metrics_clean,
            "headline_time_based": time_metrics,
            "prediction_interval": interval_stats,
            "all_splits": records,
        },
        baselines=baseline_rows.to_dict("records"),
        notes=(
            "Predicts ASKING price, not sale price. Identical specs in this dataset "
            "sell 2-9x apart, so the point estimate should be read together with the "
            "10th-90th percentile range."
        ),
    )

    metadata["metrics"]["by_segment"] = {
        name: report.to_dict("records") for name, report in segment_reports.items()
    }

    bundle = ArtifactBundle(
        pipeline=final_pipeline,
        quantile_pipelines=quantile_pipelines,
        metadata=metadata,
    )
    return bundle, results


def format_results(results: pd.DataFrame) -> str:
    """Render the results table the way the reports should quote it."""
    columns = ["model", "n", "R2", "MAE", "RMSE", "MAPE", "MedAPE", "within_20pct"]
    view = results[[c for c in columns if c in results.columns]].copy()
    for column in ("R2",):
        view[column] = view[column].map(lambda v: f"{v:.4f}")
    for column in ("MAE", "RMSE"):
        view[column] = view[column].map(lambda v: f"{v:,.0f}")
    for column in ("MAPE", "MedAPE", "within_20pct"):
        if column in view:
            view[column] = view[column].map(lambda v: f"{v:.1f}%")
    return view.to_string(index=False)


__all__ = ["FEATURE_COLUMNS", "evaluate_on_split", "format_results", "train"]
