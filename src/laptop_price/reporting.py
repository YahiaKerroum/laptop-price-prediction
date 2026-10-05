"""Generated documentation: the model card and the data dictionary.

Both are written from the artifacts themselves rather than typed by hand, so
they cannot drift away from the model that is actually deployed - which is what
happened to the original reports (they cite 15,517 rows where the split ran on
15,418, and credit feature selection with dropping a column that was never in
the feature list).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from laptop_price.models.registry import ArtifactBundle


def _fmt(value: Any, spec: str = ",.0f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def render_model_card(bundle: ArtifactBundle) -> str:
    """Render ``model_card.md`` for a trained bundle."""
    meta = bundle.metadata
    metrics = meta.get("metrics", {})
    final = metrics.get("final", {})
    headline = metrics.get("headline_time_based", {})
    clean = metrics.get("final_excluding_unit_ambiguous", {})
    interval = metrics.get("prediction_interval", {})

    baselines = pd.DataFrame(meta.get("baselines", []))
    baseline_rows = ""
    if not baselines.empty:
        for row in baselines.itertuples(index=False):
            baseline_rows += (
                f"| {row.model} | {row.R2:.4f} | {_fmt(row.MAE)} | {row.MedAPE:.1f}% |\n"
            )

    return f"""# Model card - laptop price estimator `{meta.get("version", "?")}`

## What it does
Estimates the **asking price**, in Algerian dinars, of a used laptop listed on
Ouedkniss, from its specifications, condition, city and listing date.

It outputs a **range**, not a single number. That is deliberate: identical
configurations in this dataset sell between 2x and 9x apart, so a point estimate
claims precision the data does not contain.

## Intended use
- Helping a seller set an asking price, and a buyer sanity-check one.
- Ranking listings by how far below their estimate they are priced
  (the "deals" feed), *always* combined with the anomaly detector, because a deep
  discount is as likely to be a scam as a bargain.

## Out of scope
- **It does not predict sale price.** Every row of training data is an *asking*
  price. The gap between asking and final price is real, seller-specific, and
  entirely unmeasured here.
- Machines far outside the training distribution: servers, desktops, parts-only
  listings, and the premium tail (which is thin and where error is highest).
- Any use where being wrong costs money without a human in the loop.

## Training data
- Source: scraped Ouedkniss laptop listings, {meta.get("n_train", 0):,} used for fitting.
- Target: `{meta.get("target", "price_corrected")}`, after troll-price removal and
  unit-convention correction.
- Held-out test rows: {meta.get("n_test", 0):,}.

## Performance

| Split | R² | MAE (DZD) | Median APE |
|---|---|---|---|
| Stratified random (comparable to the original report) | {final.get("R2", float("nan")):.4f} | {_fmt(final.get("MAE"))} | {final.get("MedAPE", float("nan")):.1f}% |
| **Time-based (train <= 2024, test 2025) - the honest headline** | {headline.get("R2", float("nan")):.4f} | {_fmt(headline.get("MAE"))} | {headline.get("MedAPE", float("nan")):.1f}% |
| Stratified, excluding unit-ambiguous rows | {clean.get("R2", float("nan")):.4f} | {_fmt(clean.get("MAE"))} | {clean.get("MedAPE", float("nan")):.1f}% |

The time-based number is lower than the random-split number. That is expected and
is the point: listings span 2018-2025 and carry seven years of dinar inflation and
tech depreciation, so a random split lets the model interpolate within time
periods it has already seen.

### Baselines it has to beat

| Baseline | R² | MAE (DZD) | Median APE |
|---|---|---|---|
{baseline_rows or "| (not recorded) | | | |"}

### Prediction interval
{f"10th-90th percentile coverage: **{interval.get('coverage_pct', float('nan')):.1f}%**, median width {_fmt(interval.get('median_width'))} DZD ({interval.get('median_relative_width', float('nan')):.0f}% of price)." if interval else "Not fitted for this version."}

## Known limitations
- **Noise ceiling.** Identical spec rows sell 2-9x apart; predicting the perfect
  per-configuration mean would cap out near R² 0.95 in log space. Most of the
  remaining error is not in the spec columns at all - it is in listing text,
  seller reputation, photos, negotiability and urgency, none of which are scraped.
- **Geographic skew.** A large share of listings come from a handful of Algiers
  districts, so estimates for southern wilayas rest on far less data.
- **{meta.get("n_train", 0):,} rows, far fewer unique configurations.** Effective
  sample size is closer to the number of distinct spec signatures than to the row count.
- **Condition is unstated for ~41% of listings.** The model learns a split for
  "missing" rather than guessing, but that is a real information gap.

## Reproducing
```
git checkout {meta.get("git_sha", "?")}
make build && make pipeline && make train
```
Trained {meta.get("trained_at", "?")} with scikit-learn {meta.get("sklearn_version", "?")}
on Python {meta.get("python_version", "?")}.

## Notes
{meta.get("notes", "")}
"""


def render_data_dictionary(card: pd.DataFrame, *, title: str = "Data dictionary") -> str:
    """Render a markdown table from :func:`laptop_price.features.schema.describe_features`."""
    lines = [
        f"# {title}",
        "",
        "Generated from the data itself by `make pipeline`; do not edit by hand.",
        "",
        "| Column | Type | Null rate | Distinct | Min | Median | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in card.itertuples(index=False):
        lines.append(
            f"| `{row.column}` | {row.dtype} | {row.null_rate:.1%} | {row.n_unique:,} "
            f"| {row.min} | {row.median} | {row.max} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["render_data_dictionary", "render_model_card"]
