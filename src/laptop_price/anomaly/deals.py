"""Underpriced-deal detection - the most useful thing this project can ship.

The idea is one sign flip away from the regression model that already exists.
Residual = predicted - actual:

* residual **strongly positive** and the listing otherwise ordinary -> a genuine
  bargain: the market is asking less than comparable machines fetch.
* residual **strongly positive** and the listing is also a joint (specs, price)
  outlier -> far more likely a scam, a typo, or a parts-only machine.
* residual **strongly negative** -> overpriced.

Separating the first two is the whole trick, and it is why this module consumes
:mod:`laptop_price.anomaly.detectors` rather than thresholding the residual alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from laptop_price.config import CONFIG
from laptop_price.features.build import TARGET
from laptop_price.models.registry import ArtifactBundle


def residual_scores(matrix: pd.DataFrame, bundle: ArtifactBundle) -> pd.DataFrame:
    """Predicted price, residual, and the relative discount for every listing.

    ``discount_pct`` is ``(predicted - actual) / predicted * 100``: how far below
    the model's estimate the seller is asking.
    """
    # The columns this bundle was fitted on, not the current feature list: a
    # bundle trained before a feature was added must still be scoreable.
    features = matrix[[*bundle.numeric_features, *bundle.categorical_features]]
    predicted = np.asarray(bundle.pipeline.predict(features), dtype=float)
    actual = pd.to_numeric(matrix[TARGET], errors="coerce").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        discount = np.where(predicted > 0, (predicted - actual) / predicted * 100, np.nan)

    scores = pd.DataFrame(
        {
            "predicted": predicted,
            "actual": actual,
            "residual": predicted - actual,
            "discount_pct": discount,
        },
        index=matrix.index,
    )

    # Where the interval models exist, a bargain is better defined as "below the
    # 10th percentile of what this configuration fetches" than as a bare residual.
    if bundle.quantile_pipelines:
        quantiles = sorted(bundle.quantile_pipelines)
        scores["p10"] = bundle.quantile_pipelines[quantiles[0]].predict(features)
        scores["p90"] = bundle.quantile_pipelines[quantiles[-1]].predict(features)
        scores["below_p10"] = scores["actual"] < scores["p10"]

    return scores


def classify_listings(
    matrix: pd.DataFrame,
    bundle: ArtifactBundle,
    *,
    outlier_flags: pd.Series | None = None,
    deal_threshold: float | None = None,
    scam_threshold: float | None = None,
) -> pd.DataFrame:
    """Label each listing ``bargain`` / ``suspicious`` / ``overpriced`` / ``normal``.

    Parameters
    ----------
    outlier_flags
        Joint (specs, price) outlier flags from
        :func:`laptop_price.anomaly.detectors.score_listings`. Without them every
        deep discount is reported as a bargain, which is exactly the mistake a
        deal feed must not make.
    """
    cfg = CONFIG.anomaly
    deal_threshold = (cfg.deal_threshold if deal_threshold is None else deal_threshold) * 100
    scam_threshold = (cfg.scam_threshold if scam_threshold is None else scam_threshold) * 100

    scores = residual_scores(matrix, bundle)
    flags = (
        pd.Series(False, index=matrix.index)
        if outlier_flags is None
        else outlier_flags.reindex(matrix.index).fillna(False).astype(bool)
    )
    scores["is_outlier"] = flags

    verdict = pd.Series("normal", index=matrix.index, dtype=object)
    discount = scores["discount_pct"]

    verdict[discount <= -deal_threshold] = "overpriced"
    verdict[discount >= deal_threshold] = "bargain"
    # Too good to be true, or odd in spec space as well: not a deal, a warning.
    verdict[(discount >= scam_threshold) | ((discount >= deal_threshold) & flags)] = "suspicious"

    scores["verdict"] = verdict
    return scores


def rank_deals(
    matrix: pd.DataFrame,
    bundle: ArtifactBundle,
    *,
    top_k: int | None = None,
    exclude_suspicious: bool = True,
    detect_outliers: bool = True,
) -> pd.DataFrame:
    """The "best deals right now" feed, ranked by relative discount.

    Returns the listing columns a reader needs to act, joined to the scores.
    """
    top_k = CONFIG.anomaly.top_k if top_k is None else top_k

    outlier_flags = None
    if detect_outliers:
        from laptop_price.anomaly.detectors import fit_detectors, score_listings

        ensemble = fit_detectors(matrix)
        # The union, not the majority vote. The detectors find sharply different
        # structure here - LocalOutlierFactor's flags are disjoint from the
        # global methods' (Jaccard 0.00) - so requiring agreement would let most
        # anomalies through. For a feed that tells people where to spend money,
        # a false "not a bargain" is much cheaper than a false "bargain".
        outlier_flags = score_listings(ensemble, matrix)["any_outlier"]

    scores = classify_listings(matrix, bundle, outlier_flags=outlier_flags)

    context_columns = [
        c
        for c in (
            "brand",
            "city_grouped",
            "RAM_SIZE",
            "SSD_SIZE",
            "cpu_mark",
            "gpu_g3d_mark",
            "spec_Etat",
            "listing_year",
            "cpu_name",  # optional: present when the caller joined part names on
            "gpu_name",
            TARGET,
        )
        if c in matrix.columns
    ]
    joined = matrix[context_columns].join(scores)

    candidates = joined[joined["verdict"] == "bargain"] if exclude_suspicious else joined
    return (
        candidates.sort_values("discount_pct", ascending=False).head(top_k).reset_index(drop=True)
    )


def precision_at_k(
    ranked: pd.DataFrame,
    labels: pd.Series,
    *,
    positive: str = "bargain",
    k: int = 50,
) -> float:
    """Precision@k against hand-applied labels.

    Unsupervised anomaly detection is unfalsifiable without some ground truth.
    Labelling ~200 listings by hand into ``data/labels/anomaly_labels.csv`` turns
    this section into a real result; see ``docs/modeling.md``.
    """
    top = ranked.head(k)
    matched = labels.reindex(top.index).dropna()
    if matched.empty:
        return float("nan")
    return float((matched == positive).mean())


__all__ = [
    "classify_listings",
    "precision_at_k",
    "rank_deals",
    "residual_scores",
]
