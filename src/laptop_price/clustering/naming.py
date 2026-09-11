"""Turning cluster ids into segment names.

A cluster is worthless until you can say what it *is*. "Cluster 2" tells a reader
nothing; "the ~90,000 DZD refurbished-ThinkPad-for-students segment, 12% of the
market" tells them everything. This module derives that description from each
cluster's centroid statistics against the market as a whole - no LLM needed, and
the rule is inspectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from laptop_price.features.build import TARGET

#: Thresholds separating performance tiers, in PassMark points.
GPU_GAMING = 8_000
GPU_DISCRETE = 2_000
CPU_STRONG = 18_000
CPU_WEAK = 6_000


def name_segment(profile: pd.Series, market: pd.Series) -> str:
    """Describe one cluster relative to the market.

    Reads a cluster's median specs against the market medians and composes a
    label from what actually distinguishes it.
    """
    parts: list[str] = []

    gpu = profile.get("gpu_g3d_mark", 0) or 0
    cpu = profile.get("cpu_mark", 0) or 0
    ram = profile.get("RAM_SIZE", 0) or 0
    screen = profile.get("SCREEN_SIZE_SNAPPED", 0) or 0
    price = profile.get(TARGET, 0) or 0
    market_price = market.get(TARGET, 1) or 1

    # --- what kind of machine ---
    if gpu >= GPU_GAMING and cpu >= CPU_STRONG:
        parts.append("gaming / workstation")
    elif gpu >= GPU_DISCRETE:
        parts.append("entry discrete-GPU")
    elif cpu >= CPU_STRONG:
        parts.append("performance ultrabook")
    elif cpu <= CPU_WEAK:
        parts.append("budget bureautique")
    else:
        parts.append("mainstream productivity")

    # --- form factor, only when it is distinctive ---
    if screen and screen <= 13.5:
        parts.append("compact")
    elif screen and screen >= 17:
        parts.append("desktop-replacement")

    # --- memory, only when it is distinctive ---
    if ram and ram >= 24:
        parts.append("high-memory")
    elif ram and ram <= 6:
        parts.append("low-memory")

    # --- price position ---
    ratio = price / market_price
    if ratio >= 1.6:
        position = "premium"
    elif ratio >= 1.15:
        position = "upper-mid"
    elif ratio <= 0.6:
        position = "cheap"
    elif ratio <= 0.85:
        position = "lower-mid"
    else:
        position = "mid-market"

    return f"{position} {', '.join(parts)}"


def describe_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    top_brands: int = 3,
) -> pd.DataFrame:
    """One row per segment: size, price level, defining specs, and a name.

    This is the table that makes a segmentation mean something. It is also where
    a degenerate clustering becomes obvious - a 90%-share row next to two slivers
    is visible at a glance in a way a silhouette of 0.98 is not.
    """
    frame = df.copy()
    frame["_segment"] = np.asarray(labels)

    numeric = [
        c
        for c in (
            TARGET,
            "RAM_SIZE",
            "SSD_SIZE",
            "cpu_mark",
            "gpu_g3d_mark",
            "SCREEN_SIZE_SNAPPED",
            "spec_Etat",
            "listing_year",
        )
        if c in frame.columns
    ]
    market = frame[numeric].median()

    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby("_segment"):
        profile = group[numeric].median()
        row: dict[str, object] = {
            "segment": int(segment),
            "name": "unclustered (noise)" if segment < 0 else name_segment(profile, market),
            "n": len(group),
            "share": len(group) / len(frame),
        }
        row |= {f"median_{c}": profile[c] for c in numeric}

        if "brand" in group.columns:
            brands = group["brand"].value_counts().head(top_brands)
            row["top_brands"] = ", ".join(f"{b} ({n})" for b, n in brands.items())

        # Price dispersion within the segment: a segment whose members all price
        # alike is genuinely a segment; one that spans 10x is a bag of leftovers.
        if TARGET in group:
            prices = group[TARGET].dropna()
            row["price_iqr_ratio"] = (
                float(prices.quantile(0.75) / prices.quantile(0.25)) if len(prices) > 3 else np.nan
            )
        rows.append(row)

    report = pd.DataFrame(rows).sort_values("share", ascending=False)
    return report.reset_index(drop=True)


def format_segments(report: pd.DataFrame) -> str:
    """Render the segment table for a report."""
    view = pd.DataFrame(
        {
            "seg": report["segment"],
            "name": report["name"].str.slice(0, 44),
            "n": report["n"],
            "share": report["share"].map("{:.1%}".format),
        }
    )
    if f"median_{TARGET}" in report:
        view["median price"] = report[f"median_{TARGET}"].map("{:,.0f}".format)
    if "price_iqr_ratio" in report:
        view["P75/P25"] = report["price_iqr_ratio"].map(
            lambda v: f"{v:.2f}x" if pd.notna(v) else "-"
        )
    for column, label in (("median_cpu_mark", "cpu"), ("median_gpu_g3d_mark", "gpu"),
                          ("median_RAM_SIZE", "ram")):
        if column in report:
            view[label] = report[column].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
    return view.to_string(index=False)


def price_model_by_segment(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    min_size: int = 200,
) -> pd.DataFrame:
    """Does segmenting actually help price prediction?

    Fits a small model inside each segment and compares its error with one
    global model scored on the same rows. This closes the loop: segmentation
    that measurably improves pricing is segmentation that means something. If the
    per-segment models do no better, the segments are decorative.
    """
    from sklearn.model_selection import cross_val_predict

    from laptop_price.evaluation.metrics import regression_metrics
    from laptop_price.models.pipeline import build_pipeline
    from laptop_price.models.train import FEATURE_COLUMNS

    frame = df.copy()
    frame["_segment"] = np.asarray(labels)
    features, target = frame[FEATURE_COLUMNS], frame[TARGET]

    global_predictions = cross_val_predict(build_pipeline(), features, target, cv=3)

    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby("_segment"):
        if segment < 0 or len(group) < min_size:
            continue
        index = group.index
        local = cross_val_predict(
            build_pipeline(), features.loc[index], target.loc[index], cv=3
        )
        specialist = regression_metrics(target.loc[index], local, label=f"segment {segment}")
        generalist = regression_metrics(
            target.loc[index], global_predictions[frame.index.get_indexer(index)]
        )
        rows.append(
            {
                "segment": int(segment),
                "n": len(group),
                "global_MedAPE": generalist["MedAPE"],
                "segment_MedAPE": specialist["MedAPE"],
                "improvement_pp": generalist["MedAPE"] - specialist["MedAPE"],
            }
        )

    return pd.DataFrame(rows).sort_values("improvement_pp", ascending=False).reset_index(drop=True)


__all__ = [
    "describe_segments",
    "format_segments",
    "name_segment",
    "price_model_by_segment",
]
