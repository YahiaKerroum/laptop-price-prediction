"""Turn a partial, human-supplied listing into a prediction.

The API and the Streamlit app share this module so there is one definition of
"what a request means". In particular, a caller supplies whatever they know; the
missing columns are filled with NaN rather than zero, because the model was
trained to split on missingness (that is the point of the ``spec_Etat`` fix) and
a fabricated zero would be read as a real, very low value.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from laptop_price.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from laptop_price.models.registry import ArtifactBundle, load_bundle

#: Fields a caller can reasonably be expected to know, with friendly aliases.
INPUT_ALIASES: dict[str, str] = {
    "ram": "RAM_SIZE",
    "ram_gb": "RAM_SIZE",
    "ssd": "SSD_SIZE",
    "ssd_gb": "SSD_SIZE",
    "hdd": "HDD_SIZE",
    "hdd_gb": "HDD_SIZE",
    "cpu_score": "cpu_mark",
    "gpu_score": "gpu_g3d_mark",
    "screen_size": "SCREEN_SIZE_SNAPPED",
    "resolution": "SCREEN_RESOLUTION_ENC",
    "condition": "spec_Etat",
    "city": "city_grouped",
    "year": "listing_year",
}


#: The headline range is a calibrated ~50% band around the point estimate,
#: ``point * exp(+/- K * spread)``, where ``spread`` is half the log-width of the
#: model's own 10-90 quantile band (so uncertain machines get wider ranges).
#:
#: K was fitted on the held-out stratified test fold of v20260912T2037 (3,251
#: listings), sending each one through ``to_feature_frame`` as the UI would:
#:
#: * CPU identified (``cpu_name``, plus ``gpu_name`` if dedicated): median error
#:   10.6%. Fitted on one half, K covered 49.9% of the other half; median band
#:   width 24% of the price, never wider than 28% (about +/-14%).
#: * CPU not identified (benchmark scores only): median error 27%, so an honest
#:   50% band is about +/-30%. The UI should ask for the processor by name.
#:
#: For comparison the raw 10-90 band covered only 69% on the same requests while
#: being up to 127% of the price wide. Refit both K whenever the model is retrained.
LIKELY_RANGE_K = 0.554
LIKELY_RANGE_K_CPU_UNKNOWN = 1.21
#: Cap on ``spread``: costs under a point of coverage and bounds the widest band.
LIKELY_RANGE_MAX_SPREAD = 0.25
#: Floor, so a model that is overconfident on one row cannot collapse the band.
LIKELY_RANGE_MIN_SPREAD = 0.05


@lru_cache(maxsize=4)
def get_bundle(version: str | None = None) -> ArtifactBundle:
    """Load and cache a model bundle. One disk read per process per version."""
    return load_bundle(version)


def normalise_input(listing: dict[str, Any]) -> dict[str, Any]:
    """Apply the friendly aliases and drop keys the model does not know."""
    known = {*NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    result: dict[str, Any] = {}
    for key, value in listing.items():
        column = INPUT_ALIASES.get(key, key)
        if column in known:
            result[column] = value
    return result


def to_feature_frame(listing: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row frame with every declared column, in the declared order.

    Derived features the caller cannot know (ratios, the seasonal encoding) are
    computed here so that a hand-built request produces the same representation
    as a row that came through the training pipeline.
    """
    from laptop_price.catalog import enrich

    enriched = enrich({INPUT_ALIASES.get(key, key): value for key, value in listing.items()})
    integrated_gpu = bool(enriched.pop("_integrated_gpu", False))
    cpu_known = bool(enriched.pop("_cpu_known", False))
    values = normalise_input(enriched)
    row: dict[str, Any] = {column: np.nan for column in NUMERIC_FEATURES}
    row |= {column: "UNKNOWN" for column in CATEGORICAL_FEATURES}
    row |= values

    def _f(name: str) -> float:
        try:
            value = float(row.get(name))
        except (TypeError, ValueError):
            return float("nan")
        return value

    ram, ssd, hdd = _f("RAM_SIZE"), _f("SSD_SIZE"), _f("HDD_SIZE")
    cpu, gpu = _f("cpu_mark"), _f("gpu_g3d_mark")

    ssd = 0.0 if np.isnan(ssd) else ssd
    hdd = 0.0 if np.isnan(hdd) else hdd
    row["SSD_SIZE"], row["HDD_SIZE"] = ssd, hdd

    row["total_storage"] = ssd + hdd
    row["has_hdd"] = int(hdd > 0)
    row["is_dual_drive"] = int(hdd > 0 and ssd > 0)
    row["has_dedicated_gpu"] = int(not integrated_gpu and not np.isnan(gpu) and gpu > 0)
    row["gpu_to_cpu_ratio"] = gpu / cpu if cpu else np.nan

    cost = _f("estimated_component_cost")
    if np.isnan(_f("perf_per_expected_dinar")) and cost > 0:
        row["perf_per_expected_dinar"] = cpu / cost
    if np.isnan(_f("model_family")):
        # Same four-level tier as features.build._model_family_tier.
        g = 0.0 if np.isnan(gpu) else gpu
        tier = 0
        if cpu < 10_000 and g < 4_000:
            tier = 1
        if 10_000 <= cpu <= 25_000 and 4_000 <= g <= 18_000:
            tier = 2
        if cpu >= 25_000 or g >= 18_000:
            tier = 3
        row["model_family"] = tier
    row["storage_per_ram"] = ssd / ram if ram else np.nan
    row["total_tdp"] = (0 if np.isnan(_f("tdp")) else _f("tdp")) + (
        0 if np.isnan(_f("gpu_tdp")) else _f("gpu_tdp")
    )
    row["etat_is_missing"] = int(np.isnan(_f("spec_Etat")))

    month = _f("listing_month")
    if np.isnan(month):
        month = 6.0  # mid-year, a neutral default for the seasonal encoding
        row["listing_month"] = month
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12))

    year = _f("listing_year")
    if not np.isnan(year):
        row["listing_month_index"] = (year - 2018) * 12 + month

    frame = pd.DataFrame([row])[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    frame.attrs["cpu_known"] = cpu_known
    return frame


def predict_one(
    listing: dict[str, Any],
    *,
    version: str | None = None,
) -> dict[str, Any]:
    """Price one listing.

    Returns a *range* rather than a bare number. Identical configurations in this
    dataset sell 2-9x apart, so a single figure claims more precision than the
    data supports.
    """
    bundle = get_bundle(version)
    frame = to_feature_frame(listing)

    point = float(bundle.pipeline.predict(frame)[0])
    result: dict[str, Any] = {
        "estimate_dzd": round(point, -2),
        "model_version": bundle.version,
        "currency": "DZD",
        "caveat": (
            "Asking price, not sale price. About half of comparable listings ask a "
            "price inside the range; the rest sit outside it, in both directions."
        ),
    }

    if bundle.quantile_pipelines:
        quantiles = sorted(bundle.quantile_pipelines)
        raw = [float(bundle.quantile_pipelines[q].predict(frame)[0]) for q in quantiles]

        # Quantile rearrangement. The models are fitted independently, so on
        # about 0.7% of listings they cross - a predicted 10th percentile above
        # the 50th, which is not a quantile function. Sorting the values and
        # reassigning them to the ordered levels is the standard remedy
        # (Chernozhukov et al.) and is provably no worse than leaving them.
        predictions = dict(zip(quantiles, sorted(raw), strict=True))
        low, high = predictions[quantiles[0]], predictions[quantiles[-1]]

        # The point estimate comes from a separate squared-error model, so on
        # about 1.4% of listings it lands outside the quantile interval. Showing
        # "most likely 105,200, range 111,200-175,500" is not defensible, so
        # widen the interval to contain the point rather than moving the point:
        # the interval is the softer claim, and the point estimate is what the
        # reported metrics were actually measured on.
        low, high = min(low, point), max(high, point)
        result["wide_range_dzd"] = [round(low, -2), round(high, -2)]
        result["quantiles"] = {str(q): round(v, -2) for q, v in predictions.items()}

        # The headline range: half of comparable listings ask a price inside it.
        # Centred on the point estimate in log space, so it always contains it.
        q_low, q_high = predictions[quantiles[0]], predictions[quantiles[-1]]
        if q_low > 0 and q_high > q_low:
            spread = (np.log(q_high) - np.log(q_low)) / 2
        else:
            spread = LIKELY_RANGE_MIN_SPREAD
        spread = float(np.clip(spread, LIKELY_RANGE_MIN_SPREAD, LIKELY_RANGE_MAX_SPREAD))
        cpu_known = frame.attrs.get("cpu_known", False)
        half = (LIKELY_RANGE_K if cpu_known else LIKELY_RANGE_K_CPU_UNKNOWN) * spread
        result["range_dzd"] = [
            round(point * float(np.exp(-half)), -2),
            round(point * float(np.exp(half)), -2),
        ]
        result["range_coverage"] = 0.5
        result["precision"] = "high" if cpu_known else "low"

    return result


def explain_one(
    listing: dict[str, Any],
    *,
    version: str | None = None,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Per-feature price contributions for one listing, via SHAP.

    Feeds the waterfall chart in the UI: "base 95,000 -> RTX 4060 +48,000 ->
    only 8GB RAM -12,000 -> 137,000 DZD".
    """
    import shap

    bundle = get_bundle(version)
    frame = to_feature_frame(listing)

    pipeline = getattr(bundle.pipeline, "regressor_", bundle.pipeline)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(frame)
    names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)[0]

    contributions = sorted(
        ({"feature": n, "contribution": float(v)} for n, v in zip(names, values, strict=True)),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return contributions[:top_n]


def find_similar(
    listing: dict[str, Any],
    matrix: pd.DataFrame,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """Nearest listings by specification, for the "similar machines" panel."""
    columns = ["RAM_SIZE", "SSD_SIZE", "cpu_mark", "gpu_g3d_mark", "SCREEN_SIZE_SNAPPED"]
    present = [c for c in columns if c in matrix.columns]

    query = to_feature_frame(listing)[present].to_numpy(dtype=float)
    candidates = matrix[present].to_numpy(dtype=float)

    # Scale each axis by its spread so cpu_mark (tens of thousands) does not
    # drown out RAM_SIZE (single digits).
    spread = np.nanstd(candidates, axis=0)
    spread[spread == 0] = 1.0
    distance = np.nansum(((candidates - query) / spread) ** 2, axis=1)

    order = np.argsort(distance)[:k]
    return matrix.iloc[order].assign(distance=np.sqrt(distance[order]))


__all__ = [
    "INPUT_ALIASES",
    "explain_one",
    "find_similar",
    "get_bundle",
    "normalise_input",
    "predict_one",
    "to_feature_frame",
]
