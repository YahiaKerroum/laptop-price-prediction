"""Build the model-ready feature matrix from the preprocessed listings.

What changed relative to the original ``encoding_feature_engineering`` notebook
(roadmap A2, A3, §3b, §3c, §3d):

* ``created_at``, ``city`` and ``model_name`` are **restored**. The notebook
  dropped all three and replaced brand with ``model_family``, a four-level tier
  computed from ``cpu_mark`` and ``gpu_g3d_mark`` - i.e. a lossy re-encoding of
  two features the model already had, with 76% of rows in a single bucket.
  City median price spans 2.35x and listings span 2018-2025, so both carry real
  signal. ``model_family`` is kept, but only as a comparison column.
* ``spec_Etat`` missing is encoded **NaN, not 0**. 41% of rows are missing and
  their mean price sits *between* BON ETAT and JAMAIS UTILISE, so putting them
  at the bottom of a 1-2-3 scale was actively misleading. An explicit
  ``etat_is_missing`` flag carries the (informative) missingness.
* Ratio, temporal and geographic features are added.
* The never-called ``infer_laptop_state`` helper is gone. It read
  ``price_preview`` to infer condition, which would have leaked the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from laptop_price.cleaning.memory import encode_ram_type, to_gb
from laptop_price.cleaning.screen import TIER_PIXELS
from laptop_price.config import CONFIG

TARGET = "price_corrected"

#: Plausible installed-memory bounds, in GB. Below this is a megabyte figure
#: misparsed as gigabytes; above it is a storage capacity in the wrong column.
MIN_PLAUSIBLE_RAM_GB = 1.0
MAX_PLAUSIBLE_RAM_GB = 128.0

#: Plausible single-drive capacity, in GB. Above this the value is a scraping
#: artefact - one listing carries "250320500GB", three drive options run together
#: with no separator.
MAX_PLAUSIBLE_STORAGE_GB = 16_384.0

#: Ordinal condition scale. Missing deliberately maps to NaN, not to 0.
ETAT_ORDINAL: dict[str, float] = {
    "MOYEN": 1.0,
    "BON TAT": 2.0,
    "BON ÉTAT": 2.0,
    "BON ETAT": 2.0,
    "JAMAIS UTILIS": 3.0,
    "JAMAIS UTILISÉ": 3.0,
    "JAMAIS UTILISE": 3.0,
}

NUMERIC_FEATURES: tuple[str, ...] = (
    # --- raw specifications ---
    "RAM_SIZE",
    "SSD_SIZE",
    "HDD_SIZE",
    "RAM_TYPE",
    "cores",
    "cpu_mark",
    "tdp",
    "gpu_g3d_mark",
    "gpu_g2d_mark",
    "gpu_tdp",
    "SCREEN_SIZE_SNAPPED",
    "SCREEN_RESOLUTION_ENC",
    "cpu_generation_normalized",
    "spec_Etat",
    # --- engineered ratios (roadmap 3b) ---
    "gpu_to_cpu_ratio",
    "storage_per_ram",
    "total_storage",
    "pixels",
    "ppi",
    "total_tdp",
    "perf_per_expected_dinar",
    "estimated_component_cost",
    # --- temporal (roadmap 3c) ---
    "listing_year",
    "listing_month",
    "listing_month_index",
    "month_sin",
    "month_cos",
    # --- flags ---
    "etat_is_missing",
    "has_hdd",
    "is_dual_drive",
    "has_dedicated_gpu",
    "model_family",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("brand", "city_grouped", "cpu_manufacturer", "cpu_family")

#: Columns that identify a configuration, used for group-aware cross-validation.
SIGNATURE_COLUMNS: tuple[str, ...] = (
    "RAM_SIZE",
    "SSD_SIZE",
    "HDD_SIZE",
    "cpu_mark",
    "gpu_g3d_mark",
    "SCREEN_SIZE_SNAPPED",
    "SCREEN_RESOLUTION_ENC",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_created_at(series: pd.Series) -> pd.Series:
    """Parse the scraped timestamp.

    The scraper emitted ``"2021 10 01T18:01:44.000Z"`` - space separated rather
    than ISO hyphenated - so the default parser silently produces NaT on some
    pandas versions. Normalise the separators first.
    """
    normalised = series.astype("string").str.replace(
        r"^(\d{4})\s+(\d{2})\s+(\d{2})", r"\1-\2-\3", regex=True
    )
    return pd.to_datetime(normalised, errors="coerce", format="ISO8601", utc=True)


def encode_etat(value: object) -> float:
    """Ordinal condition code; missing or unrecognised -> NaN (never 0)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return ETAT_ORDINAL.get(str(value).strip().upper(), float("nan"))


def group_rare_levels(series: pd.Series, min_count: int, other: str = "OTHER") -> pd.Series:
    """Collapse levels seen fewer than ``min_count`` times into a single level.

    Applied to ``city`` (470 levels) and ``brand`` (46). Keeps the long tail from
    exploding one-hot width while preserving the levels that carry signal.
    """
    counts = series.value_counts(dropna=True)
    keep = set(counts[counts >= min_count].index)
    return series.where(series.isin(keep), other).fillna(other).astype("string")


def spec_signature(df: pd.DataFrame, columns: tuple[str, ...] = SIGNATURE_COLUMNS) -> pd.Series:
    """A hashable key identifying a configuration.

    43% of rows are exact duplicates in feature space, so a random split can put
    the same configuration on both sides. Grouping on this signature is what
    makes :func:`laptop_price.evaluation.splits.group_kfold` honest.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    # Explicit concatenation rather than .agg("|".join): the columns are a mix of
    # float and string dtypes, and the join path is dtype-sensitive across
    # pandas versions.
    parts = [df[column].astype("string").fillna("NA") for column in present]
    signature = parts[0]
    for part in parts[1:]:
        signature = signature + "|" + part
    return signature


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce to float.

    The PassMark columns are scraped with thousands separators (``"19,108"``),
    so a bare ``to_numeric`` silently nulls almost all of them - which is exactly
    how ``cpu_mark`` would quietly become 98% missing.
    """
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        series = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _pixels_from_tier(tier: object) -> float:
    dims = TIER_PIXELS.get(str(tier))
    return float(dims[0] * dims[1]) if dims else float("nan")


def _diagonal_pixels(tier: object) -> float:
    dims = TIER_PIXELS.get(str(tier))
    return float(np.hypot(*dims)) if dims else float("nan")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame, *, drop_missing_target: bool = True) -> pd.DataFrame:
    """Turn preprocessed listings into the model-ready matrix.

    Returns a frame holding :data:`NUMERIC_FEATURES`, :data:`CATEGORICAL_FEATURES`,
    the :data:`TARGET`, and the bookkeeping columns ``spec_signature`` and
    ``price_unit_ambiguous`` that evaluation needs.
    """
    out = df.copy()
    cfg = CONFIG.features

    # --- capacities: "16GB"/"1TB" -> float GB ------------------------------
    for column in ("RAM_SIZE", "SSD_SIZE", "HDD_SIZE"):
        if column in out.columns and not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(to_gb)

    # --- DDR generation: "DDR4" -> ordinal ---------------------------------
    if "RAM_TYPE" in out.columns and not pd.api.types.is_numeric_dtype(out["RAM_TYPE"]):
        out["RAM_TYPE"] = out["RAM_TYPE"].map(encode_ram_type)

    # --- everything else numeric -------------------------------------------
    for column in (
        "cpu_mark",
        "gpu_g3d_mark",
        "gpu_g2d_mark",
        "gpu_tdp",
        "tdp",
        "cores",
        "RAM_SIZE",
        "SSD_SIZE",
        "HDD_SIZE",
        "RAM_TYPE",
        "SCREEN_SIZE_SNAPPED",
        "SCREEN_RESOLUTION_ENC",
        "cpu_generation_normalized",
    ):
        if column in out.columns:
            out[column] = _numeric(out[column])

    # A missing disk means "no disk", which is 0 - distinct from an unknown RAM
    # size, which stays NaN so the model can split on it.
    out["SSD_SIZE"] = out["SSD_SIZE"].fillna(0.0)
    out["HDD_SIZE"] = out["HDD_SIZE"].fillna(0.0)
    out["RAM_SIZE"] = out["RAM_SIZE"].replace(0.0, np.nan)

    # An implausible capacity is unknown, not zero: NaN so the model can split on
    # it rather than reading a scraping artefact as "this machine has no disk".
    for column in ("SSD_SIZE", "HDD_SIZE"):
        out.loc[out[column] > MAX_PLAUSIBLE_STORAGE_GB, column] = np.nan

    # Implausible RAM survives the swap detector in two shapes: sub-gigabyte
    # values (a megabyte figure parsed as GB) and storage capacities that leaked
    # into the column. Null them rather than let the model fit on nonsense.
    implausible_ram = out["RAM_SIZE"].notna() & (
        (out["RAM_SIZE"] < MIN_PLAUSIBLE_RAM_GB) | (out["RAM_SIZE"] > MAX_PLAUSIBLE_RAM_GB)
    )
    out.loc[implausible_ram, "RAM_SIZE"] = np.nan

    # --- condition (roadmap A2) -------------------------------------------
    raw_etat = out["spec_Etat"]
    out["etat_is_missing"] = raw_etat.isna().astype(int)
    out["spec_Etat"] = raw_etat.map(encode_etat)

    # --- restored columns (roadmap A3) ------------------------------------
    timestamp = parse_created_at(out["created_at"])
    out["listing_year"] = timestamp.dt.year
    out["listing_month"] = timestamp.dt.month
    # A single monotonic index so the model can learn price drift over 2018-2025.
    out["listing_month_index"] = (timestamp.dt.year - 2018) * 12 + timestamp.dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["listing_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["listing_month"] / 12)

    out["brand"] = group_rare_levels(out["model_name"], cfg.brand_min_count)
    out["city_grouped"] = group_rare_levels(out["city"], cfg.city_min_count)
    for column in ("cpu_manufacturer", "cpu_family"):
        out[column] = out[column].astype("string").fillna("UNKNOWN")

    # --- flags ------------------------------------------------------------
    out["has_hdd"] = (out["HDD_SIZE"] > 0).astype(int)
    out["is_dual_drive"] = ((out["HDD_SIZE"] > 0) & (out["SSD_SIZE"] > 0)).astype(int)
    out["has_dedicated_gpu"] = out["DEDICATED_GPU"].notna().astype(int)

    # --- ratios (roadmap 3b) ----------------------------------------------
    out["gpu_to_cpu_ratio"] = out["gpu_g3d_mark"] / out["cpu_mark"].replace(0, np.nan)
    out["storage_per_ram"] = out["SSD_SIZE"] / out["RAM_SIZE"].replace(0, np.nan)
    out["total_storage"] = out["SSD_SIZE"] + out["HDD_SIZE"]
    out["total_tdp"] = out["tdp"].fillna(0) + out["gpu_tdp"].fillna(0)

    out["pixels"] = out["SCREEN_RESOLUTION_STD"].map(_pixels_from_tier)
    out["ppi"] = out["SCREEN_RESOLUTION_STD"].map(_diagonal_pixels) / out[
        "SCREEN_SIZE_SNAPPED"
    ].replace(0, np.nan)

    if "estimated_component_cost" not in out.columns:
        out["estimated_component_cost"] = np.nan
    out["perf_per_expected_dinar"] = out["cpu_mark"] / out["estimated_component_cost"].replace(
        0, np.nan
    )

    # --- model_family, kept only as a comparison column --------------------
    out["model_family"] = _model_family_tier(out)

    # --- bookkeeping -------------------------------------------------------
    out["spec_signature"] = spec_signature(out)
    if "price_unit_ambiguous" not in out.columns:
        out["price_unit_ambiguous"] = False

    columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        TARGET,
        "spec_signature",
        "price_unit_ambiguous",
    ]
    matrix = out[[c for c in columns if c in out.columns]].copy()

    if drop_missing_target:
        matrix = matrix[matrix[TARGET].notna()]

    return matrix.reset_index(drop=True)


def _model_family_tier(df: pd.DataFrame) -> pd.Series:
    """The original four-level performance tier: entry=1, mid=2, high-end=3, unknown=0.

    Retained so the results table can show what restoring real brand information
    bought us over this proxy - not as a replacement for ``brand``.
    """
    cpu = df["cpu_mark"]
    gpu = df["gpu_g3d_mark"].fillna(0)
    tier = pd.Series(0, index=df.index, dtype=int)
    tier[(cpu < 10_000) & (gpu < 4_000)] = 1
    tier[(cpu.between(10_000, 25_000)) & (gpu.between(4_000, 18_000))] = 2
    tier[(cpu >= 25_000) | (gpu >= 18_000)] = 3
    return tier


__all__ = [
    "CATEGORICAL_FEATURES",
    "MAX_PLAUSIBLE_RAM_GB",
    "MAX_PLAUSIBLE_STORAGE_GB",
    "MIN_PLAUSIBLE_RAM_GB",
    "ETAT_ORDINAL",
    "NUMERIC_FEATURES",
    "SIGNATURE_COLUMNS",
    "TARGET",
    "build_feature_matrix",
    "encode_etat",
    "group_rare_levels",
    "parse_created_at",
    "spec_signature",
]
