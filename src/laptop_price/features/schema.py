"""Data contracts asserted at each stage boundary.

The audit found a dataset that was 43% duplicate configurations, carried troll
prices the report claimed were removed, and shipped a scaler whose feature count
did not match its model - none of which failed loudly. These schemas are the
tripwire: a stage that produces something structurally wrong now raises instead
of writing a plausible-looking CSV.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from laptop_price.config import CONFIG
from laptop_price.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

_PRICE_MIN = float(CONFIG.price.min_dzd)
_PRICE_MAX = float(CONFIG.price.max_dzd)


#: Contract for ``data/processed/pre_processed_data.csv``.
PREPROCESSED_SCHEMA = DataFrameSchema(
    {
        "price_preview": Column(float, nullable=True),
        "created_at": Column(object, nullable=True),
        "city": Column(object, nullable=True),
        "spec_Etat": Column(object, nullable=True),
        "model_name": Column(object, nullable=True),
        "mapped_cpu_name": Column(object, nullable=True),
        "cores": Column(float, Check.in_range(1, 128), nullable=True),
        "SCREEN_SIZE_SNAPPED": Column(float, Check.in_range(10, 20), nullable=True),
        "SCREEN_RESOLUTION_ENC": Column(float, Check.in_range(1, 9), nullable=True),
    },
    strict=False,
    coerce=True,
    name="preprocessed_listings",
)


#: Contract for the model-ready matrix produced by ``build_feature_matrix``.
FEATURE_SCHEMA = DataFrameSchema(
    {
        TARGET: Column(float, Check.in_range(_PRICE_MIN, _PRICE_MAX), nullable=False),
        "RAM_SIZE": Column(float, Check.in_range(1, 256), nullable=True),
        "SSD_SIZE": Column(float, Check.in_range(0, 16_384), nullable=True),
        "HDD_SIZE": Column(float, Check.in_range(0, 16_384), nullable=True),
        "cpu_mark": Column(float, Check.ge(0), nullable=True),
        "gpu_g3d_mark": Column(float, Check.ge(0), nullable=True),
        # NaN is the *correct* encoding for an unstated condition (roadmap A2);
        # a 0 here would mean the old, misleading scheme had crept back in.
        "spec_Etat": Column(float, Check.in_range(1, 3), nullable=True),
        "etat_is_missing": Column(int, Check.isin([0, 1])),
        "listing_year": Column(float, Check.in_range(2010, 2030), nullable=True),
        "brand": Column(object, nullable=False),
        "city_grouped": Column(object, nullable=False),
        "spec_signature": Column(object, nullable=False),
    },
    strict=False,
    coerce=True,
    name="model_ready_features",
)


def validate_preprocessed(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Assert the preprocessing stage produced a well-formed frame."""
    return PREPROCESSED_SCHEMA.validate(df, lazy=lazy)


def validate_feature_matrix(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Assert the feature matrix is model-ready.

    Raises
    ------
    pandera.errors.SchemaErrors
        With every failing check collected, rather than only the first.
    """
    missing = [c for c in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if c not in df.columns]
    if missing:
        raise ValueError(f"feature matrix is missing declared columns: {missing}")
    return FEATURE_SCHEMA.validate(df, lazy=lazy)


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """A per-column data card: dtype, null rate, cardinality, range.

    Used to generate the table in ``docs/data-dictionary.md`` so the
    documentation cannot drift away from the data.
    """
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "null_rate": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            described = series.describe()
            row |= {
                "min": round(float(described.get("min", float("nan"))), 3),
                "median": round(float(series.median(skipna=True)), 3),
                "max": round(float(described.get("max", float("nan"))), 3),
            }
        else:
            top = series.dropna().astype(str).value_counts().head(1)
            row |= {
                "min": "",
                "median": f"mode={top.index[0]}" if not top.empty else "",
                "max": "",
            }
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "FEATURE_SCHEMA",
    "PREPROCESSED_SCHEMA",
    "describe_features",
    "validate_feature_matrix",
    "validate_preprocessed",
]
