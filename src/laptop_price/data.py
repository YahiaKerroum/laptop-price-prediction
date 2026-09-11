"""Dataset loading and the stage that turns preprocessed listings into features.

Stage graph
-----------
::

    data/raw/data.csv  ---- notebooks/01_preprocessing.ipynb ---->  interim/
    interim/ + mappings/ ----------- (same notebook) ----------->  processed/pre_processed_data.csv
    processed/pre_processed_data.csv --- build_model_ready() --->  processed/features.csv

The first two arrows are the hand-curated CPU/GPU enrichment and live in the
notebook. Everything from ``pre_processed_data.csv`` onward is this module, so
training and serving share one implementation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from laptop_price import paths
from laptop_price.cleaning.price import correct_prices
from laptop_price.features.build import build_feature_matrix


def load_price_maps(
    cpu_path: Path | None = None, gpu_path: Path | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Load the CPU and GPU component price lookups, in DZD."""
    cpus = pd.read_csv(cpu_path or paths.CPU_PRICES_CSV)
    gpus = pd.read_csv(gpu_path or paths.GPU_PRICES_CSV)
    return (
        dict(zip(cpus["cpu_name"], cpus["estimated_price"], strict=True)),
        dict(zip(gpus["gpu_name"], gpus["estimated_price"], strict=True)),
    )


def load_preprocessed(path: Path | None = None) -> pd.DataFrame:
    """Read ``data/processed/pre_processed_data.csv``.

    The shipped file already carries a ``price_corrected`` column produced by the
    old, unscoped correction. It is dropped here so the rescoped rule in
    :mod:`laptop_price.cleaning.price` is the only thing that ever sets the target.
    """
    frame = pd.read_csv(path or paths.PRE_PROCESSED)
    return frame.drop(columns=["price_corrected"], errors="ignore")


def build_model_ready(
    df: pd.DataFrame | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Preprocessed listings -> the model-ready feature matrix.

    Applies troll-price removal, the rescoped unit correction, and feature
    construction, then asserts the output against the pandera contract.
    """
    frame = load_preprocessed() if df is None else df
    cpu_prices, gpu_prices = load_price_maps()

    priced = correct_prices(frame, cpu_prices, gpu_prices)
    matrix = build_feature_matrix(priced)

    if validate:
        # Imported lazily: pandera is a quality-gate dependency, and the module
        # should stay importable without it.
        from laptop_price.features.schema import validate_feature_matrix

        validate_feature_matrix(matrix)

    return matrix


def write_model_ready(matrix: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the feature matrix, creating the directory if needed."""
    target = path or paths.FEATURES
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(target, index=False)
    return target


def load_model_ready(path: Path | None = None) -> pd.DataFrame:
    """Read a previously built feature matrix."""
    return pd.read_csv(path or paths.FEATURES)


__all__ = [
    "build_model_ready",
    "load_model_ready",
    "load_preprocessed",
    "load_price_maps",
    "write_model_ready",
]
