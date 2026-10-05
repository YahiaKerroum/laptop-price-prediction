"""Canonical filesystem layout.

Every module and notebook resolves paths through here so that nothing depends on
the current working directory. ``data/raw`` is treated as immutable: nothing in
this project writes to it.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until the directory containing pyproject.toml."""
    env = os.environ.get("LAPTOP_PRICE_ROOT")
    if env:
        return Path(env).resolve()

    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate the project root (no pyproject.toml found)")


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MAPPINGS_DIR = DATA_DIR / "mappings"
LABELS_DIR = DATA_DIR / "labels"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RULES_DIR = REPORTS_DIR / "rules"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# --- raw inputs (read-only) -------------------------------------------------
RAW_LISTINGS = RAW_DIR / "data.csv"
CPUS_CSV = RAW_DIR / "cpus.csv"
GPUS_CSV = RAW_DIR / "gpus.csv"
CPU_PRICES_CSV = RAW_DIR / "cpu_prices.csv"
GPU_PRICES_CSV = RAW_DIR / "gpu_prices.csv"
CPU_DDR_MAP = MAPPINGS_DIR / "cpu_ddr_map.csv"
CPU_STORAGE_MAP = MAPPINGS_DIR / "cpu_storage_map.csv"

# --- intermediate stages ----------------------------------------------------
DATA_CLEANED = INTERIM_DIR / "data_cleaned.csv"
DATA_WITH_CPUS = INTERIM_DIR / "data_with_cpus.csv"
DATA_WITH_CPUS_GPUS = INTERIM_DIR / "data_with_cpus_gpus.csv"
RAM_STORAGE_CLEANED = INTERIM_DIR / "cleanedramstoragedata.csv"
SCREEN_CLEANED = INTERIM_DIR / "screen_cleaned_data.csv"
CPUS_PRICED = INTERIM_DIR / "cpus_priced.csv"

# --- modelling inputs -------------------------------------------------------
PRE_PROCESSED = PROCESSED_DIR / "pre_processed_data.csv"
# Two distinct model-ready tables, deliberately kept apart:
#   MODEL_READY -- written by notebooks/03, reproducing the original course
#                  artifact exactly. Consumed by notebooks 04 and 05.
#   FEATURES    -- written by laptop_price.data.build_model_ready, carrying the
#                  restored city/date/brand columns, NaN-encoded condition and
#                  troll-price removal. Consumed by training and serving.
# Before they were separated, re-running notebook 03 silently clobbered the
# package's matrix and training then failed on a missing target column.
MODEL_READY = PROCESSED_DIR / "model_ready_data.csv"
FEATURES = PROCESSED_DIR / "features.csv"
CLUSTERING_DATA = PROCESSED_DIR / "clustering_optimized_data.csv"
ANOMALY_LABELS = LABELS_DIR / "anomaly_labels.csv"

ALL_DIRS = (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    MAPPINGS_DIR,
    LABELS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RULES_DIR,
)


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    name for name in dir() if name.isupper() or name.startswith("find_") or name == "ensure_dirs"
]
