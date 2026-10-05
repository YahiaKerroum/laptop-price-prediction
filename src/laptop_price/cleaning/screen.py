"""Screen size and resolution normalisation.

Scraped screen fields arrive as free text in a mix of conventions: European
decimal commas (``15,6``), spaced resolutions (``1920 x 1080``), marketing names
(``FullHD``, ``2K``, ``3K OLED``) and outright nonsense. This module snaps sizes
to the canonical panel sizes, maps resolutions onto a nine-tier quality scale,
and fills gaps from the per-model mode - manufacturers ship one panel across a
product line, so a ThinkPad with an unknown screen is very likely the same panel
as the other ThinkPads.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

#: Panel sizes that account for ~80% of the dataset. Adding 14.1 and 16.1 lifts
#: coverage by only ~1.9%, which does not justify two more near-duplicate levels.
CANONICAL_SIZES: np.ndarray = np.array([11.6, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3])

#: A measured size within this many inches of a canonical size snaps to it.
SNAP_TOLERANCE = 0.3

#: Physically plausible laptop panel range, in inches.
MIN_SCREEN_INCHES = 10.0
MAX_SCREEN_INCHES = 20.0

RESOLUTION_MAP: dict[str, str] = {
    # --- HD ---
    "1366x768": "HD",
    "1280x720": "HD",
    "hd": "HD",
    # --- HD+ ---
    "1440x900": "HD+",
    "1600x900": "HD+",
    "1536x1024": "HD+",
    "1280x800": "HD+",
    # --- FHD ---
    "1920x1080": "FHD",
    "1920x1080fhd": "FHD",
    "1920x1080fullhd": "FHD",
    "fullhd": "FHD",
    "fhd": "FHD",
    "1080p": "FHD",
    "fhd1080p": "FHD",
    # --- WUXGA (FHD+, 16:10) ---
    "1920x1200": "WUXGA",
    "1920x1200fhd": "WUXGA",
    "1920x1200fhd+": "WUXGA",
    "1920x1200wuxga": "WUXGA",
    "1920x1280": "WUXGA",
    "fhd+": "WUXGA",
    "fullhd+": "WUXGA",
    "wuxga": "WUXGA",
    # --- QHD / 2K ---
    "2560x1440": "QHD",
    "2560x1440qhd": "QHD",
    "qhd": "QHD",
    "wqhd": "QHD",
    "2k": "QHD",
    "qhd2k": "QHD",
    "1440p": "QHD",
    "2048x1080": "QHD",
    # --- QHD+ (16:10) ---
    "2560x1600": "QHD+",
    "2560x1600qhd+": "QHD+",
    "2400x1600": "QHD+",
    "2240x1400": "QHD+",
    "2560x1664": "QHD+",
    "2256x1504": "QHD+",
    "2496x1664": "QHD+",
    "2360x1640": "QHD+",
    "2304x1536": "QHD+",
    "wqxga": "QHD+",
    "wqxga+": "QHD+",
    "qhd+": "QHD+",
    "2.5k": "QHD+",
    # --- 3K-class high-density panels ---
    "2880x1800": "3K",
    "2880x1920": "3K",
    "2880x1864": "3K",
    "3072x1920": "3K",
    "3000x2000": "3K",
    "3024x1964": "3K",
    "3200x2000": "3K",
    "2736x1824": "3K",
    "2736x1834": "3K",
    "2736x1823": "3K",
    "3456x2234": "3K",
    "3k": "3K",
    "2.8k": "3K",
    "3koled": "3K",
    "3kretina": "3K",
    # --- 4K / UHD ---
    "3840x2160": "4K",
    "3840x2400": "4K",
    "3456x2160": "4K",
    "3240x2160": "4K",
    "4k": "4K",
    "4kuhd": "4K",
    # --- 5K ---
    "5120x2880": "5K",
    "5k": "5K",
}

#: HD < HD+ < FHD < WUXGA < QHD < QHD+ < 3K < 4K < 5K
RESOLUTION_ORDINAL: dict[str, int] = {
    "HD": 1,
    "HD+": 2,
    "FHD": 3,
    "WUXGA": 4,
    "QHD": 5,
    "QHD+": 6,
    "3K": 7,
    "4K": 8,
    "5K": 9,
}

VALID_RESOLUTIONS: tuple[str, ...] = tuple(RESOLUTION_ORDINAL)

#: Pixel dimensions per tier, for the ``pixels`` and ``ppi`` features.
TIER_PIXELS: dict[str, tuple[int, int]] = {
    "HD": (1366, 768),
    "HD+": (1600, 900),
    "FHD": (1920, 1080),
    "WUXGA": (1920, 1200),
    "QHD": (2560, 1440),
    "QHD+": (2560, 1600),
    "3K": (2880, 1800),
    "4K": (3840, 2160),
    "5K": (5120, 2880),
}


def parse_screen_size(value: Any) -> float:
    """``"15,6\""`` -> 15.6. Values outside 10-20 inches become NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).replace(",", ".")
    match = pd.Series([text]).str.extract(r"(\d+\.?\d*)")[0].iloc[0]
    if pd.isna(match):
        return float("nan")
    size = float(match)
    if size < MIN_SCREEN_INCHES or size > MAX_SCREEN_INCHES:
        return float("nan")
    return size


def snap_screen_size(size: Any) -> float:
    """Snap to the nearest canonical panel size when within :data:`SNAP_TOLERANCE`.

    Rare but genuine sizes (e.g. 18.4") are left alone rather than forced.
    """
    if size is None or (isinstance(size, float) and math.isnan(size)):
        return float("nan")
    value = float(size)
    distances = np.abs(CANONICAL_SIZES - value)
    if round(float(distances.min()), 2) <= SNAP_TOLERANCE:
        return float(CANONICAL_SIZES[int(distances.argmin())])
    return value


def normalize_resolution(value: Any) -> str:
    """Lower-case and strip whitespace: ``"1920 x 1080"`` -> ``"1920x1080"``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).lower().replace(" ", "").strip()


def resolution_tier(value: Any) -> Any:
    """Map a raw resolution onto one of :data:`VALID_RESOLUTIONS`, else NaN."""
    tier = RESOLUTION_MAP.get(normalize_resolution(value))
    return tier if tier in RESOLUTION_ORDINAL else np.nan


def encode_resolution(tier: Any) -> float:
    """Ordinal code for a resolution tier; unknown -> NaN."""
    if tier is None or (isinstance(tier, float) and math.isnan(tier)):
        return float("nan")
    return float(RESOLUTION_ORDINAL.get(str(tier), float("nan")))


def fill_by_model_mode(df: pd.DataFrame, column: str, group: str = "model_name") -> pd.Series:
    """Fill missing values in ``column`` with the mode within each ``group``.

    Falls back to the global mode for models that have no observed value at all.
    """
    series = df[column]

    def _mode(values: pd.Series) -> Any:
        observed = values.dropna()
        return observed.mode().iloc[0] if not observed.empty else np.nan

    per_group = df.groupby(group)[column].transform(_mode)
    filled = series.fillna(per_group)

    global_observed = series.dropna()
    if not global_observed.empty:
        filled = filled.fillna(global_observed.mode().iloc[0])
    return filled


def clean_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise screen size and resolution, then impute from the per-model mode.

    Adds ``SCREEN_SIZE_SNAPPED``, ``SCREEN_RESOLUTION_STD``,
    ``SCREEN_RESOLUTION_ENC`` and the provenance flags
    ``screen_size_imputed`` / ``screen_resolution_imputed``.

    ``SCREEN_FREQUENCY`` is dropped: it is missing for the overwhelming majority
    of listings and is not recoverable from anything else in the dataset.
    """
    out = df.copy()

    out["SCREEN_SIZE"] = out["SCREEN_SIZE"].map(parse_screen_size)
    out["SCREEN_SIZE_SNAPPED"] = out["SCREEN_SIZE"].map(snap_screen_size)

    out["SCREEN_RESOLUTION"] = out["SCREEN_RESOLUTION"].map(normalize_resolution)
    out["SCREEN_RESOLUTION_STD"] = out["SCREEN_RESOLUTION"].map(resolution_tier)

    out["screen_size_imputed"] = out["SCREEN_SIZE_SNAPPED"].isna()
    out["screen_resolution_imputed"] = out["SCREEN_RESOLUTION_STD"].isna()

    out["SCREEN_SIZE_SNAPPED"] = fill_by_model_mode(out, "SCREEN_SIZE_SNAPPED")
    out["SCREEN_RESOLUTION_STD"] = fill_by_model_mode(out, "SCREEN_RESOLUTION_STD")
    out["SCREEN_RESOLUTION_ENC"] = out["SCREEN_RESOLUTION_STD"].map(encode_resolution)

    if "SCREEN_FREQUENCY" in out.columns:
        out = out.drop(columns=["SCREEN_FREQUENCY"])

    return out


__all__ = [
    "CANONICAL_SIZES",
    "RESOLUTION_MAP",
    "RESOLUTION_ORDINAL",
    "TIER_PIXELS",
    "VALID_RESOLUTIONS",
    "clean_screen",
    "encode_resolution",
    "fill_by_model_mode",
    "normalize_resolution",
    "parse_screen_size",
    "resolution_tier",
    "snap_screen_size",
]
