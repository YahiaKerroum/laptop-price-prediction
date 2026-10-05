"""RAM and storage normalisation.

Ouedkniss listings put RAM and storage in free-text fields, and the two are
routinely swapped ("RAM: 512GB, SSD: 16GB"). This module detects and repairs
those swaps, splits dual-drive entries, and normalises every capacity to GB.

The CPU -> DDR-type and CPU -> storage lookup tables in ``data/mappings/`` are
used to fill gaps where the listing itself is silent.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from laptop_price import paths

#: Capacities that are realistic as laptop/workstation RAM. 128GB is included
#: because high-end MacBooks and mobile workstations do ship with it.
REALISTIC_RAM_GB: frozenset[str] = frozenset(
    {"2", "4", "6", "8", "12", "16", "24", "32", "48", "64", "96", "128"}
)

#: Below this a GB figure is more likely RAM than storage.
MIN_STORAGE_GB = 256

#: Ordinal encoding of memory generations. The spacing reflects relative
#: bandwidth/recency rather than a bare 1..n index.
RAM_TYPE_ORDINAL: dict[str, float] = {
    "SDRAM": 0.5,
    "DDR": 1.0,
    "DDR2": 1.5,
    "DDR3": 2.0,
    "DDR3L": 2.0,
    "LPDDR3": 2.0,
    "LPDDR": 2.5,
    "DDR4": 3.0,
    "LPDDR4": 3.0,
    "DDR4X": 3.2,
    "LPDDR4X": 3.2,
    "DDR4/DDR5": 3.5,
    "DDR5": 4.0,
    "LPDDR5": 4.0,
    "UNIFIED": 4.0,
    "DDR5X": 4.2,
    "LPDDR5X": 4.2,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _blank(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def is_ram_value(value: Any) -> bool:
    """True when the text looks like a RAM capacity rather than a disk size."""
    if _blank(value):
        return False
    cleaned = str(value).strip().upper().replace("GB", "").replace(" ", "")
    return cleaned in REALISTIC_RAM_GB


def is_storage_value(value: Any) -> bool:
    """True when the text looks like a disk capacity (>=256GB, any TB, or ``A+B``)."""
    if _blank(value):
        return False
    cleaned = str(value).strip().upper()
    if "+" in cleaned or "TB" in cleaned:
        return True
    digits = cleaned.replace("GB", "").replace(" ", "")
    try:
        return int(float(digits)) >= MIN_STORAGE_GB
    except ValueError:
        return False


def needs_swap(ram_value: Any, ssd_value: Any) -> bool:
    """Detect the common data-entry error where RAM and SSD are reversed.

    A swap only fires when it actually *improves* the row. Two storage-looking
    values are not a swap: exchanging them leaves both columns just as wrong, and
    the implausible RAM figure gets nulled downstream either way.

    (The original notebook's docstring claimed this case returned False while its
    code returned True. The docstring described the better behaviour, so that is
    what is implemented here.)

    >>> needs_swap("512GB", "16GB")     # storage in RAM, RAM in storage
    True
    >>> needs_swap("16GB", "512GB")     # correct assignment
    False
    >>> needs_swap("", "16GB")          # RAM value stranded in the storage column
    True
    >>> needs_swap("256GB", "512GB")    # both storage - swapping gains nothing
    False
    """
    ram = "" if _blank(ram_value) else str(ram_value).strip()
    ssd = "" if _blank(ssd_value) else str(ssd_value).strip()

    ram_is_storage = is_storage_value(ram)
    ram_is_ram = is_ram_value(ram)
    ssd_is_ram = is_ram_value(ssd)

    # The RAM column is empty and the storage column holds a RAM-sized value.
    if not ram and ssd_is_ram:
        return True
    # Unambiguously reversed.
    if ram_is_storage and ssd_is_ram:
        return True
    # A disk capacity sits in the RAM column and the storage column is free.
    return ram_is_storage and not ram_is_ram and not ssd


def parse_dual_storage(value: Any) -> tuple[Any, Any]:
    """Split ``"1TB+240GB"`` into ``("1TB", "240GB")``; single drives keep a None second slot."""
    if _blank(value) or "+" not in str(value):
        return value, None
    parts = str(value).split("+")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return value, None


def normalize_storage_to_gb(value: Any) -> Any:
    """Normalise a capacity string to a ``"<n>GB"`` form using the marketing TB (1TB = 1000GB)."""
    if _blank(value):
        return value

    text = str(value).strip()

    # "1TB 512GB" - space-separated dual drive; keep the first capacity.
    if " " in text and ("GB" in text.upper() or "TB" in text.upper()):
        for part in text.split():
            if "GB" in part.upper() or "TB" in part.upper():
                text = part
                break

    upper = text.upper()
    if "TB" in upper:
        try:
            return f"{int(float(upper.replace('TB', '').strip()) * 1000)}GB"
        except ValueError:
            return text
    if "GB" in upper:
        return upper
    try:
        return f"{int(float(text))}GB"
    except ValueError:
        return text


def to_gb(value: Any) -> float:
    """Capacity string -> float GB. Blank or unparseable -> 0.0.

    Handles the three ways sellers write more than one number:

    * ``"256GB+1TB"`` - a dual-drive machine, so the capacities are **summed**
    * ``"256GB/512GB"`` - a choice of configurations, so the **first** is taken
    * ``"1TB 128GB"`` - the same, space separated

    A run-together value like ``"250320500GB"`` (three drive options with no
    separator at all) parses to a nonsense number; the plausibility clamp in
    :func:`laptop_price.features.build.build_feature_matrix` nulls it.
    """
    if _blank(value):
        return 0.0
    text = str(value).upper().strip()

    # Dual drive: both are installed, so add them.
    if "+" in text:
        return sum(to_gb(part) for part in text.split("+"))

    # A choice between configurations: take the first offered.
    for separator in ("/", "|", ","):
        if separator in text:
            return to_gb(text.split(separator)[0])
    if " " in text and text.count("GB") + text.count("TB") > 1:
        return to_gb(text.split()[0])

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    number = float(match.group(1))
    if "TB" in text:
        return number * 1000
    if "MB" in text:
        return number / 1024
    return number


def encode_ram_type(ram_type: Any) -> float:
    """Ordinal encoding of a DDR generation; unknown or missing -> NaN.

    NaN rather than 0: gradient-boosted trees learn their own split for missing,
    and a 0 would place "unknown" below SDRAM on the quality scale.
    """
    if _blank(ram_type):
        return float("nan")

    text = str(ram_type).strip().upper().replace(" ", "")
    if text in RAM_TYPE_ORDINAL:
        return RAM_TYPE_ORDINAL[text]
    # longest key first, so LPDDR5X wins over DDR5
    for key in sorted(RAM_TYPE_ORDINAL, key=len, reverse=True):
        if key in text:
            return RAM_TYPE_ORDINAL[key]
    return float("nan")


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------


def _strip_frequency(cpu_name: Any) -> str:
    """``"Intel Core i5-8250U @ 1.60GHz"`` -> ``"Intel Core i5-8250U"``."""
    return re.split(r"\s*@\s*", str(cpu_name))[0].strip().upper()


def load_cpu_ddr_map(path: Any = None) -> dict[str, str]:
    """CPU name -> supported DDR generation, keyed without the clock suffix."""
    frame = pd.read_csv(path or paths.CPU_DDR_MAP)
    name_col, value_col = frame.columns[0], frame.columns[1]
    return {
        _strip_frequency(name): value
        for name, value in zip(frame[name_col], frame[value_col], strict=True)
        if not _blank(value)
    }


def load_cpu_storage_map(path: Any = None) -> dict[str, str]:
    """CPU name -> typical storage configuration, keyed without the clock suffix."""
    frame = pd.read_csv(path or paths.CPU_STORAGE_MAP)
    name_col, value_col = frame.columns[0], frame.columns[1]
    return {
        _strip_frequency(name): value
        for name, value in zip(frame[name_col], frame[value_col], strict=True)
        if not _blank(value)
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def clean_memory_and_storage(
    df: pd.DataFrame,
    *,
    cpu_ddr_map: dict[str, str] | None = None,
    cpu_storage_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Repair swapped columns, split dual drives, normalise units, fill from CPU maps.

    Returns a copy with ``RAM_SIZE``/``SSD_SIZE``/``HDD_SIZE`` as numeric GB and
    ``RAM_TYPE`` ordinally encoded, plus the boolean provenance columns
    ``ram_ssd_swapped``, ``ram_type_imputed`` and ``is_dual_drive``.
    """
    out = df.copy()
    ddr_map = cpu_ddr_map if cpu_ddr_map is not None else load_cpu_ddr_map()
    storage_map = cpu_storage_map if cpu_storage_map is not None else load_cpu_storage_map()

    # 1. un-swap RAM / SSD
    swapped = [
        needs_swap(ram, ssd) for ram, ssd in zip(out["RAM_SIZE"], out["SSD_SIZE"], strict=True)
    ]
    out["ram_ssd_swapped"] = swapped
    swap_idx = out.index[out["ram_ssd_swapped"]]
    out.loc[swap_idx, ["RAM_SIZE", "SSD_SIZE"]] = out.loc[swap_idx, ["SSD_SIZE", "RAM_SIZE"]].values

    # 2. split dual drives: secondary capacity becomes the HDD when HDD is empty
    primary, secondary = zip(*(parse_dual_storage(v) for v in out["SSD_SIZE"]), strict=True)
    out["is_dual_drive"] = [s is not None for s in secondary]
    out["SSD_SIZE"] = [normalize_storage_to_gb(p) for p in primary]
    hdd_fill = [normalize_storage_to_gb(s) if s is not None else None for s in secondary]
    out["HDD_SIZE"] = [
        existing if not _blank(existing) else fill
        for existing, fill in zip(out["HDD_SIZE"], hdd_fill, strict=True)
    ]

    # 3. fill an absent DDR generation from the CPU's memory controller
    missing_type = out["RAM_TYPE"].isna() | (out["RAM_TYPE"].astype(str).str.strip() == "")
    inferred = out.loc[missing_type, "mapped_cpu_name"].map(
        lambda name: ddr_map.get(_strip_frequency(name))
    )
    out.loc[missing_type, "RAM_TYPE"] = inferred
    out["ram_type_imputed"] = missing_type & out["RAM_TYPE"].notna()

    # 4. fill an absent SSD capacity from the CPU's typical configuration
    missing_ssd = out["SSD_SIZE"].isna() | (out["SSD_SIZE"].astype(str).str.strip() == "")
    out.loc[missing_ssd, "SSD_SIZE"] = out.loc[missing_ssd, "mapped_cpu_name"].map(
        lambda name: storage_map.get(_strip_frequency(name))
    )

    # 5. numeric GB + ordinal RAM type
    out["RAM_SIZE"] = out["RAM_SIZE"].map(to_gb).replace(0.0, np.nan)
    out["SSD_SIZE"] = out["SSD_SIZE"].map(to_gb)
    out["HDD_SIZE"] = out["HDD_SIZE"].map(to_gb)
    out["RAM_TYPE"] = out["RAM_TYPE"].map(encode_ram_type)

    return out


__all__ = [
    "RAM_TYPE_ORDINAL",
    "REALISTIC_RAM_GB",
    "clean_memory_and_storage",
    "encode_ram_type",
    "is_ram_value",
    "is_storage_value",
    "load_cpu_ddr_map",
    "load_cpu_storage_map",
    "needs_swap",
    "normalize_storage_to_gb",
    "parse_dual_storage",
    "to_gb",
]
