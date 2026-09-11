"""CPU name normalisation and benchmark lookup.

Listings write CPU names however the seller felt like it: ``"i5 11eme gen"``,
``"INTEL CORE I7 11800H"``, ``"ryzen 7 surface edition"``, plus OCR damage. This
module maps that free text onto a row of ``data/raw/cpus.csv`` so the PassMark
score, core count and TDP can be attached.

Scope
-----
The heavy, hand-curated enrichment that produced ``pre_processed_data.csv`` lives
in ``notebooks/01_preprocessing.ipynb``; its 126-entry correction table has been
lifted out to ``data/mappings/cpu_corrections.csv`` so it is versionable data
rather than a wall of literals inside a cell. What lives here is the part that
*serving* needs: turning a user-typed CPU string into a benchmark row at
inference time, using the same normalisation the training data was built with.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from laptop_price import paths

#: Below this fuzzy score a match is not trustworthy and we return nothing
#: rather than attach the wrong benchmark.
MATCH_THRESHOLD = 70

CPU_CORRECTIONS_CSV = paths.MAPPINGS_DIR / "cpu_corrections.csv"


def normalize(value: Any) -> str:
    """Strip vendor noise so ``"Intel Core i7-11800H"`` and ``"i7 11800h"`` agree.

    The clock-speed suffix is dropped first. PassMark names carry it
    (``"Intel Core i5-1135G7 @ 2.40GHz"``) and listings almost never do, and
    leaving it in breaks matching twice over: an exact lookup on a bare model
    number misses, and the extra ``2`` / ``40ghz`` tokens dilute
    ``token_set_ratio`` enough to push a correct match below the threshold.
    ``"11TH GEN INTEL CORE I5 1135G7"`` scored 69.2 against its own reference
    row before this - just under the cut - and silently matched an i5-1235U.

    >>> normalize("Intel Core i5-1135G7 @ 2.40GHz")
    'i5 1135g7'
    >>> normalize("11TH GEN INTEL CORE I5 1135G7")
    '11th gen i5 1135g7'
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower()
    text = re.split(r"\s*@\s*", text)[0]
    text = re.sub(r"intel|processor|core|cpu", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_corrections(path: str | None = None) -> dict[str, str]:
    """Load the normalised-query -> canonical-name typo table."""
    target = Path(path) if path else CPU_CORRECTIONS_CSV
    if not target.exists():
        return {}
    frame = pd.read_csv(target)
    return dict(zip(frame["normalized_query"], frame["canonical_cpu_name"], strict=True))


def apply_corrections(normalized: str, corrections: dict[str, str] | None = None) -> str | None:
    """Return the canonical CPU name for a known typo, else None."""
    table = load_corrections() if corrections is None else corrections
    return table.get(normalized)


@lru_cache(maxsize=1)
def load_cpu_reference(path: str | None = None) -> pd.DataFrame:
    """The PassMark reference table with a ``normalized`` join key added."""
    frame = pd.read_csv(path or paths.CPUS_CSV, on_bad_lines="warn")
    frame["normalized"] = frame["name"].map(normalize)
    return frame


def extract_manufacturer(cpu_name: Any) -> str:
    """INTEL / AMD / APPLE / OTHER."""
    text = str(cpu_name).upper()
    if "APPLE" in text or re.search(r"\bM[1-4]\b", text):
        return "APPLE"
    if "AMD" in text or "RYZEN" in text or "ATHLON" in text:
        return "AMD"
    if "INTEL" in text or re.search(r"\bI[3579]\b|CELERON|PENTIUM|XEON", text):
        return "INTEL"
    return "OTHER"


def extract_family(cpu_name: Any) -> str:
    """Coarse product family: i3/i5/i7/i9, Ryzen 3/5/7/9, M-series, Celeron, ..."""
    text = str(cpu_name).upper()
    for pattern, family in (
        (r"\bI3\b", "i3"),
        (r"\bI5\b", "i5"),
        (r"\bI7\b", "i7"),
        (r"\bI9\b", "i9"),
        (r"RYZEN\s*3\b", "Ryzen 3"),
        (r"RYZEN\s*5\b", "Ryzen 5"),
        (r"RYZEN\s*7\b", "Ryzen 7"),
        (r"RYZEN\s*9\b", "Ryzen 9"),
        (r"\bM[1-4]\b", "Apple M"),
        (r"CELERON", "Celeron"),
        (r"PENTIUM", "Pentium"),
        (r"XEON", "Xeon"),
        (r"ATHLON", "Athlon"),
        (r"ULTRA", "Core Ultra"),
    ):
        if re.search(pattern, text):
            return family
    return "UNKNOWN"


def match_cpu(
    query: Any,
    reference: pd.DataFrame | None = None,
    *,
    threshold: int = MATCH_THRESHOLD,
) -> dict[str, Any] | None:
    """Resolve a free-text CPU name to a benchmark row.

    Tries, in order: the typo table, an exact normalised match, then a fuzzy
    ``token_set_ratio`` match. Returns None below ``threshold`` rather than
    attaching a benchmark that belongs to a different chip.

    Returns
    -------
    dict or None
        ``name``, ``cpumark``, ``tdp``, ``cores``, ``match_score``, ``match_type``.
    """
    from rapidfuzz import fuzz, process

    table = load_cpu_reference() if reference is None else reference
    normalized = normalize(query)
    if not normalized:
        return None

    def _row(row: pd.Series, score: float, kind: str) -> dict[str, Any]:
        return {
            "name": row["name"],
            "cpumark": row.get("cpumark"),
            "tdp": row.get("tdp"),
            "cores": row.get("cores"),
            "match_score": float(score),
            "match_type": kind,
        }

    # 1. known typo -> canonical name
    corrected = apply_corrections(normalized)
    if corrected is not None:
        hit = table[table["normalized"] == normalize(corrected)]
        if not hit.empty:
            return _row(hit.iloc[0], 100.0, "correction")

    # 2. exact normalised match
    hit = table[table["normalized"] == normalized]
    if not hit.empty:
        return _row(hit.iloc[0], 100.0, "exact")

    # 3. fuzzy
    choices = table["normalized"].tolist()
    match = process.extractOne(normalized, choices, scorer=fuzz.token_set_ratio)
    if match is None or match[1] < threshold:
        return None
    return _row(table.iloc[match[2]], match[1], "fuzzy")


def attach_cpu_benchmarks(
    df: pd.DataFrame,
    *,
    column: str = "CPU",
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``mapped_cpu_name``, ``cpu_mark``, ``cores``, ``tdp`` and match provenance."""
    table = load_cpu_reference() if reference is None else reference
    out = df.copy()

    # Resolve each distinct string once; the dataset has ~16k rows but far fewer
    # distinct CPU strings, and fuzzy matching dominates the runtime.
    unique = out[column].dropna().unique()
    resolved = {value: match_cpu(value, table) for value in unique}

    out["mapped_cpu_name"] = out[column].map(
        lambda v: resolved.get(v, {}).get("name") if resolved.get(v) else None
    )
    out["cpu_mark"] = out[column].map(
        lambda v: resolved.get(v, {}).get("cpumark") if resolved.get(v) else None
    )
    out["cores"] = out[column].map(
        lambda v: resolved.get(v, {}).get("cores") if resolved.get(v) else None
    )
    out["tdp"] = out[column].map(
        lambda v: resolved.get(v, {}).get("tdp") if resolved.get(v) else None
    )
    out["match_score"] = out[column].map(
        lambda v: resolved.get(v, {}).get("match_score") if resolved.get(v) else None
    )
    out["cpu_manufacturer"] = out["mapped_cpu_name"].map(extract_manufacturer)
    out["cpu_family"] = out["mapped_cpu_name"].map(extract_family)
    return out


__all__ = [
    "MATCH_THRESHOLD",
    "apply_corrections",
    "attach_cpu_benchmarks",
    "extract_family",
    "extract_manufacturer",
    "load_corrections",
    "load_cpu_reference",
    "match_cpu",
    "normalize",
]
