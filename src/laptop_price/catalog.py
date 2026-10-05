"""CPU and GPU lookups, so a caller can name a part instead of quoting benchmarks.

The model leans hard on features a person cannot type: the CPU family and
generation, the integrated GPU's scores, and above all ``estimated_component_cost``,
the rule-based build price computed during cleaning. Without them the median
error on a hand-filled request roughly doubles. Naming the CPU (and the GPU, if
there is a dedicated one) is enough to recover all of them.

Lookups are built from ``pre_processed_data.csv`` - the same rows the model was
trained on - so every part offered is one that actually appears in the market.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from laptop_price import paths
from laptop_price.cleaning.memory import RAM_TYPE_ORDINAL
from laptop_price.cleaning.price import estimate_component_cost
from laptop_price.data import load_price_maps

CPU_FIELDS = ("cpu_mark", "tdp", "cores", "cpu_manufacturer", "cpu_family", "cpu_generation_normalized")
GPU_FIELDS = ("gpu_g3d_mark", "gpu_g2d_mark", "gpu_tdp")

# DDR ordinal back to a name the component-cost price table understands.
_RAM_TYPE_NAMES = {value: name for name, value in reversed(list(RAM_TYPE_ORDINAL.items()))}


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def _mode(series: pd.Series) -> Any:
    values = series.dropna()
    return values.mode().iloc[0] if len(values) else None


@lru_cache(maxsize=1)
def cpu_table() -> pd.DataFrame:
    """One row per CPU seen in the listings, most common first."""
    df = pd.read_csv(paths.PRE_PROCESSED)
    df = df[df["mapped_cpu_name"].notna()].copy()
    for column in ("cpu_mark", "tdp", "cores", "cpu_generation_normalized", *GPU_FIELDS):
        df[column] = _to_float(df[column])

    integrated = df[df["DEDICATED_GPU"].isna()]
    grouped = df.groupby("mapped_cpu_name")
    table = pd.DataFrame(
        {
            "listings": grouped.size(),
            "cpu_mark": grouped["cpu_mark"].median(),
            "tdp": grouped["tdp"].median(),
            "cores": grouped["cores"].median(),
            "cpu_generation_normalized": grouped["cpu_generation_normalized"].median(),
            "cpu_manufacturer": grouped["cpu_manufacturer"].agg(_mode),
            "cpu_family": grouped["cpu_family"].agg(_mode),
        }
    )
    # The integrated GPU this CPU shipped with, used when there is no dedicated one.
    igpu = integrated.groupby("mapped_cpu_name")
    table["igpu_name"] = igpu["gpu_name"].agg(_mode)
    for column in GPU_FIELDS:
        table[f"igpu_{column}"] = igpu[column].median()
    return table.sort_values("listings", ascending=False)


@lru_cache(maxsize=1)
def gpu_table() -> pd.DataFrame:
    """One row per dedicated GPU seen in the listings, most common first."""
    df = pd.read_csv(paths.PRE_PROCESSED)
    df = df[df["DEDICATED_GPU"].notna() & df["gpu_name"].notna()].copy()
    for column in GPU_FIELDS:
        df[column] = _to_float(df[column])
    grouped = df.groupby("gpu_name")
    table = pd.DataFrame({"listings": grouped.size(), **{c: grouped[c].median() for c in GPU_FIELDS}})
    return table.sort_values("listings", ascending=False)


@lru_cache(maxsize=1)
def _price_maps() -> tuple[dict[str, float], dict[str, float]]:
    return load_price_maps()


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or value == ""


def _clean(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def enrich(listing: dict[str, Any]) -> dict[str, Any]:
    """Fill benchmark, family and cost columns from ``cpu_name`` / ``gpu_name``.

    Values the caller supplied explicitly always win. Unknown part names are
    ignored rather than rejected: a request is still valid without them.
    """
    out = dict(listing)
    cpu_name, gpu_name = out.pop("cpu_name", None), out.pop("gpu_name", None)
    cpus, gpus = cpu_table(), gpu_table()

    cpu = cpus.loc[cpu_name] if cpu_name in cpus.index else None
    gpu = gpus.loc[gpu_name] if gpu_name in gpus.index else None

    # Whether the processor is identified (by name, or by a caller that already
    # supplies the family) - the single biggest driver of how precise we can be.
    out["_cpu_known"] = cpu is not None or str(out.get("cpu_family") or "UNKNOWN") != "UNKNOWN"
    if cpu is not None:
        for column in CPU_FIELDS:
            if _blank(out.get(column)):
                out[column] = _clean(cpu[column])
    if gpu is not None:
        for column in GPU_FIELDS:
            if _blank(out.get(column)):
                out[column] = _clean(gpu[column])
    elif cpu is not None and _blank(out.get("gpu_g3d_mark")):
        # No dedicated GPU: training rows carry the integrated GPU's scores.
        for column in GPU_FIELDS:
            out[column] = _clean(cpu[f"igpu_{column}"])
        out["_integrated_gpu"] = True

    if _blank(out.get("estimated_component_cost")):
        cpu_prices, gpu_prices = _price_maps()
        ram_type = out.get("RAM_TYPE")
        out["estimated_component_cost"] = estimate_component_cost(
            {
                "mapped_cpu_name": cpu_name if cpu is not None else None,
                "gpu_name": gpu_name if gpu is not None else (cpu["igpu_name"] if cpu is not None else None),
                "RAM_SIZE": out.get("RAM_SIZE"),
                "RAM_TYPE": None if _blank(ram_type) else _RAM_TYPE_NAMES.get(float(ram_type)),
                "SSD_SIZE": out.get("SSD_SIZE"),
                "HDD_SIZE": out.get("HDD_SIZE"),
                "model_name": out.get("brand"),
            },
            cpu_prices,
            gpu_prices,
        )
    return out


def catalog(limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """The parts a UI can offer, most common first."""
    cpus, gpus = cpu_table(), gpu_table()
    if limit:
        cpus, gpus = cpus.head(limit), gpus.head(limit)
    return {
        "cpus": [
            {
                "name": name,
                "listings": int(row["listings"]),
                "cpu_mark": _clean(row["cpu_mark"]),
                "cores": _clean(row["cores"]),
                "family": _clean(row["cpu_family"]),
                "igpu": _clean(row["igpu_name"]),
            }
            for name, row in cpus.iterrows()
        ],
        "gpus": [
            {"name": name, "listings": int(row["listings"]), "g3d_mark": _clean(row["gpu_g3d_mark"])}
            for name, row in gpus.iterrows()
        ],
    }


__all__ = ["catalog", "cpu_table", "enrich", "gpu_table"]
