"""Write the static JSON the Next.js front end (web/) reads.

Everything that does not depend on a visitor's input is computed here once -
the parts catalog, the market charts, and the deals feed - so the site renders
instantly and keeps working when the prediction API is asleep. Only /predict
and /similar are called live.

    python scripts/export_web_data.py            # writes web/public/data/*.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from laptop_price.catalog import catalog
from laptop_price.cleaning.price import correct_prices
from laptop_price.data import load_model_ready, load_preprocessed, load_price_maps
from laptop_price.features.build import TARGET
from laptop_price.models.registry import load_bundle

OUT = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
CONDITION = {1.0: "Fair", 2.0: "Good", 3.0: "Never used"}


def _r(value: float, digits: int = -2) -> float | None:
    return None if value is None or not np.isfinite(value) else float(round(float(value), digits))


def _write(name: str, payload: object) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path.relative_to(OUT.parents[2])} ({path.stat().st_size / 1024:.0f} KB)")


def _part_names(matrix: pd.DataFrame) -> pd.DataFrame:
    """CPU and dedicated-GPU names for each matrix row.

    The feature matrix drops the names; it is ``correct_prices`` on the
    preprocessed rows, minus the rows without a target, in the same order.
    """
    cpu_prices, gpu_prices = load_price_maps()
    priced = correct_prices(load_preprocessed(), cpu_prices, gpu_prices)
    kept = priced[priced[TARGET].notna()].reset_index(drop=True)
    if len(kept) != len(matrix) or not np.allclose(kept[TARGET], matrix[TARGET]):
        raise RuntimeError("preprocessed rows no longer line up with features.csv; rebuild it")
    return pd.DataFrame(
        {
            "cpu_name": kept["mapped_cpu_name"],
            "gpu_name": kept["gpu_name"].where(kept["DEDICATED_GPU"].notna()),
        }
    )


def market(matrix: pd.DataFrame) -> dict:
    price = matrix[TARGET]
    known_brand = matrix[~matrix["brand"].isin(["UNKNOWN", "OTHER"])]

    share = known_brand["brand"].value_counts().head(10)
    brands = [
        {
            "brand": b,
            "share": float(n / len(matrix) * 100),
            "count": int(n),
            "median": _r(known_brand.loc[known_brand["brand"] == b, TARGET].median()),
        }
        for b, n in share.items()
    ]

    by_year = matrix.groupby("listing_year")[TARGET].agg(["median", "count"]).dropna()
    years = [{"year": int(y), "median": _r(r["median"]), "count": int(r["count"])} for y, r in by_year.iterrows()]

    by_city = (
        matrix[~matrix["city_grouped"].isin(["UNKNOWN", "OTHER"])]
        .groupby("city_grouped")[TARGET]
        .agg(["median", "count"])
        .query("count >= 100")
        .sort_values("median", ascending=False)
    )
    cities = [{"city": c, "median": _r(r["median"]), "count": int(r["count"])} for c, r in by_city.iterrows()]

    sample = matrix.dropna(subset=["cpu_mark", TARGET])
    sample = sample[sample[TARGET] < sample[TARGET].quantile(0.99)].sample(1400, random_state=7)
    scatter = [
        [int(r.cpu_mark), int(r[TARGET]), int((r.gpu_g3d_mark or 0) > 4000 if pd.notna(r.gpu_g3d_mark) else 0)]
        for _, r in sample.iterrows()
    ]

    top = share.head(8).index
    cond = matrix.assign(cond=matrix["spec_Etat"].map(CONDITION))
    cond = cond[cond["brand"].isin(top) & cond["cond"].notna()]
    pivot = cond.pivot_table(values=TARGET, index="brand", columns="cond", aggfunc="median").reindex(top)
    heat = {
        "rows": list(pivot.index),
        "cols": [c for c in CONDITION.values() if c in pivot.columns],
        "values": [[_r(pivot.loc[b, c]) for c in CONDITION.values() if c in pivot.columns] for b in pivot.index],
    }

    ram = matrix.dropna(subset=["RAM_SIZE"])
    ram = ram[ram["RAM_SIZE"].isin([4, 8, 12, 16, 32, 64])]
    by_ram = ram.groupby("RAM_SIZE")[TARGET].agg(["median", "count"]).query("count >= 50")
    rams = [{"gb": int(g), "median": _r(r["median"]), "count": int(r["count"])} for g, r in by_ram.iterrows()]

    first, last = years[0], years[-1]
    return {
        "listings": int(len(matrix)),
        "median": _r(price.median()),
        "p25": _r(price.quantile(0.25)),
        "p75": _r(price.quantile(0.75)),
        "brandsTracked": int(matrix["brand"].nunique()),
        "citiesTracked": int(matrix["city_grouped"].nunique()),
        "yearSpan": [first["year"], last["year"]],
        "brands": brands,
        "years": years,
        "cities": cities,
        "scatter": scatter,
        "heat": heat,
        "ram": rams,
    }


def deals(matrix: pd.DataFrame, names: pd.DataFrame) -> list[dict]:
    from laptop_price.anomaly.deals import rank_deals

    bundle = load_bundle()
    ranked = rank_deals(matrix.join(names), bundle, top_k=240)
    out = []
    for _, r in ranked.iterrows():
        out.append(
            {
                "price": _r(r[TARGET]),
                "expected": _r(r["predicted"]),
                "discount": round(float(r["discount_pct"]), 1),
                "brand": None if r["brand"] in ("UNKNOWN", "OTHER") else str(r["brand"]),
                "city": None if r["city_grouped"] in ("UNKNOWN", "OTHER") else str(r["city_grouped"]),
                "ram": _r(r.get("RAM_SIZE"), 0),
                "ssd": _r(r.get("SSD_SIZE"), 0),
                "cpuMark": _r(r.get("cpu_mark"), 0),
                "gpuMark": _r(r.get("gpu_g3d_mark"), 0),
                "condition": CONDITION.get(r.get("spec_Etat")),
                "year": _r(r.get("listing_year"), 0),
                "cpu": r["cpu_name"] if isinstance(r.get("cpu_name"), str) else None,
                "gpu": r["gpu_name"] if isinstance(r.get("gpu_name"), str) else None,
            }
        )
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = load_model_ready()
    names = _part_names(matrix)

    parts = catalog()
    counts = lambda col: [  # noqa: E731
        v for v in matrix[col].dropna().astype(str).value_counts().index if v not in ("UNKNOWN", "OTHER")
    ]
    _write("catalog.json", {**parts, "brands": counts("brand"), "cities": counts("city_grouped")})
    _write("market.json", market(matrix))
    _write("deals.json", deals(matrix, names))


if __name__ == "__main__":
    main()
