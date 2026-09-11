"""«Combien vaut mon PC ?» - the Streamlit front end.

Three surfaces:

* **Estimate** - a spec form returning a *range*, plus a SHAP waterfall showing
  how each feature moved the price away from the market baseline, plus the
  nearest real listings.
* **Deals** - the underpriced-listing feed from ``laptop_price.anomaly.deals``.
* **Market** - a few charts that answer questions someone actually asked.

The app talks to the model directly rather than through the API, so it still
works if the API container is down; ``LAPTOP_PRICE_API_URL`` is only used to show
whether the API is reachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from laptop_price import __version__
from laptop_price.cleaning.screen import RESOLUTION_ORDINAL
from laptop_price.features.build import TARGET

st.set_page_config(
    page_title="Laptop Price Intelligence",
    page_icon="💻",
    layout="wide",
)

CONDITIONS = {
    "Unknown / not stated": None,
    "MOYEN (fair)": 1.0,
    "BON ÉTAT (good)": 2.0,
    "JAMAIS UTILISÉ (never used)": 3.0,
}

RAM_TYPES = {
    "Unknown": None,
    "DDR3": 2.0,
    "DDR4": 3.0,
    "DDR5": 4.0,
    "LPDDR5X": 4.2,
}


@st.cache_resource(show_spinner="Loading model…")
def _bundle():
    from laptop_price.models.registry import load_bundle

    return load_bundle()


@st.cache_data(show_spinner="Loading listings…")
def _matrix() -> pd.DataFrame:
    from laptop_price.data import load_model_ready

    return load_model_ready()


def _levels(column: str, fallback: str = "UNKNOWN") -> list[str]:
    try:
        values = _matrix()[column].dropna().astype(str)
    except Exception:
        return [fallback]
    ordered = values.value_counts().index.tolist()
    return [fallback, *[v for v in ordered if v != fallback]]


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def render_estimate() -> None:
    st.subheader("What is this laptop worth?")
    st.caption(
        "Enter what you know and leave the rest blank — the model learned a split "
        "for “not stated”, so a blank is more informative than a guess."
    )

    left, middle, right = st.columns(3)
    with left:
        ram = st.select_slider("RAM (GB)", [4, 6, 8, 12, 16, 24, 32, 64], value=8)
        ssd = st.select_slider("SSD (GB)", [0, 128, 256, 512, 1024, 2048], value=256)
        hdd = st.select_slider("HDD (GB)", [0, 320, 500, 1000, 2000], value=0)
        ram_type = st.selectbox("Memory type", list(RAM_TYPES), index=0)
    with middle:
        cpu_mark = st.number_input("CPU PassMark score", 0, 80_000, 8_000, step=500)
        gpu_mark = st.number_input("GPU G3D score (0 = integrated)", 0, 60_000, 0, step=500)
        cores = st.number_input("CPU cores", 1, 64, 4)
        screen = st.select_slider(
            "Screen (inches)", [11.6, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3], value=15.6
        )
    with right:
        resolution = st.selectbox("Resolution", list(RESOLUTION_ORDINAL), index=2)
        condition = st.selectbox("Condition", list(CONDITIONS), index=0)
        brand = st.selectbox("Model family", _levels("brand"))
        city = st.selectbox("City", _levels("city_grouped"))
        year = st.number_input("Listing year", 2018, 2030, 2025)

    if not st.button("Estimate price", type="primary", use_container_width=True):
        return

    listing = {
        "RAM_SIZE": float(ram),
        "SSD_SIZE": float(ssd),
        "HDD_SIZE": float(hdd),
        "RAM_TYPE": RAM_TYPES[ram_type],
        "cpu_mark": float(cpu_mark) or None,
        "gpu_g3d_mark": float(gpu_mark),
        "cores": float(cores),
        "SCREEN_SIZE_SNAPPED": float(screen),
        "SCREEN_RESOLUTION_ENC": float(RESOLUTION_ORDINAL[resolution]),
        "spec_Etat": CONDITIONS[condition],
        "brand": brand,
        "city_grouped": city,
        "listing_year": float(year),
    }

    from laptop_price.serving import explain_one, find_similar, predict_one

    result = predict_one(listing)
    low, high = result.get("range_dzd", [result["estimate_dzd"]] * 2)

    st.markdown("---")
    a, b, c = st.columns(3)
    a.metric("Most likely", f"{result['estimate_dzd']:,.0f} DZD")
    b.metric("Fair range", f"{low:,.0f} – {high:,.0f}")
    c.metric("Model", result["model_version"])
    st.info(result["caveat"])

    st.subheader("Why this number")
    try:
        contributions = explain_one(listing, top_n=10)
        frame = pd.DataFrame(contributions).set_index("feature")["contribution"]
        # Contributions are in log space; show them as a percentage effect on price,
        # which is what a reader can actually interpret.
        st.bar_chart((np.expm1(frame) * 100).rename("effect on price (%)"))
        st.caption(
            "Each bar is that feature's contribution relative to the market baseline, "
            "expressed as a percentage effect on the estimate."
        )
    except Exception as exc:
        st.warning(f"Explanation unavailable: {exc}")

    st.subheader("Similar listings in the dataset")
    try:
        similar = find_similar(listing, _matrix(), k=8)
        columns = [
            c
            for c in (
                TARGET,
                "RAM_SIZE",
                "SSD_SIZE",
                "cpu_mark",
                "gpu_g3d_mark",
                "brand",
                "city_grouped",
                "listing_year",
                "distance",
            )
            if c in similar.columns
        ]
        st.dataframe(similar[columns], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Similar listings unavailable: {exc}")


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------


def render_deals() -> None:
    st.subheader("Listings priced below what the model expects")
    st.caption(
        "Scam-shaped listings are filtered out first: a deep discount on an "
        "otherwise anomalous spec/price pair is more likely bait than a bargain."
    )

    limit = st.slider("How many", 5, 100, 25)
    if not st.button("Find deals", type="primary"):
        return

    from laptop_price.anomaly.deals import rank_deals

    with st.spinner("Scoring every listing…"):
        ranked = rank_deals(_matrix(), _bundle(), top_k=limit)

    columns = [
        c
        for c in (
            TARGET,
            "predicted",
            "discount_pct",
            "brand",
            "city_grouped",
            "RAM_SIZE",
            "SSD_SIZE",
            "cpu_mark",
            "verdict",
        )
        if c in ranked.columns
    ]
    st.dataframe(
        ranked[columns].style.format(
            {TARGET: "{:,.0f}", "predicted": "{:,.0f}", "discount_pct": "{:.0f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def render_market() -> None:
    st.subheader("The market")
    matrix = _matrix()

    a, b, c = st.columns(3)
    a.metric("Listings", f"{len(matrix):,}")
    b.metric("Unique configurations", f"{matrix['spec_signature'].nunique():,}")
    c.metric("Median price", f"{matrix[TARGET].median():,.0f} DZD")

    st.markdown("#### Price by listing year")
    by_year = matrix.groupby("listing_year")[TARGET].median().dropna()
    st.line_chart(by_year.rename("median price (DZD)"))
    st.caption(
        "Seven years of dinar inflation and tech depreciation sit in this chart. "
        "It is why the headline metric uses a time-based split, not a random one."
    )

    st.markdown("#### Median price by city")
    by_city = (
        matrix.groupby("city_grouped")[TARGET]
        .agg(["median", "count"])
        .query("count >= 100")
        .sort_values("median", ascending=False)
    )
    st.bar_chart(by_city["median"].rename("median price (DZD)"))
    st.caption(
        f"{by_city.index[0]} to {by_city.index[-1]}: a {by_city['median'].iloc[0] / by_city['median'].iloc[-1]:.2f}x spread."
    )

    st.markdown("#### How much do identical configurations vary?")
    spread = (
        matrix.groupby("spec_signature")[TARGET]
        .agg(["min", "max", "count"])
        .query("count >= 5")
        .assign(ratio=lambda d: d["max"] / d["min"])
    )
    st.metric(
        "Median max/min ratio for repeated configurations", f"{spread['ratio'].median():.2f}x"
    )
    st.caption(
        "This is the noise ceiling. The remaining error is mostly not in the spec "
        "columns at all — it is in listing text, seller context and negotiability, "
        "none of which were scraped."
    )


def main() -> None:
    st.title("💻 Laptop Price Intelligence — Algeria")

    try:
        bundle = _bundle()
        st.sidebar.success(f"Model {bundle.version}")
        headline = bundle.metadata.get("metrics", {}).get("headline_time_based", {})
        if headline:
            st.sidebar.caption(
                f"Time-based test: R² {headline['R2']:.3f}, "
                f"typical error {headline['MedAPE']:.1f}%"
            )
    except FileNotFoundError:
        st.error("No trained model found. Run `make train` first.")
        st.stop()

    st.sidebar.caption(f"package v{__version__}")
    estimate, deals, market = st.tabs(["Estimate", "Deals", "Market"])
    with estimate:
        render_estimate()
    with deals:
        render_deals()
    with market:
        render_market()


main()
