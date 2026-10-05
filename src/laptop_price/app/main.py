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
    st.markdown("## 💡 Get Your Laptop Valued")
    st.markdown(
        "Enter the specs you know — leave the rest blank. "
        "The model learned a split for *not stated*, so a blank is more informative than a guess."
    )
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("**Core specs**")
        cpu_mark = st.number_input("CPU PassMark score", 0, 80_000, 8_000, step=500)
        ram = st.select_slider("RAM (GB)", [4, 6, 8, 12, 16, 24, 32, 64], value=8)
        ssd = st.select_slider("SSD (GB)", [0, 128, 256, 512, 1024, 2048], value=256)
    with right:
        st.markdown("**Identity**")
        brand = st.selectbox("Brand / model family", _levels("brand"))
        condition = st.selectbox("Condition", list(CONDITIONS), index=0)
        city = st.selectbox("City", _levels("city_grouped"))

    with st.expander("Advanced options (GPU, HDD, screen, year…)"):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            gpu_mark = st.number_input("GPU G3D score (0 = integrated)", 0, 60_000, 0, step=500)
            gpu_tdp = st.number_input("GPU TDP (W, 0 = unknown)", 0, 250, 0, step=5)
            hdd = st.select_slider("HDD (GB)", [0, 320, 500, 1000, 2000], value=0)
        with adv2:
            cores = st.number_input("CPU cores", 1, 64, 4)
            ram_type = st.selectbox("Memory type", list(RAM_TYPES), index=0)
            year = st.number_input("Listing year", 2018, 2030, 2025)
        with adv3:
            screen = st.select_slider(
                "Screen (inches)", [11.6, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3], value=15.6
            )
            resolution = st.selectbox("Resolution", list(RESOLUTION_ORDINAL), index=2)

    if not st.button("Estimate price", type="primary", use_container_width=True):
        return

    listing = {
        "RAM_SIZE": float(ram),
        "SSD_SIZE": float(ssd),
        "HDD_SIZE": float(hdd),
        "RAM_TYPE": RAM_TYPES[ram_type],
        "cpu_mark": float(cpu_mark) or None,
        "gpu_g3d_mark": float(gpu_mark),
        "gpu_tdp": float(gpu_tdp) if gpu_tdp else None,
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
    mid = result["estimate_dzd"]

    st.markdown("---")
    st.metric(
        "Estimated Price Range",
        f"{low:,.0f} – {high:,.0f} DZD",
        delta=f"Most likely ~{mid:,.0f} DZD",
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Point estimate", f"{mid:,.0f} DZD")
    col_b.metric("Model version", result["model_version"])
    st.info(result["caveat"])

    st.subheader("Why this number")
    try:
        import plotly.graph_objects as go

        contributions = explain_one(listing, top_n=10)
        frame = pd.DataFrame(contributions)
        effects_pct = np.expm1(frame["contribution"]) * 100

        fig = go.Figure(
            go.Waterfall(
                name="Price factors",
                orientation="h",
                measure=["relative"] * len(frame),
                y=frame["feature"].tolist(),
                x=effects_pct.tolist(),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#22c55e"}},
                decreasing={"marker": {"color": "#ef4444"}},
            )
        )
        fig.update_layout(
            title="Feature contributions (% effect on price)",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
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
    st.markdown("## 🔍 Find Underpriced Listings")
    st.caption(
        "Scam-shaped listings are filtered out first: a deep discount on an "
        "otherwise anomalous spec/price pair is more likely bait than a bargain."
    )

    flt1, flt2, flt3 = st.columns(3)
    with flt1:
        brand_filter = st.multiselect("Brand", options=_levels("brand")[1:])  # skip UNKNOWN
    with flt2:
        price_cap = st.slider("Max price (DZD)", 10_000, 500_000, 200_000, step=5_000)
    with flt3:
        sort_by = st.selectbox("Sort by", ["Discount %", "Absolute savings", "Price"])

    limit = st.slider("How many results", 5, 100, 25)
    if not st.button("Find deals", type="primary"):
        return

    from laptop_price.anomaly.deals import rank_deals

    with st.spinner("Scoring every listing…"):
        ranked = rank_deals(_matrix(), _bundle(), top_k=200)

    if brand_filter:
        ranked = ranked[ranked["brand"].isin(brand_filter)]
    ranked = ranked[ranked[TARGET] <= price_cap]

    _sort_map = {
        "Discount %": "discount_pct",
        "Absolute savings": "savings_dzd",
        "Price": TARGET,
    }
    sort_col = _sort_map[sort_by]
    if sort_col in ranked.columns:
        ranked = ranked.sort_values(sort_col, ascending=(sort_by == "Price"))

    ranked = ranked.head(limit)
    st.caption(f"Showing {len(ranked)} deal(s) after filters.")

    cols_per_row = 3
    for row_start in range(0, len(ranked), cols_per_row):
        card_cols = st.columns(cols_per_row)
        for col_idx, (_, row) in zip(
            range(cols_per_row), ranked.iloc[row_start : row_start + cols_per_row].iterrows()
        ):
            with card_cols[col_idx]:
                brand = row.get("brand", "Unknown")
                price = row.get(TARGET, 0)
                predicted = row.get("predicted", 0)
                discount = row.get("discount_pct", 0)
                ram = row.get("RAM_SIZE", "?")
                ssd = row.get("SSD_SIZE", "?")
                cpu = row.get("cpu_mark", "?")
                verdict = row.get("verdict", "")
                with st.container(border=True):
                    st.markdown(f"**{brand}**")
                    st.metric(
                        "Listed price",
                        f"{price:,.0f} DZD",
                        delta=f"-{discount:.0f}% vs model" if discount else None,
                        delta_color="normal",
                    )
                    st.caption(
                        f"RAM {ram}GB · SSD {ssd}GB · CPU {cpu:,.0f}" if isinstance(cpu, float) else f"RAM {ram}GB · SSD {ssd}GB"
                    )
                    if verdict:
                        st.caption(f"_{verdict}_")


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def render_market() -> None:
    st.markdown("## 📊 Market Overview")
    matrix = _matrix()

    a, b, c = st.columns(3)
    a.metric("Listings", f"{len(matrix):,}")
    b.metric("Unique configurations", f"{matrix['spec_signature'].nunique():,}")
    c.metric("Median price", f"{matrix[TARGET].median():,.0f} DZD")

    st.markdown("---")

    # --- brand market share donut ------------------------------------------
    try:
        import plotly.express as px

        brand_counts = matrix["brand"].value_counts().head(15)
        fig_donut = px.pie(
            values=brand_counts.values,
            names=brand_counts.index,
            hole=0.45,
            title="Top 15 Brands by Listing Count",
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)
    except Exception:
        brand_counts = matrix["brand"].value_counts().head(15)
        st.bar_chart(brand_counts)

    # --- price vs CPU performance scatter ----------------------------------
    try:
        scatter_df = matrix.dropna(subset=["cpu_mark", TARGET]).sample(
            min(3000, len(matrix)), random_state=42
        )
        fig_scatter = px.scatter(
            scatter_df,
            x="cpu_mark",
            y=TARGET,
            color="brand",
            size="RAM_SIZE",
            size_max=12,
            hover_data=["SSD_SIZE", "city_grouped"],
            title="Price vs CPU Performance",
            labels={"cpu_mark": "CPU PassMark", TARGET: "Price (DZD)"},
            opacity=0.6,
        )
        fig_scatter.update_layout(
            showlegend=False, margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    except Exception as exc:
        st.warning(f"Scatter unavailable: {exc}")

    # --- brand × condition median price heatmap ----------------------------
    try:
        cond_map = {1.0: "MOYEN", 2.0: "BON ÉTAT", 3.0: "JAMAIS UTILISÉ"}
        hm_df = matrix.copy()
        hm_df["condition_label"] = hm_df["spec_Etat"].map(cond_map).fillna("Unknown")
        top_brands = matrix["brand"].value_counts().head(12).index
        hm_df = hm_df[hm_df["brand"].isin(top_brands)]
        pivot = hm_df.pivot_table(
            values=TARGET, index="brand", columns="condition_label", aggfunc="median"
        )
        fig_heat = px.imshow(
            pivot,
            title="Median Price (DZD): Brand × Condition",
            labels=dict(color="Median price"),
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".0f",
        )
        fig_heat.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as exc:
        st.warning(f"Heatmap unavailable: {exc}")

    # --- price trend and city breakdown ------------------------------------
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
    if len(by_city) >= 2:
        st.caption(
            f"{by_city.index[0]} to {by_city.index[-1]}: "
            f"a {by_city['median'].iloc[0] / by_city['median'].iloc[-1]:.2f}x spread."
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
