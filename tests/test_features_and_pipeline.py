"""Feature construction, evaluation splits, and the end-to-end pipeline artifact.

The last test here is the one that matters most: fit -> save -> load -> predict.
The previous release shipped a scaler expecting 14 features next to a model
expecting 10, and nothing caught it. A round-trip test does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laptop_price.evaluation.baselines import baseline_table, spec_median_baseline
from laptop_price.evaluation.metrics import interval_coverage, regression_metrics
from laptop_price.evaluation.splits import grouped_split, stratified_split, time_based_split
from laptop_price.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_matrix,
    encode_etat,
    group_rare_levels,
    parse_created_at,
    spec_signature,
)


@pytest.fixture
def listings() -> pd.DataFrame:
    """A small synthetic frame shaped like ``pre_processed_data.csv``."""
    rng = np.random.default_rng(0)
    n = 240
    return pd.DataFrame(
        {
            "price_corrected": rng.uniform(20_000, 400_000, n).round(-3),
            "created_at": [
                f"202{rng.integers(1, 6)} 0{rng.integers(1, 10)} 15T10:00:00.000Z" for _ in range(n)
            ],
            "city": rng.choice(["EZZOUAR", "ORAN", "TINY_TOWN"], n, p=[0.6, 0.35, 0.05]),
            "spec_Etat": rng.choice(["BON TAT", "MOYEN", None], n),
            "model_name": rng.choice(["THINKPAD", "MACBOOK", "RAREBOOK"], n, p=[0.5, 0.45, 0.05]),
            "DEDICATED_GPU": rng.choice([None, "NVIDIA RTX 3060"], n),
            "RAM_SIZE": rng.choice(["8GB", "16GB", "32GB"], n),
            "SSD_SIZE": rng.choice(["256GB", "512GB", "1TB"], n),
            "HDD_SIZE": rng.choice([None, "1TB"], n),
            "RAM_TYPE": rng.choice(["DDR4", "DDR5", None], n),
            "cores": rng.integers(2, 16, n).astype(float),
            "cpu_mark": [f"{v:,}" for v in rng.integers(3_000, 30_000, n)],
            "tdp": rng.choice([15.0, 28.0, 45.0], n),
            "gpu_g3d_mark": [f"{v:,}" for v in rng.integers(500, 20_000, n)],
            "gpu_g2d_mark": rng.integers(200, 1_200, n).astype(float),
            "gpu_tdp": rng.choice([0.0, 45.0, 115.0], n),
            "cpu_manufacturer": rng.choice(["INTEL", "AMD"], n),
            "cpu_family": rng.choice(["i5", "i7", "Ryzen 5"], n),
            "cpu_generation_normalized": rng.uniform(0, 1, n),
            "SCREEN_SIZE_SNAPPED": rng.choice([13.3, 14.0, 15.6], n),
            "SCREEN_RESOLUTION_STD": rng.choice(["HD", "FHD", "QHD"], n),
            "SCREEN_RESOLUTION_ENC": rng.choice([1.0, 3.0, 5.0], n),
            "estimated_component_cost": rng.uniform(20_000, 300_000, n),
            "price_unit_ambiguous": False,
        }
    )


class TestConditionEncoding:
    def test_missing_condition_is_nan_never_zero(self):
        """Roadmap A2: 41% of rows are missing and sit mid-scale, not at the bottom."""
        assert np.isnan(encode_etat(None))
        assert np.isnan(encode_etat(float("nan")))

    def test_known_conditions_are_ordered(self):
        assert encode_etat("MOYEN") < encode_etat("BON TAT") < encode_etat("JAMAIS UTILIS")

    def test_accented_and_unaccented_spellings_agree(self):
        assert encode_etat("BON ÉTAT") == encode_etat("BON TAT")


class TestRestoredColumns:
    def test_created_at_with_space_separators_parses(self):
        """The scraper emitted '2021 10 01T…' rather than ISO hyphens."""
        parsed = parse_created_at(pd.Series(["2021 10 01T18:01:44.000Z"]))
        assert parsed.dt.year.iloc[0] == 2021
        assert parsed.dt.month.iloc[0] == 10

    def test_unparseable_dates_become_nat_not_an_exception(self):
        assert parse_created_at(pd.Series(["not a date"])).isna().all()

    def test_rare_levels_collapse(self):
        series = pd.Series(["A"] * 50 + ["B"] * 40 + ["C"] * 2)
        grouped = group_rare_levels(series, min_count=10)
        assert set(grouped.unique()) == {"A", "B", "OTHER"}

    def test_missing_levels_become_other(self):
        grouped = group_rare_levels(pd.Series(["A"] * 20 + [None]), min_count=5)
        assert grouped.iloc[-1] == "OTHER"


class TestFeatureMatrix:
    def test_declared_columns_are_all_present(self, listings):
        matrix = build_feature_matrix(listings)
        for column in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET):
            assert column in matrix.columns

    def test_benchmark_thousands_separators_survive(self, listings):
        """'19,108' must not silently become NaN."""
        matrix = build_feature_matrix(listings)
        assert matrix["cpu_mark"].notna().all()
        assert matrix["cpu_mark"].max() > 1_000

    def test_capacities_become_numeric_gb(self, listings):
        matrix = build_feature_matrix(listings)
        assert matrix["RAM_SIZE"].between(1, 128).all()
        assert matrix["SSD_SIZE"].max() >= 1000

    def test_implausible_ram_is_nulled(self, listings):
        listings.loc[0, "RAM_SIZE"] = "512GB"
        listings.loc[1, "RAM_SIZE"] = "128MB"
        matrix = build_feature_matrix(listings)
        assert matrix["RAM_SIZE"].dropna().between(1, 128).all()

    def test_etat_missing_flag_matches_the_nulls(self, listings):
        matrix = build_feature_matrix(listings)
        assert (matrix["spec_Etat"].isna() == matrix["etat_is_missing"].astype(bool)).all()

    def test_temporal_features_are_derived(self, listings):
        matrix = build_feature_matrix(listings)
        assert matrix["listing_year"].between(2018, 2030).all()
        assert matrix["month_sin"].between(-1, 1).all()

    def test_use_case_flags_are_binary(self, listings):
        """is_gaming, is_ultrabook, is_back_to_school, is_ramadan must be 0 or 1."""
        matrix = build_feature_matrix(listings)
        for col in ("is_gaming", "is_ultrabook", "is_back_to_school", "is_ramadan"):
            assert set(matrix[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    def test_gaming_flag_requires_discrete_gpu_and_high_tdp(self, listings):
        """is_gaming = 1 only when has_dedicated_gpu=1 AND gpu_tdp > 45."""
        matrix = build_feature_matrix(listings)
        gaming = matrix[matrix["is_gaming"] == 1]
        assert (gaming["has_dedicated_gpu"] == 1).all()
        assert (gaming["gpu_tdp"] > 45).all()

    def test_ultrabook_flag_excludes_discrete_gpu(self, listings):
        """is_ultrabook must never co-occur with has_dedicated_gpu=1."""
        matrix = build_feature_matrix(listings)
        assert (matrix.loc[matrix["is_ultrabook"] == 1, "has_dedicated_gpu"] == 0).all()

    def test_spec_signature_groups_identical_configurations(self):
        frame = pd.DataFrame({"RAM_SIZE": [8.0, 8.0, 16.0], "SSD_SIZE": [256.0, 256.0, 512.0]})
        signature = spec_signature(frame, ("RAM_SIZE", "SSD_SIZE"))
        assert signature.iloc[0] == signature.iloc[1] != signature.iloc[2]


class TestSplits:
    def test_stratified_split_partitions_every_row(self, listings):
        matrix = build_feature_matrix(listings)
        split = stratified_split(matrix[TARGET])
        total = sum(split.sizes().values())
        assert total == len(matrix)
        assert not set(split.train_idx) & set(split.test_idx)

    def test_grouped_split_keeps_a_configuration_on_one_side(self, listings):
        matrix = build_feature_matrix(listings)
        split = grouped_split(matrix["spec_signature"])
        train_groups = set(matrix["spec_signature"].iloc[split.train_idx])
        test_groups = set(matrix["spec_signature"].iloc[split.test_idx])
        assert not train_groups & test_groups

    def test_time_split_never_trains_on_the_future(self, listings):
        matrix = build_feature_matrix(listings)
        split = time_based_split(matrix["listing_year"], train_max_year=2023)
        assert matrix["listing_year"].iloc[split.train_idx].max() <= 2023
        assert matrix["listing_year"].iloc[split.test_idx].min() > 2023


class TestMetrics:
    def test_perfect_prediction(self):
        y = pd.Series([100.0, 200.0, 300.0])
        metrics = regression_metrics(y, y)
        assert metrics["R2"] == pytest.approx(1.0)
        assert metrics["MedAPE"] == pytest.approx(0.0)

    def test_median_ape_is_reported(self):
        metrics = regression_metrics([100.0, 100.0], [110.0, 90.0])
        assert metrics["MedAPE"] == pytest.approx(10.0)
        assert metrics["within_20pct"] == pytest.approx(100.0)

    def test_non_finite_pairs_are_dropped(self):
        metrics = regression_metrics([100.0, np.nan, 300.0], [100.0, 5.0, 300.0])
        assert metrics["n"] == 2

    def test_all_non_finite_raises(self):
        with pytest.raises(ValueError):
            regression_metrics([np.nan], [np.nan])

    def test_interval_coverage(self):
        stats = interval_coverage(
            [100.0, 200.0, 300.0], [90.0, 250.0, 280.0], [110.0, 260.0, 320.0]
        )
        assert stats["coverage_pct"] == pytest.approx(200 / 3, abs=0.1)


class TestBaselines:
    def test_spec_median_uses_the_group(self):
        predicted = spec_median_baseline(
            pd.Series([100.0, 200.0, 1000.0]),
            pd.Series(["a", "a", "b"]),
            pd.Series(["a", "b"]),
        )
        assert predicted.tolist() == [150.0, 1000.0]

    def test_unseen_group_falls_back_to_the_global_median(self):
        predicted = spec_median_baseline(
            pd.Series([100.0, 200.0]), pd.Series(["a", "a"]), pd.Series(["zzz"])
        )
        assert predicted.tolist() == [150.0]

    def test_component_baseline_is_bounded(self, listings):
        """An unbounded heuristic drove R2 to about -48,000 before clipping."""
        table = baseline_table(
            pd.Series([50_000.0] * 10),
            pd.Series([60_000.0] * 10),
            component_cost_test=pd.Series([1e12] * 10),
        )
        row = table[table["model"].str.contains("component")].iloc[0]
        assert np.isfinite(row["MAE"])


class TestPipelineRoundTrip:
    """fit -> save -> load -> predict, which is what the old artifacts failed."""

    def test_artifact_round_trip(self, listings, tmp_path):
        from laptop_price.models.pipeline import build_pipeline
        from laptop_price.models.registry import (
            ArtifactBundle,
            build_metadata,
            latest_version,
            load_bundle,
            save_bundle,
        )

        matrix = build_feature_matrix(listings)
        features = matrix[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]

        pipeline = build_pipeline()
        pipeline.fit(features, matrix[TARGET])
        before = pipeline.predict(features.head(5))

        bundle = ArtifactBundle(
            pipeline=pipeline,
            metadata=build_metadata(
                numeric_features=list(NUMERIC_FEATURES),
                categorical_features=list(CATEGORICAL_FEATURES),
                target=TARGET,
                n_train=len(matrix),
                n_test=0,
                split="test",
                metrics={},
            ),
        )
        save_bundle(bundle, models_dir=tmp_path, mark_latest=True)

        assert latest_version(tmp_path) == bundle.version
        reloaded = load_bundle(bundle.version, models_dir=tmp_path)
        after = reloaded.pipeline.predict(features.head(5))

        np.testing.assert_allclose(before, after)

    def test_feature_contract_is_published(self, listings, tmp_path):
        """The metadata must name every column the pipeline was fitted on."""
        from laptop_price.models.pipeline import build_pipeline
        from laptop_price.models.registry import (
            ArtifactBundle,
            build_metadata,
            load_bundle,
            save_bundle,
        )

        matrix = build_feature_matrix(listings)
        pipeline = build_pipeline()
        pipeline.fit(matrix[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]], matrix[TARGET])

        bundle = ArtifactBundle(
            pipeline=pipeline,
            metadata=build_metadata(
                numeric_features=list(NUMERIC_FEATURES),
                categorical_features=list(CATEGORICAL_FEATURES),
                target=TARGET,
                n_train=len(matrix),
                n_test=0,
                split="test",
                metrics={},
            ),
        )
        save_bundle(bundle, models_dir=tmp_path)
        reloaded = load_bundle(bundle.version, models_dir=tmp_path)

        assert reloaded.feature_names == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

    def test_missing_model_raises_an_actionable_error(self, tmp_path):
        from laptop_price.models.registry import load_bundle

        with pytest.raises(FileNotFoundError, match="make train"):
            load_bundle(models_dir=tmp_path)

    def test_predictions_stay_inside_the_trained_band(self, listings):
        """A pricing service must never quote 40 million dinars for a laptop."""
        from laptop_price.config import CONFIG
        from laptop_price.models.pipeline import build_pipeline

        matrix = build_feature_matrix(listings)
        features = matrix[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        pipeline = build_pipeline()
        pipeline.fit(features, matrix[TARGET])

        extreme = features.head(1).copy()
        extreme.loc[:, "cpu_mark"] = 1e9
        extreme.loc[:, "estimated_component_cost"] = 1e12

        prediction = pipeline.predict(extreme)[0]
        assert CONFIG.price.min_dzd <= prediction <= CONFIG.price.max_dzd


class TestMonotonicConstraints:
    """More RAM must never lower the estimate.

    These were declared in the config and computed by a helper that nothing
    called - the same "dead code" pattern the audit flagged in the original
    project. The constraint is now passed to the estimator, and this test is
    what keeps it passed.
    """

    @pytest.fixture
    def fitted(self, listings):
        from laptop_price.models.pipeline import build_pipeline

        matrix = build_feature_matrix(listings)
        features = matrix[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        pipeline = build_pipeline()
        pipeline.fit(features, matrix[TARGET])
        return pipeline, features

    def test_constraints_are_declared(self):
        from laptop_price.models.pipeline import monotonic_constraints

        constraints = monotonic_constraints()
        assert constraints["RAM_SIZE"] == 1
        assert set(constraints) <= set(NUMERIC_FEATURES)

    def test_constraints_reach_the_estimator(self, fitted):
        pipeline, _ = fitted
        inner = getattr(pipeline, "regressor_", pipeline)
        assert inner.named_steps["model"].monotonic_cst

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            ("RAM_SIZE", [4.0, 8.0, 16.0, 32.0, 64.0]),
            ("SSD_SIZE", [128.0, 256.0, 512.0, 1024.0]),
            ("cpu_mark", [3_000.0, 8_000.0, 15_000.0, 25_000.0]),
        ],
    )
    def test_more_is_never_worth_less(self, fitted, column, values):
        pipeline, features = fitted
        row = features.head(1)

        predictions = []
        for value in values:
            probe = row.copy()
            probe.loc[:, column] = value
            predictions.append(float(pipeline.predict(probe)[0]))

        assert all(
            later >= earlier - 1e-6
            for earlier, later in zip(predictions, predictions[1:], strict=False)
        ), f"{column} {values} -> {predictions}"

    def test_preprocessor_emits_named_features(self, fitted):
        """Constraints are keyed by name, which needs DataFrame output."""
        pipeline, features = fitted
        inner = getattr(pipeline, "regressor_", pipeline)
        transformed = inner.named_steps["preprocess"].transform(features.head(3))
        assert isinstance(transformed, pd.DataFrame)
        assert "RAM_SIZE" in transformed.columns
