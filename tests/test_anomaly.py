"""Anomaly detection: detectors, deal ranking, and rule-based consistency.

The first test here exists because of a bug that shipped silently: PyOD marks
outliers with ``1`` while scikit-learn marks them with ``-1``. Reading PyOD with
the scikit-learn convention made ECOD and COPOD flag *nothing at all* while still
appearing to run — two of four detectors contributing zero, invisibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laptop_price.anomaly.detectors import (
    detector_agreement,
    fit_detectors,
    score_listings,
)
from laptop_price.anomaly.rules_check import (
    _parse_itemset,
    check_spec_consistency,
    summarise_violations,
)
from laptop_price.features.build import TARGET


@pytest.fixture
def listings() -> pd.DataFrame:
    """Ordinary laptops, plus a handful of deliberately impossible ones."""
    rng = np.random.default_rng(7)
    n = 400

    cpu = rng.uniform(3_000, 25_000, n)
    frame = pd.DataFrame(
        {
            "RAM_SIZE": rng.choice([8.0, 16.0, 32.0], n),
            "SSD_SIZE": rng.choice([256.0, 512.0, 1024.0], n),
            "HDD_SIZE": np.zeros(n),
            "cpu_mark": cpu,
            "gpu_g3d_mark": cpu * rng.uniform(0.4, 0.9, n),
            "gpu_tdp": rng.uniform(15, 90, n),
            "total_tdp": rng.uniform(30, 140, n),
            "SCREEN_SIZE_SNAPPED": rng.choice([14.0, 15.6], n),
            "SCREEN_RESOLUTION_ENC": rng.choice([3.0, 5.0], n),
            "estimated_component_cost": cpu * 6,
            # price tracks performance, as it should
            TARGET: cpu * 6 * rng.uniform(0.9, 1.1, n),
        }
    )

    # Five monsters priced like netbooks: normal on each axis alone, absurd jointly.
    frame.loc[:4, ["cpu_mark", "gpu_g3d_mark", "estimated_component_cost"]] = [
        60_000,
        55_000,
        400_000,
    ]
    frame.loc[:4, TARGET] = 25_000.0
    return frame


class TestDetectors:
    def test_all_four_detectors_actually_flag_something(self, listings):
        """PyOD inverts scikit-learn's label convention; reading it wrong is silent."""
        ensemble = fit_detectors(listings, contamination=0.05)
        scores = score_listings(ensemble, listings)

        for name in ensemble.detectors:
            flagged = scores[f"{name}_outlier"].sum()
            assert flagged > 0, f"{name} flagged nothing - label convention bug?"

    def test_pyod_detectors_are_present(self, listings):
        ensemble = fit_detectors(listings, contamination=0.05)
        assert {"ecod", "copod"} <= set(ensemble.detectors)

    def test_joint_outliers_are_found(self, listings):
        """A 60k-cpumark machine at 25,000 DZD is normal on every single axis."""
        ensemble = fit_detectors(listings, contamination=0.05)
        scores = score_listings(ensemble, listings)
        assert scores["any_outlier"].iloc[:5].any()

    def test_consensus_is_never_broader_than_the_union(self, listings):
        ensemble = fit_detectors(listings, contamination=0.05)
        scores = score_listings(ensemble, listings)
        assert scores["is_outlier"].sum() <= scores["any_outlier"].sum()
        assert (scores["is_outlier"] <= scores["any_outlier"]).all()

    def test_consensus_columns_are_excluded_from_the_agreement_matrix(self, listings):
        """`is_outlier` also ends in `_outlier`; it must not rate itself."""
        ensemble = fit_detectors(listings, contamination=0.05)
        agreement = detector_agreement(score_listings(ensemble, listings))

        assert "is_outlier" not in agreement.columns
        assert "any_outlier" not in agreement.columns
        assert set(agreement.columns) == {f"{n}_outlier" for n in ensemble.detectors}

    def test_agreement_matrix_is_a_valid_similarity(self, listings):
        ensemble = fit_detectors(listings, contamination=0.05)
        agreement = detector_agreement(score_listings(ensemble, listings))

        assert np.allclose(np.diag(agreement.to_numpy()), 1.0)
        assert np.allclose(agreement.to_numpy(), agreement.to_numpy().T)
        assert ((agreement >= 0) & (agreement <= 1)).all().all()


@pytest.fixture
def matrix_and_bundle():
    """A real feature matrix and a fitted bundle, with one planted bargain.

    Row 0 is given a price a fifth of what its specs command, so a correctly
    wired deal finder must surface it first.
    """
    from laptop_price.features.build import build_feature_matrix
    from laptop_price.models.pipeline import build_pipeline
    from laptop_price.models.registry import ArtifactBundle

    rng = np.random.default_rng(11)
    n = 300
    cpu = rng.uniform(4_000, 30_000, n)

    raw = pd.DataFrame(
        {
            "price_corrected": cpu * 6,
            "created_at": ["2025 03 15T10:00:00.000Z"] * n,
            "city": rng.choice(["EZZOUAR", "ORAN"], n),
            "spec_Etat": rng.choice(["BON TAT", None], n),
            "model_name": rng.choice(["THINKPAD", "MACBOOK"], n),
            "DEDICATED_GPU": [None] * n,
            "RAM_SIZE": rng.choice(["8GB", "16GB"], n),
            "SSD_SIZE": rng.choice(["256GB", "512GB"], n),
            "HDD_SIZE": [None] * n,
            "RAM_TYPE": ["DDR4"] * n,
            "cores": rng.integers(4, 12, n).astype(float),
            "cpu_mark": cpu,
            "tdp": rng.choice([15.0, 45.0], n),
            "gpu_g3d_mark": cpu * 0.6,
            "gpu_g2d_mark": rng.uniform(200, 900, n),
            "gpu_tdp": rng.uniform(10, 80, n),
            "cpu_manufacturer": ["INTEL"] * n,
            "cpu_family": ["i5"] * n,
            "cpu_generation_normalized": rng.uniform(0, 1, n),
            "SCREEN_SIZE_SNAPPED": rng.choice([14.0, 15.6], n),
            "SCREEN_RESOLUTION_STD": ["FHD"] * n,
            "SCREEN_RESOLUTION_ENC": [3.0] * n,
            "estimated_component_cost": cpu * 6,
            "price_unit_ambiguous": False,
        }
    )
    # The planted bargain: same specs, a fifth of the price.
    raw.loc[0, "price_corrected"] = raw.loc[0, "price_corrected"] / 5

    matrix = build_feature_matrix(raw)

    from laptop_price.models.train import FEATURE_COLUMNS

    pipeline = build_pipeline()
    pipeline.fit(matrix[FEATURE_COLUMNS], matrix[TARGET])
    return matrix, ArtifactBundle(pipeline=pipeline, metadata={"version": "test"})


class TestDeals:
    def test_discount_is_relative_to_the_prediction(self, matrix_and_bundle):
        """discount = (predicted - actual) / predicted, as a percentage."""
        from laptop_price.anomaly.deals import residual_scores

        matrix, bundle = matrix_and_bundle
        scores = residual_scores(matrix, bundle)

        expected = (scores["predicted"] - scores["actual"]) / scores["predicted"] * 100
        np.testing.assert_allclose(scores["discount_pct"], expected)

    def test_the_planted_bargain_is_ranked_first(self, matrix_and_bundle):
        from laptop_price.anomaly.deals import rank_deals

        matrix, bundle = matrix_and_bundle
        # detect_outliers=False: with a planted row this synthetic, the joint
        # detectors correctly call it suspicious. Here we test the ranking.
        ranked = rank_deals(matrix, bundle, top_k=5, detect_outliers=False)

        assert len(ranked) > 0
        assert ranked["discount_pct"].iloc[0] > 50
        assert ranked["discount_pct"].is_monotonic_decreasing

    def test_a_deep_discount_on_an_odd_listing_is_suspicious_not_a_bargain(self, matrix_and_bundle):
        """The distinction the whole deal feed rests on."""
        from laptop_price.anomaly.deals import classify_listings

        matrix, bundle = matrix_and_bundle
        flags = pd.Series(False, index=matrix.index)
        flags.iloc[0] = True  # the planted row is also a joint outlier

        verdicts = classify_listings(matrix, bundle, outlier_flags=flags)
        assert verdicts["verdict"].iloc[0] == "suspicious"

    def test_ordinary_listings_are_not_flagged(self, matrix_and_bundle):
        from laptop_price.anomaly.deals import classify_listings

        matrix, bundle = matrix_and_bundle
        verdicts = classify_listings(matrix, bundle)
        assert (verdicts["verdict"] == "normal").mean() > 0.8


class TestRulesCheck:
    def test_parse_mlxtend_frozenset_repr(self):
        assert _parse_itemset("frozenset({'RAM_high', 'GPU_strong'})") == frozenset(
            {"RAM_high", "GPU_strong"}
        )

    def test_parse_plain_set_repr(self):
        assert _parse_itemset("{'a'}") == frozenset({"a"})

    def test_unparseable_itemset_is_empty_not_an_exception(self):
        assert _parse_itemset("not a set at all") == frozenset()

    def test_violation_detection(self):
        transactions = pd.DataFrame(
            {
                "GPU_strong": [True, True, False],
                "RAM_high": [True, False, False],
            }
        )
        rules = pd.DataFrame(
            {
                "antecedents": [frozenset({"GPU_strong"})],
                "consequents": [frozenset({"RAM_high"})],
                "confidence": [0.9],
            }
        )
        result = check_spec_consistency(transactions, rules)

        # Row 1 claims a strong GPU with low RAM - the rule says that is unusual.
        assert result["is_inconsistent"].tolist() == [False, True, False]
        assert result.loc[1, "max_violated_confidence"] == pytest.approx(0.9)

    def test_rules_outside_the_vocabulary_are_skipped(self):
        transactions = pd.DataFrame({"GPU_strong": [True]})
        rules = pd.DataFrame(
            {
                "antecedents": [frozenset({"SOMETHING_ELSE"})],
                "consequents": [frozenset({"UNKNOWN"})],
                "confidence": [0.99],
            }
        )
        assert not check_spec_consistency(transactions, rules)["is_inconsistent"].any()

    def test_no_rules_means_no_violations(self):
        transactions = pd.DataFrame({"a": [True, False]})
        empty = pd.DataFrame(columns=["antecedents", "consequents", "confidence"])
        result = check_spec_consistency(transactions, empty)
        assert result["n_rules_violated"].sum() == 0

    def test_violation_summary(self):
        consistency = pd.DataFrame({"violations": [["rule A"], ["rule A", "rule B"], []]})
        summary = summarise_violations(consistency)
        assert summary["rule A"] == 2
        assert summary["rule B"] == 1


class TestPredictionCoherence:
    """The point estimate must never fall outside the range shown beside it.

    The point comes from a squared-error model and the bounds from independently
    fitted quantile models, so on ~1.4% of listings they disagree. "Most likely
    105,200, range 111,200-175,500" is not a defensible thing to show a user.
    """

    def test_point_estimate_lies_inside_the_range(self, matrix_and_bundle):
        from laptop_price.models.pipeline import build_quantile_pipelines
        from laptop_price.models.train import FEATURE_COLUMNS
        from laptop_price.serving import predict_one

        matrix, bundle = matrix_and_bundle
        bundle.quantile_pipelines = build_quantile_pipelines(quantiles=(0.1, 0.9))
        for pipeline in bundle.quantile_pipelines.values():
            pipeline.fit(matrix[FEATURE_COLUMNS], matrix[TARGET])

        import laptop_price.serving as serving

        serving.get_bundle.cache_clear()
        original = serving.get_bundle
        serving.get_bundle = lambda version=None: bundle
        try:
            for _, row in matrix[FEATURE_COLUMNS].head(40).iterrows():
                result = predict_one(row.to_dict())
                low, high = result["range_dzd"]
                assert low <= result["estimate_dzd"] <= high, result
        finally:
            serving.get_bundle = original

    def test_range_is_ordered(self, matrix_and_bundle):
        import laptop_price.serving as serving
        from laptop_price.models.pipeline import build_quantile_pipelines
        from laptop_price.models.train import FEATURE_COLUMNS

        matrix, bundle = matrix_and_bundle
        bundle.quantile_pipelines = build_quantile_pipelines(quantiles=(0.1, 0.9))
        for pipeline in bundle.quantile_pipelines.values():
            pipeline.fit(matrix[FEATURE_COLUMNS], matrix[TARGET])

        original = serving.get_bundle
        serving.get_bundle = lambda version=None: bundle
        try:
            result = serving.predict_one(matrix[FEATURE_COLUMNS].iloc[0].to_dict())
            assert result["range_dzd"][0] <= result["range_dzd"][1]
        finally:
            serving.get_bundle = original

    def test_headline_range_is_bounded_and_inside_the_wide_band(self, matrix_and_bundle):
        """The 10-90 band was up to 127% of the price wide - too vague to act on."""
        import laptop_price.serving as serving
        from laptop_price.models.pipeline import build_quantile_pipelines
        from laptop_price.models.train import FEATURE_COLUMNS

        matrix, bundle = matrix_and_bundle
        bundle.quantile_pipelines = build_quantile_pipelines(quantiles=(0.1, 0.9))
        for pipeline in bundle.quantile_pipelines.values():
            pipeline.fit(matrix[FEATURE_COLUMNS], matrix[TARGET])

        max_ratio = np.exp(2 * serving.LIKELY_RANGE_K_CPU_UNKNOWN * serving.LIKELY_RANGE_MAX_SPREAD)
        original = serving.get_bundle
        serving.get_bundle = lambda version=None: bundle
        try:
            for _, row in matrix[FEATURE_COLUMNS].head(40).iterrows():
                result = serving.predict_one(row.to_dict())
                low, high = result["range_dzd"]
                assert low <= result["estimate_dzd"] <= high, result
                # +100 DZD of slack for rounding to the nearest hundred
                assert high <= low * max_ratio + 100, result
        finally:
            serving.get_bundle = original

    def test_quantiles_are_monotonic_in_the_level(self, matrix_and_bundle):
        """Independently fitted quantile models cross on ~0.7% of listings.

        A predicted 10th percentile above the 50th is not a quantile function.
        Rearrangement (sorting the values onto the ordered levels) is the
        standard remedy and is provably no worse than leaving them crossed.
        """
        import laptop_price.serving as serving
        from laptop_price.models.pipeline import build_quantile_pipelines
        from laptop_price.models.train import FEATURE_COLUMNS

        matrix, bundle = matrix_and_bundle
        bundle.quantile_pipelines = build_quantile_pipelines(quantiles=(0.1, 0.5, 0.9))
        for pipeline in bundle.quantile_pipelines.values():
            pipeline.fit(matrix[FEATURE_COLUMNS], matrix[TARGET])

        original = serving.get_bundle
        serving.get_bundle = lambda version=None: bundle
        try:
            for _, row in matrix[FEATURE_COLUMNS].head(40).iterrows():
                quantiles = serving.predict_one(row.to_dict())["quantiles"]
                values = [quantiles[k] for k in sorted(quantiles, key=float)]
                assert values == sorted(values), quantiles
        finally:
            serving.get_bundle = original
