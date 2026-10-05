"""Market segmentation: Gower distance, honest metrics, and the segment report.

The tests that matter here guard against the *specific* way the original
clustering went wrong: a silhouette of 0.9796 reported for a partition of
14,161 / 498 / 550, measured in a space other than the one clustered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laptop_price.clustering.distances import (
    gower_matrix,
    gower_vector,
    sample_for_distance,
)
from laptop_price.clustering.metrics import cluster_metrics, stability_score
from laptop_price.clustering.naming import describe_segments, name_segment
from laptop_price.features.build import TARGET


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Numeric and categorical columns, including a constant one."""
    return pd.DataFrame(
        {
            "ram": [8.0, 8.0, 16.0, 32.0],
            "cpu": [5_000.0, 5_000.0, 15_000.0, 25_000.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "brand": ["THINKPAD", "THINKPAD", "MACBOOK", "ROG"],
        }
    )


class TestGower:
    def test_identical_rows_are_zero_apart(self, mixed_frame):
        matrix = gower_matrix(mixed_frame)
        assert matrix[0, 1] == pytest.approx(0.0)

    def test_distance_is_bounded_and_symmetric(self, mixed_frame):
        matrix = gower_matrix(mixed_frame)
        assert (matrix >= 0).all() and (matrix <= 1).all()
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)

    def test_a_constant_column_contributes_nothing(self, mixed_frame):
        """Its range is zero; dividing by it would be a crash or a NaN."""
        with_constant = gower_matrix(mixed_frame)
        without = gower_matrix(mixed_frame.drop(columns=["constant"]))
        # The constant column only dilutes by its share of the weight.
        assert np.all(with_constant <= without + 1e-9)
        assert np.isfinite(with_constant).all()

    def test_categorical_mismatch_costs_a_full_unit(self):
        frame = pd.DataFrame({"brand": ["A", "B"]})
        assert gower_matrix(frame)[0, 1] == pytest.approx(1.0)

    def test_missing_is_maximally_distant(self):
        """'Unknown' is not evidence of similarity, including to another unknown."""
        frame = pd.DataFrame({"x": [1.0, np.nan, 2.0]})
        matrix = gower_matrix(frame)
        assert matrix[0, 1] == pytest.approx(1.0)
        assert matrix[1, 2] == pytest.approx(1.0)

    def test_vector_matches_the_matrix_row(self, mixed_frame):
        matrix = gower_matrix(mixed_frame)
        vector = gower_vector(mixed_frame.iloc[0], mixed_frame)
        np.testing.assert_allclose(vector, matrix[0], atol=1e-9)

    def test_oversized_input_is_refused_not_attempted(self):
        """A 16k x 16k float64 matrix is 2.1 GB; fail loudly, not by OOM."""
        big = pd.DataFrame({"x": np.arange(200.0)})
        with pytest.raises(ValueError, match="refusing to build"):
            gower_matrix(big, max_rows=100)

    def test_subsampling_preserves_the_price_distribution(self):
        frame = pd.DataFrame(
            {TARGET: np.concatenate([np.full(900, 50_000.0), np.full(100, 500_000.0)])}
        )
        sample = sample_for_distance(frame, n=200, stratify_on=TARGET)
        assert len(sample) <= 200
        # The premium tail must survive; a uniform sample is what let the
        # original mistake a handful of outliers for a segment.
        assert (sample[TARGET] > 100_000).any()


class TestClusterMetrics:
    def test_reports_the_space_it_was_given(self):
        matrix = np.random.default_rng(0).normal(size=(200, 3))
        labels = np.repeat([0, 1], 100)
        result = cluster_metrics(matrix, labels, space="test space")
        assert result["space"] == "test space"

    def test_degenerate_partition_is_visible_in_the_shape(self):
        """A high silhouette with a 90% largest cluster is the failure mode."""
        rng = np.random.default_rng(0)
        blob = rng.normal(0, 0.1, size=(900, 2))
        far = rng.normal(50, 0.1, size=(100, 2))
        matrix = np.vstack([blob, far])
        labels = np.array([0] * 900 + [1] * 100)

        result = cluster_metrics(matrix, labels, space="test")
        assert result["silhouette"] > 0.9  # looks wonderful
        assert result["largest_share"] == pytest.approx(0.9)  # and is not
        assert result["size_ratio"] == pytest.approx(9.0)

    def test_noise_labels_are_counted_not_clustered(self):
        matrix = np.random.default_rng(0).normal(size=(100, 2))
        labels = np.array([-1] * 20 + [0] * 40 + [1] * 40)
        result = cluster_metrics(matrix, labels, space="test")
        assert result["n_clusters"] == 2
        assert result["n_noise"] == 20
        assert result["noise_fraction"] == pytest.approx(0.2)

    def test_single_cluster_scores_nan_rather_than_raising(self):
        matrix = np.random.default_rng(0).normal(size=(50, 2))
        result = cluster_metrics(matrix, np.zeros(50, dtype=int), space="test")
        assert result["n_clusters"] == 1
        assert np.isnan(result["silhouette"])

    def test_stability_is_high_for_real_structure(self):
        from sklearn.cluster import KMeans

        rng = np.random.default_rng(0)
        matrix = np.vstack([rng.normal(-5, 0.5, size=(150, 2)), rng.normal(5, 0.5, size=(150, 2))])
        result = stability_score(
            KMeans(n_clusters=2, n_init=10, random_state=0), matrix, n_bootstrap=5
        )
        assert result["mean_ari"] > 0.9

    def test_stability_is_low_for_structure_that_is_not_there(self):
        """Clustering uniform noise gives a different answer every resample."""
        from sklearn.cluster import KMeans

        matrix = np.random.default_rng(0).uniform(size=(300, 8))
        result = stability_score(
            KMeans(n_clusters=6, n_init=3, random_state=0), matrix, n_bootstrap=5
        )
        assert result["mean_ari"] < 0.7


class TestSegmentNaming:
    def test_a_gaming_machine_is_named_as_one(self):
        profile = pd.Series(
            {
                "gpu_g3d_mark": 19_000.0,
                "cpu_mark": 25_000.0,
                "RAM_SIZE": 16.0,
                "SCREEN_SIZE_SNAPPED": 15.6,
                TARGET: 260_000.0,
            }
        )
        market = pd.Series({TARGET: 95_000.0})
        name = name_segment(profile, market)
        assert "gaming" in name
        assert "premium" in name

    def test_a_budget_machine_is_named_as_one(self):
        profile = pd.Series(
            {
                "gpu_g3d_mark": 900.0,
                "cpu_mark": 4_000.0,
                "RAM_SIZE": 4.0,
                "SCREEN_SIZE_SNAPPED": 15.6,
                TARGET: 45_000.0,
            }
        )
        market = pd.Series({TARGET: 95_000.0})
        name = name_segment(profile, market)
        assert "budget" in name or "cheap" in name

    def test_segment_report_shape(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                TARGET: rng.uniform(30_000, 300_000, 200),
                "RAM_SIZE": rng.choice([8.0, 16.0], 200),
                "cpu_mark": rng.uniform(4_000, 25_000, 200),
                "gpu_g3d_mark": rng.uniform(500, 20_000, 200),
                "SCREEN_SIZE_SNAPPED": rng.choice([14.0, 15.6], 200),
                "brand": rng.choice(["THINKPAD", "MACBOOK"], 200),
            }
        )
        labels = np.repeat([0, 1], 100)
        report = describe_segments(frame, labels)

        assert len(report) == 2
        assert report["share"].sum() == pytest.approx(1.0)
        assert report["name"].str.len().min() > 0
        assert "top_brands" in report.columns

    def test_noise_is_labelled_as_noise(self):
        frame = pd.DataFrame({TARGET: [50_000.0] * 10, "cpu_mark": [5_000.0] * 10})
        labels = np.array([-1] * 5 + [0] * 5)
        report = describe_segments(frame, labels)
        assert "noise" in report.set_index("segment").loc[-1, "name"]


class TestConsensus:
    """Consensus clustering, and the instability that motivates it.

    The base pipeline stops being seed-reproducible somewhere between 4,500 and
    8,000 rows (measured: identical at 4,500, ARI 0.43 at 8,000, 0.25 on the full
    dataset). Every number from a single run above that threshold would describe
    the run rather than the market.
    """

    def test_co_association_counts_shared_membership(self):
        from laptop_price.clustering.consensus import co_association_matrix

        runs = [
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
        ]
        matrix = co_association_matrix(runs)

        assert matrix[0, 1] == pytest.approx(2 / 3)  # together twice of three
        assert matrix[2, 3] == pytest.approx(1.0)  # always together
        assert matrix[0, 3] == pytest.approx(0.0)  # never together
        np.testing.assert_allclose(np.diag(matrix), 1.0)

    def test_noise_agrees_with_nothing(self):
        """HDBSCAN declined to place these; that is not evidence of similarity."""
        from laptop_price.clustering.consensus import co_association_matrix

        matrix = co_association_matrix([np.array([-1, -1, 0, 0])])
        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[2, 3] == pytest.approx(1.0)

    def test_run_agreement_detects_identical_runs(self):
        from laptop_price.clustering.consensus import run_to_run_agreement

        labels = np.array([0, 0, 1, 1, 2, 2])
        assert run_to_run_agreement([labels, labels.copy()])["mean_ari"] == pytest.approx(1.0)

    def test_run_agreement_detects_unrelated_runs(self):
        from laptop_price.clustering.consensus import run_to_run_agreement

        rng = np.random.default_rng(0)
        runs = [rng.integers(0, 5, size=200) for _ in range(3)]
        assert run_to_run_agreement(runs)["mean_ari"] < 0.2

    def test_consensus_sample_stays_below_the_reproducibility_threshold(self):
        """4,000 is not an arbitrary memory bound - above ~4,500 the base
        clusterer stops being seed-reproducible, so the ensemble would be built
        from noise rather than from repeatable parts."""
        from laptop_price.clustering.consensus import CONSENSUS_SAMPLE

        assert CONSENSUS_SAMPLE <= 4_500
