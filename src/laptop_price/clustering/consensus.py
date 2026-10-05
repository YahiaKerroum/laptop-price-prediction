"""Consensus clustering, because a single run of this pipeline is not reproducible.

The problem
-----------
On the full dataset, UMAP 0.5.7 does not produce a reproducible embedding. Two
identical calls - same data, same ``random_state``, same process - gave
clusterings with an **adjusted Rand index of 0.25**, barely better than
unrelated. Setting ``n_jobs=1``, forcing ``NUMBA_NUM_THREADS=1``, and supplying a
precomputed exact k-nearest-neighbour graph all left it non-deterministic.

Every number you could report from one such run - silhouette, cluster count,
segment names, sizes - would be a property of that run rather than of the
market. Reporting one as a finding would repeat the original project's mistake
in a new form.

But it is not unconditional. Measured directly, with the same seed:

===========  ==========================
Rows         Two runs identical?
===========  ==========================
2,000        yes (ARI 1.000)
4,000        yes (ARI 1.000)
4,500        yes (ARI 1.000)
8,000        **no** (ARI 0.431)
16,255       **no** (ARI 0.253)
===========  ==========================

There is a threshold between 4,500 and 8,000 rows. The likely mechanism is
UMAP's switch from exact nearest neighbours to approximate ones (pynndescent)
on larger inputs, whose parallel graph construction is not seed-reproducible.

The answer
----------
Two parts, and the first is what makes the whole pipeline reproducible:

1. **Build the consensus below the threshold.** The co-association matrix is
   assembled from a stratified subsample small enough that each individual run
   is deterministic, so the ensemble is built from repeatable parts rather than
   from noise.
2. **Average over seeds anyway.** Run the pipeline ``n_runs`` times with
   *different* seeds, count how often each pair of listings lands in the same
   cluster, and cluster that. Seed-to-seed variation is real even where a single
   seed is reproducible, and averaging over it is what makes the segments a
   property of the data.

Three things fall out:

* the result is stable by construction - it is an average over runs
* the **co-association strength** is a direct, interpretable measure of how real
  the structure is: pairs that always cluster together are genuinely similar,
  pairs near 0.5 are an artefact of whichever run you happened to look at
* the disagreement between runs becomes evidence rather than a defect

Memory
------
A co-association matrix is O(n²): 2.1 GB at 16,255 listings, which is the other
reason to subsample. The remaining listings are assigned to the nearest
consensus centroid **in feature space** - not through the embedding, since that
is the part that is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from laptop_price.clustering.distances import sample_for_distance
from laptop_price.clustering.metrics import cluster_metrics
from laptop_price.config import CONFIG
from laptop_price.features.build import TARGET

#: Rows used to build the co-association matrix. Chosen to sit below the
#: threshold where the base clusterer stops being seed-reproducible (measured
#: between 4,500 and 8,000 rows), not merely to bound memory.
CONSENSUS_SAMPLE = 4_000

#: Pairs co-clustering at least this often are treated as belonging together.
DEFAULT_CONSENSUS_THRESHOLD = 0.5


@dataclass
class ConsensusResult:
    """A segmentation averaged over many unstable runs."""

    labels: np.ndarray
    sample_index: np.ndarray
    co_association: np.ndarray
    n_runs: int
    metrics: dict[str, Any] = field(default_factory=dict)
    run_agreement: dict[str, float] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))


def run_to_run_agreement(label_sets: list[np.ndarray]) -> dict[str, float]:
    """Adjusted Rand index between every pair of runs.

    This is the diagnostic that motivates the whole module. A mean ARI near 1
    means the base clusterer is reproducible and consensus is unnecessary; near
    0 means a single run tells you nothing about the market.
    """
    from sklearn.metrics import adjusted_rand_score

    scores = [
        float(adjusted_rand_score(label_sets[i], label_sets[j]))
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    if not scores:  # pragma: no cover
        return {"mean_ari": float("nan"), "min_ari": float("nan"), "n_pairs": 0}
    return {
        "mean_ari": float(np.mean(scores)),
        "std_ari": float(np.std(scores)),
        "min_ari": float(np.min(scores)),
        "max_ari": float(np.max(scores)),
        "n_pairs": len(scores),
    }


def co_association_matrix(label_sets: list[np.ndarray]) -> np.ndarray:
    """Fraction of runs in which each pair of points shares a cluster.

    Noise points (label < 0) are treated as belonging with nothing, which is the
    honest reading: HDBSCAN declined to place them.
    """
    n = len(label_sets[0])
    counts = np.zeros((n, n), dtype=np.float32)

    for labels in label_sets:
        labels = np.asarray(labels)
        # Outer equality, with noise excluded from agreeing with anything.
        same = labels[:, None] == labels[None, :]
        valid = (labels >= 0)[:, None] & (labels >= 0)[None, :]
        counts += (same & valid).astype(np.float32)

    return counts / len(label_sets)


def consensus_segments(
    df: pd.DataFrame,
    *,
    n_runs: int = 10,
    n_clusters: int | None = None,
    sample_size: int = CONSENSUS_SAMPLE,
    min_cluster_size: int | None = None,
    random_state: int | None = None,
) -> ConsensusResult:
    """Segment the market by consensus over ``n_runs`` independent runs.

    Parameters
    ----------
    n_clusters
        Cut the consensus dendrogram into this many segments. Left as None, the
        median cluster count across the individual runs is used - letting the
        base clusterer choose its own granularity, then averaging that choice.

    Returns
    -------
    ConsensusResult
        Labels for the *subsample*, the co-association matrix, per-run agreement,
        and the usual cluster metrics measured on the consensus distance.
    """
    from laptop_price.clustering.segment import segment_market

    random_state = CONFIG.model.random_state if random_state is None else random_state

    sample = sample_for_distance(df, n=sample_size, stratify_on=TARGET, random_state=random_state)
    min_cluster_size = min_cluster_size or max(30, len(sample) // 40)

    label_sets: list[np.ndarray] = []
    for run in range(n_runs):
        result = segment_market(
            sample,
            supervised=True,
            with_stability=False,
            min_cluster_size=min_cluster_size,
            # Vary the seed deliberately. The point is to average over the
            # pipeline's variability, not to pretend it is not there.
            random_state=random_state + run,
        )
        label_sets.append(result.labels)

    agreement = run_to_run_agreement(label_sets)
    matrix = co_association_matrix(label_sets)

    if n_clusters is None:
        counts = [len(np.unique(labels[labels >= 0])) for labels in label_sets]
        n_clusters = max(2, int(np.median(counts)))

    # Average linkage on the consensus *distance*: 1 - co-association.
    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(distance.astype(np.float64))

    metrics = cluster_metrics(
        distance.astype(np.float64),
        labels,
        space=f"consensus over {n_runs} runs (1 - co-association)",
        random_state=random_state,
    )
    metrics["method"] = f"consensus ({n_runs} runs) -> average linkage"
    metrics["consensus_strength"] = float(_mean_within_cluster_consensus(matrix, labels))
    metrics |= {f"run_{k}": v for k, v in agreement.items()}

    return ConsensusResult(
        labels=labels,
        sample_index=sample.index.to_numpy(),
        co_association=matrix,
        n_runs=n_runs,
        metrics=metrics,
        run_agreement=agreement,
    )


def _mean_within_cluster_consensus(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Average co-association among pairs the consensus places together.

    1.0 means every run agreed on every within-segment pair. Near 0.5 means the
    segments are an average of runs that mostly disagreed - which is worth
    knowing before anyone builds on them.
    """
    scores = []
    for cluster in np.unique(labels):
        index = np.flatnonzero(labels == cluster)
        if len(index) < 2:
            continue
        block = matrix[np.ix_(index, index)]
        upper = block[np.triu_indices(len(index), k=1)]
        scores.append(float(upper.mean()))
    return float(np.mean(scores)) if scores else float("nan")


def assign_remaining(
    df: pd.DataFrame,
    result: ConsensusResult,
    *,
    feature_columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Extend consensus labels from the subsample to every listing.

    Assignment is by nearest consensus centroid **in feature space**, not in the
    embedding: the embedding is the part that is not reproducible, so projecting
    new points through it would reintroduce exactly the instability the consensus
    removed.
    """
    from sklearn.neighbors import NearestCentroid

    from laptop_price.clustering.segment import build_preprocessor

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(df)

    positions = df.index.get_indexer(result.sample_index)
    positions = positions[positions >= 0]

    classifier = NearestCentroid().fit(matrix[positions], result.labels[: len(positions)])
    return classifier.predict(matrix)


__all__ = [
    "CONSENSUS_SAMPLE",
    "ConsensusResult",
    "assign_remaining",
    "co_association_matrix",
    "consensus_segments",
    "run_to_run_agreement",
]
