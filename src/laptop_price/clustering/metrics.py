"""Cluster quality, reported honestly.

Two rules this module exists to enforce.

**Report silhouette in the space you clustered in.** The original
``clustering_optimized.ipynb`` clustered on scaled features, then measured
silhouette on the PCA projection. PCA discards exactly the directions that make
clusters overlap, so the score it reports is not the score of the clustering it
performed. :func:`cluster_metrics` takes the matrix used for fitting and refuses
to be handed a different one silently.

**A single number is not evidence.** A silhouette of 0.98 with sizes
14,161 / 498 / 550 is a red flag, not a trophy, and no silhouette threshold
would have caught it. So every clustering is reported with Davies-Bouldin and
Calinski-Harabasz alongside, with the size distribution, and with **bootstrap
stability** - whether the same structure reappears when the data is resampled.
Stability is what separates real structure from an artefact of one particular
sample, and it would have flagged the original result immediately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

#: Silhouette on a large matrix is O(n²); sample beyond this.
SILHOUETTE_SAMPLE = 5_000


def cluster_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    space: str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score a clustering in the space it was fitted in.

    Parameters
    ----------
    matrix
        **The matrix the clusterer was fitted on.** Passing a projection here is
        the mistake this module exists to prevent.
    space
        A human-readable name for that space, recorded in the output so a reader
        can tell what was actually measured.

    Returns
    -------
    dict
        ``silhouette`` (higher is better, -1 to 1), ``davies_bouldin`` (lower is
        better), ``calinski_harabasz`` (higher is better), the cluster count and
        sizes, the noise fraction, and the ``size_ratio`` of the largest cluster
        to the smallest.
    """
    labels = np.asarray(labels)
    clustered = labels >= 0  # HDBSCAN marks noise as -1
    unique = np.unique(labels[clustered])

    sizes = pd.Series(labels[clustered]).value_counts().sort_index()
    result: dict[str, Any] = {
        "space": space,
        "n_clusters": int(len(unique)),
        "n_noise": int((~clustered).sum()),
        "noise_fraction": float((~clustered).mean()),
        "sizes": sizes.to_dict(),
        "largest_share": float(sizes.max() / sizes.sum()) if len(sizes) else float("nan"),
        "size_ratio": float(sizes.max() / sizes.min()) if len(sizes) > 1 else float("nan"),
    }

    if len(unique) < 2:
        result |= {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }
        return result

    scored = matrix[clustered]
    scored_labels = labels[clustered]

    sample_size = min(SILHOUETTE_SAMPLE, len(scored))
    result["silhouette"] = float(
        silhouette_score(scored, scored_labels, sample_size=sample_size, random_state=random_state)
    )
    result["davies_bouldin"] = float(davies_bouldin_score(scored, scored_labels))
    result["calinski_harabasz"] = float(calinski_harabasz_score(scored, scored_labels))
    return result


def stability_score(
    estimator: Any,
    matrix: np.ndarray,
    *,
    n_bootstrap: int = 20,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """How reproducible is this clustering under resampling?

    Refits the estimator on ``n_bootstrap`` subsamples and measures the adjusted
    Rand index between each subsample's labels and the labels the full fit gave
    those same rows. High mean ARI means the structure is in the data; low means
    it is an artefact of this particular sample.

    This is the diagnostic that would have caught the original 0.98-silhouette
    result: peeling a few outliers off one blob is not stable, because which
    points count as extreme shifts with the sample.

    Returns
    -------
    dict
        ``mean_ari``, ``std_ari``, ``min_ari`` and the number of successful fits.
    """
    rng = np.random.default_rng(random_state)
    reference = np.asarray(_fit_predict(clone_estimator(estimator), matrix))

    scores: list[float] = []
    n_sample = int(len(matrix) * sample_fraction)

    for _ in range(n_bootstrap):
        index = rng.choice(len(matrix), size=n_sample, replace=False)
        try:
            labels = _fit_predict(clone_estimator(estimator), matrix[index])
        except Exception:  # pragma: no cover - a degenerate subsample
            continue
        scores.append(float(adjusted_rand_score(reference[index], labels)))

    if not scores:  # pragma: no cover
        return {
            "mean_ari": float("nan"),
            "std_ari": float("nan"),
            "min_ari": float("nan"),
            "n_bootstrap": 0,
        }

    return {
        "mean_ari": float(np.mean(scores)),
        "std_ari": float(np.std(scores)),
        "min_ari": float(np.min(scores)),
        "n_bootstrap": len(scores),
    }


def clone_estimator(estimator: Any) -> Any:
    """Clone, falling back to the original for estimators sklearn cannot clone."""
    try:
        return clone(estimator)
    except TypeError:  # pragma: no cover - non-sklearn clusterers
        return estimator


def _fit_predict(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "fit_predict"):
        return estimator.fit_predict(matrix)
    estimator.fit(matrix)  # pragma: no cover
    return estimator.predict(matrix)


def compare_clusterings(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble scored clusterings into one comparison table.

    Ordered so the reader sees the shape of the partition (cluster count, how
    much sits in the largest cluster) *before* the quality scores - a high
    silhouette next to a 90% largest-cluster share is the signature of the
    failure mode this rebuild is correcting.
    """
    frame = pd.DataFrame(results)
    columns = [
        "method",
        "space",
        "n_clusters",
        "largest_share",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_ari",
    ]
    present = [c for c in columns if c in frame.columns]
    return frame[present].sort_values("silhouette", ascending=False).reset_index(drop=True)


def format_comparison(frame: pd.DataFrame) -> str:
    """Render the comparison table for a report."""
    view = frame.copy()
    for column, spec in (
        ("largest_share", "{:.1%}"),
        ("noise_fraction", "{:.1%}"),
        ("silhouette", "{:.3f}"),
        ("davies_bouldin", "{:.3f}"),
        ("calinski_harabasz", "{:,.0f}"),
        ("mean_ari", "{:.3f}"),
    ):
        if column in view:
            view[column] = view[column].map(lambda v, s=spec: s.format(v) if pd.notna(v) else "-")
    return view.to_string(index=False)


__all__ = [
    "SILHOUETTE_SAMPLE",
    "cluster_metrics",
    "clone_estimator",
    "compare_clusterings",
    "format_comparison",
    "stability_score",
]
