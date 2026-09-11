"""Market segmentation: UMAP -> HDBSCAN, with Gaussian mixtures for soft membership.

Why not PCA -> K-Means
----------------------
K-Means assumes roughly spherical clusters of similar size and *forces every
point into one*. A used-laptop marketplace is neither: it is a dense mainstream
mass with long thin tails, and a genuinely unusual listing belongs to no segment
at all. Given that shape, K-Means produces exactly what the original notebook
got - one enormous cluster plus a few slivers of outliers.

HDBSCAN finds variable-density clusters and is allowed to label points as noise,
which is the honest answer for a long-tailed marketplace. UMAP first, because
HDBSCAN degrades in high dimensions and UMAP preserves local neighbourhood
structure far better than PCA, which optimises for global variance and flattens
precisely the local structure clustering depends on.

**The silhouette is reported in the embedding that was clustered**, not in a
further projection. See :mod:`laptop_price.clustering.metrics`.

Supervised embedding
--------------------
``supervised=True`` passes the price to UMAP as a target. Clusters then group by
*what drives their price* rather than by raw specification similarity, which is
the difference between a taxonomy of hardware and a segmentation of a market.
Both are produced so the comparison is on the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from laptop_price.clustering.metrics import cluster_metrics, stability_score
from laptop_price.config import CONFIG
from laptop_price.features.build import TARGET

#: Columns the segmentation sees. Price is deliberately excluded from the
#: feature set - segmenting on price and then reporting that the segments differ
#: in price would be circular. It enters only as the supervision target.
SEGMENT_NUMERIC: tuple[str, ...] = (
    "RAM_SIZE",
    "SSD_SIZE",
    "HDD_SIZE",
    "cpu_mark",
    "gpu_g3d_mark",
    "gpu_tdp",
    "total_tdp",
    "cores",
    "SCREEN_SIZE_SNAPPED",
    "SCREEN_RESOLUTION_ENC",
    "RAM_TYPE",
    "spec_Etat",
)

SEGMENT_CATEGORICAL: tuple[str, ...] = ("brand",)


@dataclass
class SegmentationResult:
    """A fitted segmentation, everything needed to score and explain it."""

    labels: np.ndarray
    embedding: np.ndarray
    matrix: np.ndarray
    method: str
    space: str
    estimator: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    probabilities: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))


def build_preprocessor() -> ColumnTransformer:
    """Scale for distance, not for interpretability.

    ``QuantileTransformer`` rather than ``RobustScaler``: the original used the
    latter on columns whose IQR is zero. ``HDD_SIZE`` is 0 at both the 25th and
    75th percentile - 92% of listings have no HDD - so robust scaling was a
    no-op there and raw magnitudes dominated the distance. A rank-based
    transform maps every column onto the same bounded, uniform scale no matter
    how degenerate its spread.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", QuantileTransformer(output_distribution="uniform", random_state=42)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=100, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(SEGMENT_NUMERIC)),
            ("cat", categorical, list(SEGMENT_CATEGORICAL)),
        ],
        remainder="drop",
    )


def embed(
    matrix: np.ndarray,
    *,
    target: np.ndarray | None = None,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    random_state: int | None = None,
) -> np.ndarray:
    """UMAP embedding, optionally supervised by price.

    ``min_dist=0.0`` is deliberate: it lets UMAP pack points tightly, which is
    what HDBSCAN wants. A larger value produces a prettier scatter plot and a
    worse clustering.
    """
    import umap

    random_state = CONFIG.model.random_state if random_state is None else random_state
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    # Supervised UMAP: log price, because the raw scale would dominate.
    return reducer.fit_transform(matrix, y=np.log1p(target) if target is not None else None)


def segment_market(
    df: pd.DataFrame,
    *,
    supervised: bool = True,
    min_cluster_size: int | None = None,
    with_stability: bool = True,
    random_state: int | None = None,
) -> SegmentationResult:
    """UMAP -> HDBSCAN over the listings.

    Parameters
    ----------
    supervised
        Let price supervise the embedding, so clusters group by price behaviour
        rather than by raw hardware similarity.
    min_cluster_size
        Defaults to 1% of the dataset. A segment smaller than that is not a
        market segment, it is a handful of listings.
    """
    random_state = CONFIG.model.random_state if random_state is None else random_state
    min_cluster_size = min_cluster_size or max(50, len(df) // 100)

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(df)

    target = df[TARGET].to_numpy(dtype=float) if (supervised and TARGET in df) else None
    embedding = embed(matrix, target=target, random_state=random_state)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embedding)

    space = f"UMAP({'supervised' if target is not None else 'unsupervised'}, 2D)"
    metrics = cluster_metrics(embedding, labels, space=space, random_state=random_state)
    metrics["method"] = f"UMAP -> HDBSCAN ({'supervised' if target is not None else 'unsupervised'})"

    if with_stability:
        metrics |= stability_score(
            HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10),
            embedding,
            random_state=random_state,
        )

    return SegmentationResult(
        labels=labels,
        embedding=embedding,
        matrix=matrix,
        method=metrics["method"],
        space=space,
        estimator=clusterer,
        metrics=metrics,
        probabilities=getattr(clusterer, "probabilities_", None),
    )


def segment_with_mixture(
    df: pd.DataFrame,
    *,
    n_components: int = 5,
    random_state: int | None = None,
) -> SegmentationResult:
    """Gaussian mixture over the UMAP embedding, for **soft** membership.

    A laptop can be 70% "student bureautique" and 30% "budget gaming". A hard
    label denies that; ``probabilities`` keeps it. Useful wherever a listing sits
    genuinely between segments, which in this market is most of the mainstream.
    """
    random_state = CONFIG.model.random_state if random_state is None else random_state

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(df)
    target = df[TARGET].to_numpy(dtype=float) if TARGET in df else None
    embedding = embed(matrix, target=target, random_state=random_state)

    mixture = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        random_state=random_state,
    )
    labels = mixture.fit_predict(embedding)

    space = "UMAP(supervised, 2D)"
    metrics = cluster_metrics(embedding, labels, space=space, random_state=random_state)
    metrics["method"] = f"UMAP -> GaussianMixture (k={n_components})"
    metrics["bic"] = float(mixture.bic(embedding))

    return SegmentationResult(
        labels=labels,
        embedding=embedding,
        matrix=matrix,
        method=metrics["method"],
        space=space,
        estimator=mixture,
        metrics=metrics,
        probabilities=mixture.predict_proba(embedding),
    )


def segment_with_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int | None = None,
    n_init: int = 20,
) -> SegmentationResult:
    """K-Means on scaled features - the original approach, scored honestly.

    Kept so the comparison table can show what the rebuild bought. Two
    differences from the original: the scaler is fitted **once** (the original
    called ``fit_transform`` a second time over the full dataset, discarding the
    train-fitted scaler), and the silhouette is measured in the space actually
    clustered rather than in a PCA projection of it.
    """
    random_state = CONFIG.model.random_state if random_state is None else random_state

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(matrix)

    space = "scaled features (the clustering space)"
    metrics = cluster_metrics(matrix, labels, space=space, random_state=random_state)
    metrics["method"] = f"K-Means (k={n_clusters})"
    metrics |= stability_score(
        KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state),
        matrix,
        random_state=random_state,
    )

    return SegmentationResult(
        labels=labels,
        embedding=matrix,
        matrix=matrix,
        method=metrics["method"],
        space=space,
        estimator=kmeans,
        metrics=metrics,
    )


__all__ = [
    "SEGMENT_CATEGORICAL",
    "SEGMENT_NUMERIC",
    "SegmentationResult",
    "build_preprocessor",
    "embed",
    "segment_market",
    "segment_with_kmeans",
    "segment_with_mixture",
]
