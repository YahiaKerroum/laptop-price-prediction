"""Market segmentation, rebuilt.

The original `clustering.ipynb` reported a silhouette of **0.9796** at k=4 and
called it a success. The cluster sizes give it away: 14,161 / 498 / 550. That is
not a segmented market, it is a handful of extreme outliers peeled off one blob.
Three things caused it:

* ``RobustScaler`` on features whose IQR is zero. ``HDD_SIZE`` is 0 at both the
  25th and 75th percentile - 92% of listings have no HDD - so scaling is a no-op
  there and raw magnitudes dominate.
* Raw ``price_preview`` (1,800 to 3,550,000) fed in alongside ``SSD_SIZE``
  (up to 12,800), so Euclidean distance lived in one or two dimensions.
* The scaler fitted on the train split had ``fit_transform`` called on it again
  over the full dataset, silently discarding the earlier fit.

``clustering_optimized.ipynb`` is better (silhouette 0.67) but measures
silhouette **in PCA space after reducing**, which flatters the score because PCA
discards exactly the directions that make clusters look messy.

This package rebuilds the analysis around a question - *what natural segments
exist in this market, and how does each one price?* - with:

``distances``  Gower distance, so mixed numeric/categorical data is compared
               without butchering it into numbers first
``segment``    UMAP -> HDBSCAN and Gaussian mixtures, instead of PCA -> K-Means
``metrics``    silhouette **in the space actually clustered**, plus
               Davies-Bouldin, Calinski-Harabasz, and bootstrap stability
``naming``     readable segment labels generated from centroid statistics
"""

from laptop_price.clustering.distances import gower_matrix, gower_vector
from laptop_price.clustering.metrics import (
    cluster_metrics,
    compare_clusterings,
    stability_score,
)
from laptop_price.clustering.naming import describe_segments, name_segment
from laptop_price.clustering.segment import (
    Embedding,
    SegmentationResult,
    embed,
    segment_market,
)

__all__ = [
    "Embedding",
    "SegmentationResult",
    "cluster_metrics",
    "compare_clusterings",
    "describe_segments",
    "embed",
    "gower_matrix",
    "gower_vector",
    "name_segment",
    "segment_market",
    "stability_score",
]
