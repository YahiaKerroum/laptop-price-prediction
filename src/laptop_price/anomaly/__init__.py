"""Anomaly detection: the deliverable the brief asked for and the project never built.

Four products fall out of one model family:

``detectors``
    Unsupervised outlier scoring on the joint (specs, price) distribution -
    IsolationForest, LocalOutlierFactor, and ECOD/COPOD from PyOD.
``deals``
    The useful half: rank listings where the model's estimate far exceeds the
    asking price, minus the ones that look like scams rather than bargains.
``rules_check``
    Spec-inconsistency detection driven by the association rules the project
    already mines. This is the bridge that makes regression, association rules
    and anomaly detection one story instead of three separate assignments.
"""

from laptop_price.anomaly.deals import rank_deals, residual_scores
from laptop_price.anomaly.detectors import (
    AnomalyEnsemble,
    fit_detectors,
    score_listings,
)
from laptop_price.anomaly.rules_check import check_spec_consistency, load_rules

__all__ = [
    "AnomalyEnsemble",
    "check_spec_consistency",
    "fit_detectors",
    "load_rules",
    "rank_deals",
    "residual_scores",
    "score_listings",
]
