"""Unsupervised outlier detection on the joint (specs, price) distribution.

An RTX 4090 machine listed at 60,000 DZD is not a bargain, it is a scam or a
typo. Detecting that needs the *joint* distribution: neither the specs alone nor
the price alone is unusual, only the combination is.

Four detectors are benchmarked rather than one, because unsupervised anomaly
detection is genuinely hard to evaluate and agreement between independent
methods is the cheapest evidence available. ``AnomalyEnsemble`` reports each
detector separately and a majority vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from laptop_price.config import CONFIG
from laptop_price.features.build import TARGET

#: Columns the detectors see. Categorical levels are deliberately excluded: a
#: rare city is not an anomaly, and one-hot columns dominate Euclidean distance.
DETECTOR_FEATURES: tuple[str, ...] = (
    "RAM_SIZE",
    "SSD_SIZE",
    "HDD_SIZE",
    "cpu_mark",
    "gpu_g3d_mark",
    "gpu_tdp",
    "total_tdp",
    "SCREEN_SIZE_SNAPPED",
    "SCREEN_RESOLUTION_ENC",
    "estimated_component_cost",
)


@dataclass
class AnomalyEnsemble:
    """Fitted detectors plus the preprocessing they share."""

    preprocessor: Pipeline
    detectors: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = DETECTOR_FEATURES
    contamination: float = 0.02
    #: How many detectors must agree before `is_outlier` is set.
    consensus_min: int = 2

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(_detector_frame(df, self.feature_names))


def _detector_frame(df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Specs plus the (log) price - the joint distribution the detectors need."""
    columns = [c for c in features if c in df.columns]
    frame = df[columns].copy()
    if TARGET in df.columns:
        # Log price, because the raw scale spans three orders of magnitude and
        # would otherwise be the only dimension distance can see.
        frame["log_price"] = np.log1p(pd.to_numeric(df[TARGET], errors="coerce"))
    return frame


def fit_detectors(
    df: pd.DataFrame,
    *,
    features: tuple[str, ...] = DETECTOR_FEATURES,
    contamination: float | None = None,
    random_state: int | None = None,
) -> AnomalyEnsemble:
    """Fit IsolationForest, LOF and (when PyOD is available) ECOD and COPOD."""
    contamination = CONFIG.anomaly.contamination if contamination is None else contamination
    random_state = CONFIG.model.random_state if random_state is None else random_state

    preprocessor = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    matrix = preprocessor.fit_transform(_detector_frame(df, features))

    detectors: dict[str, Any] = {
        "isolation_forest": IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        ).fit(matrix),
        # novelty=False: we score the same rows we fit on, which is what a
        # "flag the odd listings already in the dataset" task actually wants.
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=35, contamination=contamination, novelty=False
        ),
    }
    detectors["local_outlier_factor"].fit(matrix)

    try:
        from pyod.models.copod import COPOD
        from pyod.models.ecod import ECOD

        # Parameter-free and fast; underused outside the PyOD community.
        detectors["ecod"] = ECOD(contamination=contamination).fit(matrix)
        detectors["copod"] = COPOD(contamination=contamination).fit(matrix)
    except ImportError:  # pragma: no cover - pyod is a pinned dependency
        pass

    return AnomalyEnsemble(
        preprocessor=preprocessor,
        detectors=detectors,
        feature_names=features,
        contamination=contamination,
    )


def score_listings(ensemble: AnomalyEnsemble, df: pd.DataFrame) -> pd.DataFrame:
    """Per-detector outlier flags and a majority vote.

    Returns
    -------
    DataFrame
        One ``<detector>_outlier`` boolean column per detector, an
        ``n_detectors_flagged`` count, and ``is_outlier`` for the majority vote.
    """
    matrix = ensemble.transform(df)
    result = pd.DataFrame(index=df.index)

    for name, detector in ensemble.detectors.items():
        if name == "local_outlier_factor":
            # LOF without novelty only exposes the labels from its fit.
            flags = detector.fit_predict(matrix) == -1
        elif _is_pyod(detector):
            # PyOD marks outliers with 1; scikit-learn marks them with -1.
            # Getting this backwards makes ECOD and COPOD flag nothing at all,
            # silently, while still looking like they ran.
            flags = detector.predict(matrix) == 1
        else:
            flags = detector.predict(matrix) == -1
        result[f"{name}_outlier"] = np.asarray(flags, dtype=bool)

    # Built from the detector names rather than by suffix: "is_outlier" also ends
    # in "_outlier", and including the consensus column in its own input is the
    # kind of bug that quietly halves every count.
    flag_columns = [f"{name}_outlier" for name in ensemble.detectors]
    result["n_detectors_flagged"] = result[flag_columns].sum(axis=1)

    # Two views, because the detectors disagree sharply on this data (see
    # detector_agreement) and the right threshold depends on the question:
    #   is_outlier  - majority vote, for "what is definitely odd?"
    #   any_outlier - union, for "what should not be advertised as a bargain?"
    result["is_outlier"] = result["n_detectors_flagged"] >= ensemble.consensus_min
    result["any_outlier"] = result["n_detectors_flagged"] >= 1
    return result


def _is_pyod(detector: Any) -> bool:
    """True for PyOD estimators, which invert scikit-learn's label convention."""
    return type(detector).__module__.startswith("pyod")


def detector_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard agreement between detectors.

    Evaluating unsupervised detection without labels is mostly impossible; how
    much independent methods agree is the honest proxy to report.
    """
    consensus = {"is_outlier", "any_outlier"}
    columns = [c for c in scores.columns if c.endswith("_outlier") and c not in consensus]
    frame = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for left in columns:
        for right in columns:
            a, b = scores[left], scores[right]
            union = (a | b).sum()
            frame.loc[left, right] = float((a & b).sum() / union) if union else 1.0
    return frame


__all__ = [
    "DETECTOR_FEATURES",
    "AnomalyEnsemble",
    "detector_agreement",
    "fit_detectors",
    "score_listings",
]
