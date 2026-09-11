"""Construction of the model-ready feature matrix.

``laptop_price.features.schema`` is deliberately *not* imported here: it depends
on pandera, which is a quality-gate dependency rather than a modelling one.
Import it explicitly where validation is wanted.
"""

from laptop_price.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_matrix,
    spec_signature,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "TARGET",
    "build_feature_matrix",
    "spec_signature",
]
