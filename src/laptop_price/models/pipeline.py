"""One end-to-end sklearn Pipeline: imputation, encoding and the estimator.

Why one object
--------------
The shipped artifacts did not fit together::

    saved_models/scaler.pkl      -> expects 14 features
    saved_models/best_model.pkl  -> expects 10 features
    selected_features.json       -> lists 10

and the model had been trained on *unscaled* data, so the shipped scaler was
both the wrong shape and something the estimator had never seen. Anyone loading
the three files and following the obvious path (scale, then predict) got silent
garbage.

A single ``Pipeline`` makes that failure mode unrepresentable: there is one
object, it carries its own preprocessing, and train/inference skew cannot occur
because the same fitted transformers run in both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from laptop_price.config import CONFIG
from laptop_price.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor(
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Column-wise preprocessing.

    Numeric columns pass through untouched by default: the tree ensembles this
    project uses handle raw magnitudes and NaN natively, and *not* imputing is
    the point of the ``spec_Etat`` fix - a learned "missing" split is better than
    a fabricated value. ``scale_numeric=True`` adds median imputation and
    standardisation for the linear baselines, which need both.
    """
    if scale_numeric:
        numeric_steps = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
    else:
        numeric_steps = "passthrough"

    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            # An unseen city or brand at inference time must not raise.
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
            ),
        ]
    )

    transformer = ColumnTransformer(
        [
            ("num", numeric_steps, list(numeric)),
            ("cat", categorical_steps, list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # pandas output keeps feature names through the transform, which is what
    # lets monotonic constraints be declared by name (see monotonic_constraints).
    return transformer.set_output(transform="pandas")


#: Predictions are clamped to the band the model was trained on. Training trims
#: the target to [min_dzd, max_dzd], so anything outside it is extrapolation the
#: model has no evidence for.
_LOG_PREDICTION_FLOOR = float(np.log1p(CONFIG.price.min_dzd))
_LOG_PREDICTION_CEILING = float(np.log1p(CONFIG.price.max_dzd))


def _expm1_clipped(values: np.ndarray) -> np.ndarray:
    """Invert the log transform, clamped to the trained price band.

    Two things this prevents. A linear model extrapolating on a sparse region can
    emit a log prediction large enough that ``np.expm1`` overflows to inf, which
    poisons every aggregate metric computed from it. And a deployed pricing
    service quoting 40 million dinars for a laptop is worse than one quoting the
    ceiling - so the guardrail belongs in the artifact, not in the caller.
    """
    return np.expm1(np.clip(values, _LOG_PREDICTION_FLOOR, _LOG_PREDICTION_CEILING))


def monotonic_constraints() -> dict[str, int]:
    """``{feature: +1}`` for features where more must never mean cheaper.

    More RAM should not lower the predicted price. Beyond correctness this keeps
    the "what would raise the value" panel in the UI from producing an
    embarrassing recommendation.

    Returned as a dict keyed by feature name rather than a positional array,
    because the width of the transformed matrix depends on how many one-hot
    levels survive ``min_frequency`` and is therefore unknown until fit time.
    That only works if the preprocessor emits a DataFrame - see
    :func:`make_preprocessor`, which sets pandas output for exactly this reason.
    """
    return dict.fromkeys(CONFIG.model.monotonic_increasing, 1)


def build_pipeline(
    *,
    estimator: object | None = None,
    scale_numeric: bool = False,
    log_target: bool | None = None,
) -> Pipeline | TransformedTargetRegressor:
    """Assemble preprocessing + estimator into one fitted-together object.

    Parameters
    ----------
    estimator
        Defaults to :class:`HistGradientBoostingRegressor`, which handles NaN
        natively - the behaviour the ``spec_Etat`` fix depends on.
    log_target
        Wrap in a :class:`TransformedTargetRegressor` so callers always work in
        dinars; the log transform and its inverse live inside the artifact
        instead of in notebook cells that can be forgotten at inference time.
    """
    log_target = CONFIG.model.log_target if log_target is None else log_target

    if estimator is None:
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.06,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            monotonic_cst=monotonic_constraints(),
            # Explicit, not 'auto'. The default enables early stopping only
            # above 10,000 samples, so a 60% training split (9.7k rows) trained
            # for the full max_iter while the 80% refit (13k rows) early-stopped
            # AND held back another 10% internally - the shipped artifact was
            # trained differently from the model whose metrics were reported,
            # silently, as a function of split size. Regularisation comes from
            # l2_regularization and max_leaf_nodes instead.
            early_stopping=False,
            random_state=CONFIG.model.random_state,
        )

    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )

    if not log_target:
        return pipeline

    # np.log1p / _expm1_clipped rather than a fitted transformer: the mapping is
    # fixed, so it cannot drift between training and serving.
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=_expm1_clipped
    )


def build_quantile_pipelines(
    quantiles: tuple[float, ...] | None = None,
) -> dict[float, TransformedTargetRegressor | Pipeline]:
    """One pipeline per quantile, for the predicted *range*.

    Identical laptops in this dataset sell 2-9x apart, so a single number is the
    wrong output shape for the product. Three quantile models give
    "78,000 - 96,000 DZD, most likely ~86,000" at negligible extra cost.
    """
    quantiles = tuple(CONFIG.model.quantiles) if quantiles is None else quantiles

    pipelines: dict[float, TransformedTargetRegressor | Pipeline] = {}
    for q in quantiles:
        estimator = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=300,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=1.0,
            # The interval bounds must move the same way the point estimate
            # does; an upper bound that falls when RAM rises is indefensible.
            monotonic_cst=monotonic_constraints(),
            early_stopping=False,
            random_state=CONFIG.model.random_state,
        )
        pipelines[q] = build_pipeline(estimator=estimator)
    return pipelines


def make_catboost_preprocessor(
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Preprocessor for CatBoost: numerics pass through, categoricals get
    missing-value imputation only (no one-hot encoding, so CatBoost can use
    its native categorical handling instead of OHE which defeats the purpose).
    """
    categorical_steps = Pipeline(
        [("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN"))]
    )
    return ColumnTransformer(
        [
            ("num", "passthrough", list(numeric)),
            ("cat", categorical_steps, list(categorical)),
        ],
        remainder="drop",
    )


def build_catboost_pipeline(
    *,
    log_target: bool | None = None,
) -> Pipeline | TransformedTargetRegressor:
    """Pipeline using CatBoost's native categorical handling.

    CatBoost accepts raw string categoricals directly, so we bypass the OHE
    step and only impute missing values in categorical columns. The
    ``cat_features`` parameter is passed as column *indices* in the
    transformed output (numerics first, then categoricals).
    """
    from catboost import CatBoostRegressor  # optional dep; imported lazily

    log_target = CONFIG.model.log_target if log_target is None else log_target

    # Numeric columns come first in make_catboost_preprocessor output
    cat_indices = list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

    estimator = CatBoostRegressor(
        iterations=500,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=CONFIG.model.random_state,
        verbose=0,
        cat_features=cat_indices,
    )

    pipeline = Pipeline(
        [
            ("preprocess", make_catboost_preprocessor()),
            ("model", estimator),
        ]
    )

    if not log_target:
        return pipeline

    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=_expm1_clipped
    )


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the declared feature columns, in the declared order.

    Serving code calls this so that a caller passing extra or differently
    ordered columns cannot silently shift the feature matrix.
    """
    columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    return df[columns]


__all__ = [
    "build_pipeline",
    "build_catboost_pipeline",
    "monotonic_constraints",
    "build_quantile_pipelines",
    "feature_frame",
    "make_preprocessor",
    "make_catboost_preprocessor",
]
