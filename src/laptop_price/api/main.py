"""FastAPI prediction service.

Run locally::

    uvicorn laptop_price.api.main:app --reload

or via ``make serve``. Interactive docs at ``/docs``.

Design note: the service refuses to start answering predictions if no trained
bundle is present, and ``/schema`` publishes the exact feature contract the
loaded pipeline was fitted on. That is deliberate - the failure this project
previously shipped was a scaler and a model with mismatched feature counts, which
produced confident nonsense rather than an error.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from laptop_price import __version__
from laptop_price.api.schemas import (
    DealResponse,
    HealthResponse,
    ListingRequest,
    PredictionResponse,
    SchemaResponse,
)

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"bundle": None, "error": None}


def _load_model() -> None:
    from laptop_price.models.registry import load_bundle

    try:
        _STATE["bundle"] = load_bundle()
        _STATE["error"] = None
        logger.info("loaded model %s", _STATE["bundle"].version)
    except FileNotFoundError as exc:
        # Start anyway so /health can report *why* the service is not ready.
        _STATE["bundle"] = None
        _STATE["error"] = str(exc)
        logger.warning("no model available: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _STATE.clear()


app = FastAPI(
    title="Laptop Price Intelligence API",
    description=(
        "Estimates the asking price of a used laptop on the Algerian market. "
        "Returns a range rather than a single number: identical configurations "
        "in the training data sell between 2x and 9x apart."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_bundle():
    bundle = _STATE.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=_STATE.get("error") or "no trained model is loaded; run `make train`",
        )
    return bundle


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness and readiness in one. Used by the container healthcheck."""
    bundle = _STATE.get("bundle")
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
        package_version=__version__,
    )


@app.get("/schema", response_model=SchemaResponse, tags=["meta"])
def feature_schema() -> SchemaResponse:
    """The feature contract of the loaded model, straight from its metadata."""
    bundle = _require_bundle()
    meta = bundle.metadata
    return SchemaResponse(
        model_version=bundle.version,
        target=meta.get("target", "price_corrected"),
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        trained_at=meta.get("trained_at"),
        git_sha=meta.get("git_sha"),
        metrics=meta.get("metrics", {}).get("headline_time_based"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    listing: ListingRequest,
    explain: bool = Query(False, description="include per-feature SHAP contributions"),
) -> PredictionResponse:
    """Estimate what a laptop would be listed for.

    The `range_dzd` field is the real answer; `estimate_dzd` is its midpoint in
    spirit and should not be quoted alone.
    """
    _require_bundle()
    from laptop_price.serving import explain_one, predict_one

    payload = listing.to_listing()
    result = predict_one(payload)

    if explain:
        try:
            result["contributions"] = explain_one(payload)
        except Exception as exc:  # pragma: no cover - SHAP is best-effort
            logger.warning("explanation failed: %s", exc)
            result["contributions"] = None

    return PredictionResponse(**result)


@app.get("/deals", response_model=list[DealResponse], tags=["anomaly"])
def deals(
    limit: int = Query(20, ge=1, le=200),
    min_discount: float = Query(35.0, ge=0, le=95, description="minimum discount, percent"),
) -> list[DealResponse]:
    """Listings priced well below what the model expects.

    Scam-shaped listings are filtered out first: a deep discount on an otherwise
    anomalous (specs, price) pair is far more likely to be a bait listing than a
    bargain, and a deal feed that cannot tell the two apart is worse than none.
    """
    bundle = _require_bundle()
    from laptop_price.anomaly.deals import rank_deals
    from laptop_price.data import load_model_ready
    from laptop_price.features.build import TARGET

    try:
        matrix = load_model_ready()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"feature matrix not built: {exc}") from exc

    ranked = rank_deals(matrix, bundle, top_k=limit * 3)
    ranked = ranked[ranked["discount_pct"] >= min_discount].head(limit)

    return [
        DealResponse(
            asking_price_dzd=float(row[TARGET]),
            predicted_dzd=float(row["predicted"]),
            discount_pct=float(row["discount_pct"]),
            verdict=str(row["verdict"]),
            brand=_opt_str(row.get("brand")),
            city=_opt_str(row.get("city_grouped")),
            ram_gb=_opt_float(row.get("RAM_SIZE")),
            ssd_gb=_opt_float(row.get("SSD_SIZE")),
            cpu_mark=_opt_float(row.get("cpu_mark")),
        )
        for _, row in ranked.iterrows()
    ]


def _opt_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def _opt_str(value: Any) -> str | None:
    if value is None or value != value:  # NaN
        return None
    return str(value)


@app.post("/reload", tags=["meta"])
def reload_model() -> dict[str, Any]:
    """Pick up a newly trained model without restarting the container."""
    _load_model()
    bundle = _STATE.get("bundle")
    return {
        "reloaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
        "error": _STATE.get("error"),
    }


__all__ = ["app"]
