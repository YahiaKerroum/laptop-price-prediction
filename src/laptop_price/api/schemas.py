"""Request and response models for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ListingRequest(BaseModel):
    """A laptop described by whatever the caller knows.

    Every field is optional on purpose. The model was trained to split on
    missingness rather than on imputed placeholders, so omitting a field you do
    not know is strictly better than guessing it.
    """

    cpu_name: str | None = Field(
        None,
        max_length=120,
        description=(
            "Processor, as listed by /catalog. Fills benchmarks, family and build "
            "cost, and roughly halves the width of the range"
        ),
    )
    gpu_name: str | None = Field(
        None, max_length=120, description="Dedicated GPU, as listed by /catalog; omit if integrated"
    )
    ram_gb: float | None = Field(None, ge=1, le=128, description="Installed RAM in GB")
    ssd_gb: float | None = Field(None, ge=0, le=16384, description="SSD capacity in GB")
    hdd_gb: float | None = Field(None, ge=0, le=16384, description="HDD capacity in GB")
    cpu_mark: float | None = Field(None, ge=0, description="PassMark CPU score")
    gpu_g3d_mark: float | None = Field(None, ge=0, description="PassMark G3D score")
    gpu_tdp: float | None = Field(None, ge=0, description="GPU TDP in watts")
    tdp: float | None = Field(None, ge=0, description="CPU TDP in watts")
    cores: float | None = Field(None, ge=1, le=128)
    screen_size: float | None = Field(None, ge=10, le=20, description="Diagonal in inches")
    resolution: float | None = Field(
        None, ge=1, le=9, description="Resolution tier: 1=HD, 3=FHD, 5=QHD, 8=4K"
    )
    condition: float | None = Field(
        None, ge=1, le=3, description="1=MOYEN, 2=BON ETAT, 3=JAMAIS UTILISE; omit if unknown"
    )
    ram_type: float | None = Field(None, ge=0, le=5, description="DDR generation, ordinal")
    brand: str = Field("UNKNOWN", description="Model family, e.g. THINKPAD")
    city: str = Field("UNKNOWN", description="Listing city")
    listing_year: float | None = Field(None, ge=2010, le=2035)
    listing_month: float | None = Field(None, ge=1, le=12)

    def to_listing(self) -> dict[str, Any]:
        """Map onto the column names the serving layer expects."""
        return {
            "cpu_name": self.cpu_name,
            "gpu_name": self.gpu_name,
            "RAM_SIZE": self.ram_gb,
            "SSD_SIZE": self.ssd_gb,
            "HDD_SIZE": self.hdd_gb,
            "cpu_mark": self.cpu_mark,
            "gpu_g3d_mark": self.gpu_g3d_mark,
            "gpu_tdp": self.gpu_tdp,
            "tdp": self.tdp,
            "cores": self.cores,
            "SCREEN_SIZE_SNAPPED": self.screen_size,
            "SCREEN_RESOLUTION_ENC": self.resolution,
            "spec_Etat": self.condition,
            "RAM_TYPE": self.ram_type,
            "brand": self.brand,
            "city_grouped": self.city,
            "listing_year": self.listing_year,
            "listing_month": self.listing_month,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "ram_gb": 16,
                "ssd_gb": 512,
                "cpu_mark": 19776,
                "gpu_g3d_mark": 16758,
                "screen_size": 15.6,
                "resolution": 3,
                "condition": 2,
                "brand": "THINKPAD",
                "city": "ALGER CENTRE",
                "listing_year": 2025,
            }
        }
    }


class Contribution(BaseModel):
    """One feature's signed contribution to the estimate."""

    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    """The estimate, as a range."""

    estimate_dzd: float = Field(..., description="Point estimate, DZD")
    range_dzd: list[float] | None = Field(
        None,
        description=(
            "Likely range: a calibrated band that about half of comparable listings "
            "fall inside. Read this, not the point estimate"
        ),
    )
    range_coverage: float | None = Field(
        None, description="Share of held-out listings that fell inside range_dzd"
    )
    wide_range_dzd: list[float] | None = Field(
        None, description="Raw 10th-90th percentile band; wider and less precise"
    )
    precision: str | None = Field(
        None, description="'high' when the CPU was identified, 'low' when it was not"
    )
    quantiles: dict[str, float] | None = None
    model_version: str
    currency: str = "DZD"
    caveat: str
    contributions: list[Contribution] | None = Field(
        None, description="Top SHAP contributions, when explain=true"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    package_version: str


class SchemaResponse(BaseModel):
    """The feature contract the loaded pipeline was fitted on."""

    model_version: str
    target: str
    numeric_features: list[str]
    categorical_features: list[str]
    trained_at: str | None = None
    git_sha: str | None = None
    metrics: dict[str, Any] | None = None


class DealResponse(BaseModel):
    """One candidate bargain."""

    asking_price_dzd: float
    predicted_dzd: float
    discount_pct: float
    verdict: str
    brand: str | None = None
    city: str | None = None
    ram_gb: float | None = None
    ssd_gb: float | None = None
    cpu_mark: float | None = None


class SimilarListing(BaseModel):
    """One nearest-neighbour result."""

    rank: int
    asking_price_dzd: float
    predicted_dzd: float | None = None
    distance: float
    brand: str | None = None
    city: str | None = None
    ram_gb: float | None = None
    ssd_gb: float | None = None
    cpu_mark: float | None = None
    gpu_g3d_mark: float | None = None
    cpu_family: str | None = None
    listing_year: float | None = None


class MarketStatsResponse(BaseModel):
    total_listings: int
    median_price: float
    mean_price: float
    price_std: float
    brands: int
    price_by_brand: dict[str, float]


class AnomalyCheckResponse(BaseModel):
    anomaly_score: float = Field(..., description="Higher = more anomalous (0–1 scale)")
    is_anomalous: bool
    reasons: list[str]


class CatalogResponse(BaseModel):
    """Values a front end can offer in its pickers, most common first."""

    cpus: list[dict[str, Any]]
    gpus: list[dict[str, Any]]
    brands: list[str]
    cities: list[str]


class BatchRequest(BaseModel):
    listings: list[ListingRequest]


class BatchPredictionItem(BaseModel):
    index: int
    estimate_dzd: float | None = None
    range_dzd: list[float] | None = None
    wide_range_dzd: list[float] | None = None
    model_version: str
    error: str | None = None


__all__ = [
    "AnomalyCheckResponse",
    "BatchPredictionItem",
    "BatchRequest",
    "CatalogResponse",
    "Contribution",
    "DealResponse",
    "HealthResponse",
    "ListingRequest",
    "MarketStatsResponse",
    "PredictionResponse",
    "SchemaResponse",
    "SimilarListing",
]
