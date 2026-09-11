"""Versioned model artifacts.

Every training run writes a self-describing directory::

    models/
      v20260911T0214/
        price_pipeline.joblib      the single end-to-end Pipeline
        quantile_pipelines.joblib  the 10th/50th/90th percentile models
        metadata.json              feature contract, metrics, git SHA, timestamps
        model_card.md              intended use and limitations
      latest -> v20260911T0214     (a plain text pointer, not a symlink)

``metadata.json`` carries the feature contract, so the API can refuse to serve a
request whose columns do not match what the pipeline was fitted on, instead of
quietly producing a number. That is the failure the previous artifacts had.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from laptop_price import __version__, paths

LATEST_POINTER = "latest"
PIPELINE_FILE = "price_pipeline.joblib"
QUANTILE_FILE = "quantile_pipelines.joblib"
METADATA_FILE = "metadata.json"
MODEL_CARD_FILE = "model_card.md"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class ArtifactBundle:
    """A trained pipeline plus everything needed to serve and audit it."""

    pipeline: Any
    quantile_pipelines: dict[float, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    @property
    def numeric_features(self) -> list[str]:
        return list(self.metadata.get("features", {}).get("numeric", []))

    @property
    def categorical_features(self) -> list[str]:
        return list(self.metadata.get("features", {}).get("categorical", []))

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


@dataclass
class TrainingMetadata:
    """Everything a reader needs to reproduce or distrust a model."""

    version: str
    trained_at: str
    git_sha: str
    package_version: str
    python_version: str
    sklearn_version: str
    features: dict[str, list[str]]
    target: str
    n_train: int
    n_test: int
    split: str
    metrics: dict[str, Any]
    baselines: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_version() -> str:
    """A sortable version string: ``v20260911T0214``."""
    return datetime.now(UTC).strftime("v%Y%m%dT%H%M")


def build_metadata(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
    n_train: int,
    n_test: int,
    split: str,
    metrics: dict[str, Any],
    baselines: list[dict[str, Any]] | None = None,
    notes: str = "",
    version: str | None = None,
) -> dict[str, Any]:
    """Assemble the metadata record written beside the pipeline."""
    return TrainingMetadata(
        version=version or new_version(),
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        package_version=__version__,
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        features={"numeric": numeric_features, "categorical": categorical_features},
        target=target,
        n_train=n_train,
        n_test=n_test,
        split=split,
        metrics=metrics,
        baselines=baselines or [],
        notes=notes,
    ).to_dict()


def save_bundle(
    bundle: ArtifactBundle,
    *,
    models_dir: Path | None = None,
    model_card: str | None = None,
    mark_latest: bool = True,
) -> Path:
    """Write a bundle to ``models/<version>/`` and update the ``latest`` pointer."""
    root = models_dir or paths.MODELS_DIR
    version = bundle.metadata.get("version") or new_version()
    bundle.metadata["version"] = version

    target_dir = root / version
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.pipeline, target_dir / PIPELINE_FILE)
    if bundle.quantile_pipelines:
        joblib.dump(bundle.quantile_pipelines, target_dir / QUANTILE_FILE)

    (target_dir / METADATA_FILE).write_text(json.dumps(bundle.metadata, indent=2) + "\n")
    if model_card:
        (target_dir / MODEL_CARD_FILE).write_text(model_card)

    if mark_latest:
        (root / LATEST_POINTER).write_text(version + "\n")

    return target_dir


def latest_version(models_dir: Path | None = None) -> str | None:
    """Resolve the ``latest`` pointer, falling back to the newest directory."""
    root = models_dir or paths.MODELS_DIR
    pointer = root / LATEST_POINTER
    if pointer.exists():
        version = pointer.read_text().strip()
        if (root / version / PIPELINE_FILE).exists():
            return version

    candidates = sorted(
        (d.name for d in root.glob("v*") if (d / PIPELINE_FILE).exists()), reverse=True
    )
    return candidates[0] if candidates else None


def load_bundle(version: str | None = None, *, models_dir: Path | None = None) -> ArtifactBundle:
    """Load a trained bundle. Defaults to the version the ``latest`` pointer names.

    Raises
    ------
    FileNotFoundError
        With an actionable message - the API surfaces this rather than starting
        up and returning nonsense.
    """
    root = models_dir or paths.MODELS_DIR
    version = version or latest_version(root)
    if version is None:
        raise FileNotFoundError(
            f"no trained model found under {root}. Run `make train` (or "
            "`python -m laptop_price.cli train`) first."
        )

    directory = root / version
    pipeline_path = directory / PIPELINE_FILE
    if not pipeline_path.exists():
        raise FileNotFoundError(f"{pipeline_path} does not exist")

    quantiles: dict[float, Any] = {}
    quantile_path = directory / QUANTILE_FILE
    if quantile_path.exists():
        quantiles = joblib.load(quantile_path)

    metadata: dict[str, Any] = {}
    metadata_path = directory / METADATA_FILE
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return ArtifactBundle(
        pipeline=joblib.load(pipeline_path),
        quantile_pipelines=quantiles,
        metadata=metadata,
    )


__all__ = [
    "ArtifactBundle",
    "TrainingMetadata",
    "build_metadata",
    "latest_version",
    "load_bundle",
    "new_version",
    "save_bundle",
]
