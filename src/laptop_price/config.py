"""Configuration loading.

Values live in ``config.yaml`` next to this module. Point ``$LAPTOP_PRICE_CONFIG``
at another YAML file to override any subset of them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config(dict):
    """A dict that also supports attribute access, recursively.

    ``CONFIG.price.min_dzd`` and ``CONFIG["price"]["min_dzd"]`` are equivalent.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({dict.__repr__(self)})"


def load_config(path: str | Path | None = None) -> Config:
    """Load the default config, then merge in an override file if one is set."""
    with open(DEFAULT_CONFIG_PATH) as handle:
        data: dict[str, Any] = yaml.safe_load(handle)

    override_path = path or os.environ.get("LAPTOP_PRICE_CONFIG")
    if override_path:
        with open(override_path) as handle:
            data = _deep_merge(data, yaml.safe_load(handle) or {})

    return Config(data)


CONFIG = load_config()

__all__ = ["CONFIG", "Config", "load_config", "DEFAULT_CONFIG_PATH"]
