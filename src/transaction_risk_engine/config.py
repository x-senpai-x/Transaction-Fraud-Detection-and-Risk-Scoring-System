"""Configuration loading helpers for Transaction Risk Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML config file and fail loudly when it is missing or invalid."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return loaded


def get_seed(config: dict[str, Any]) -> int:
    """Return the deterministic project seed from config."""
    try:
        seed = config["project"]["seed"]
    except KeyError as exc:
        raise KeyError("Missing required config key: project.seed") from exc

    if not isinstance(seed, int):
        raise TypeError("Config key project.seed must be an integer")

    return seed
