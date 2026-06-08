from pathlib import Path

import pytest

from transaction_risk_engine.config import get_seed, load_config


def test_load_config_reads_project_name() -> None:
    config = load_config("config/config.yaml")

    assert config["project"]["name"] == "transaction-risk-engine"


def test_get_seed_returns_integer_seed() -> None:
    config = load_config("config/config.yaml")

    assert get_seed(config) == 42


def test_load_config_fails_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)
