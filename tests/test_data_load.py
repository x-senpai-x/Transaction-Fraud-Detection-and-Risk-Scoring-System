from pathlib import Path

import pandas as pd
import pytest

from transaction_risk_engine.data.load import (
    add_proxy_ids,
    add_relative_time_columns,
    join_transaction_identity,
    load_identity_data,
    load_transaction_data,
    standardize_column_names,
    write_data_profile,
)
from transaction_risk_engine.data.schema import TARGET, TRANSACTION_ID


def test_missing_transaction_file_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_transaction_data(tmp_path / "missing.csv")


def test_missing_identity_file_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_identity_data(tmp_path / "missing.csv")


def test_join_preserves_all_transaction_rows() -> None:
    tx = pd.DataFrame({TRANSACTION_ID: [1, 2, 3], TARGET: [0, 1, 0]})
    idx = pd.DataFrame({TRANSACTION_ID: [2], "id_01": [99]})
    joined = join_transaction_identity(tx, idx)
    assert len(joined) == 3
    assert joined[TARGET].tolist() == [0, 1, 0]


def test_join_works_without_identity_row() -> None:
    tx = pd.DataFrame({TRANSACTION_ID: [1], TARGET: [0]})
    idx = pd.DataFrame(columns=[TRANSACTION_ID, "id_01"])
    joined = join_transaction_identity(tx, idx)
    assert len(joined) == 1
    assert joined[TARGET].tolist() == [0]


def test_standardize_column_names() -> None:
    df = pd.DataFrame({"id-01": [1], "id-31": [2], "DeviceType": ["desktop"]})
    std = standardize_column_names(df)
    assert "id_01" in std.columns
    assert "id_31" in std.columns
    assert "DeviceType" in std.columns
    assert "id-01" not in std.columns


def test_proxy_id_creation() -> None:
    df = pd.DataFrame({
        "card1": [1, 1], "card2": [2, None], "card3": [3, 3], "card5": [5, 5], "card6": ["credit", "debit"],
        "addr1": [100, None], "addr2": [87, 87],
        "P_emaildomain": ["gmail.com", None],
        "R_emaildomain": ["yahoo.com", "yahoo.com"],
        "DeviceType": ["mobile", None], "DeviceInfo": ["iOS", None], "id_30": ["Mac", None], "id_31": ["Safari", None]
    })
    proxied = add_proxy_ids(df)
    assert "card_uid" in proxied.columns
    assert "address_uid" in proxied.columns
    assert "email_uid" in proxied.columns
    assert "receiver_email_uid" in proxied.columns
    assert "device_uid" in proxied.columns

    # Check deterministic handling of NA
    assert "nan" not in proxied["card_uid"].iloc[1].lower() # We use "UNKNOWN"
    assert "UNKNOWN" in proxied["card_uid"].iloc[1]


def test_relative_time_columns() -> None:
    df = pd.DataFrame({"TransactionDT": [86400, 86400 + 3600]})
    timed = add_relative_time_columns(df)
    assert "relative_day" in timed.columns
    assert "relative_hour" in timed.columns
    assert timed["relative_day"].tolist() == [1, 1]
    assert timed["relative_hour"].tolist() == [0, 1]


def test_data_profile_generation(tmp_path: Path) -> None:
    df = pd.DataFrame({
        TRANSACTION_ID: [1, 2],
        TARGET: [0, 1],
        "TransactionDT": [1000, 2000],
        "id_01": [10, None],
        "card1": [100, 200]
    })
    out_path = tmp_path / "profile.md"
    write_data_profile(df, out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "Shape:" in content
    assert "Rate = 50.0" in content
    assert "Identity Match Coverage" in content
    assert "Missingness" in content
    assert "Numeric Summary" in content
