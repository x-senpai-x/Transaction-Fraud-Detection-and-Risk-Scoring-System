from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from transaction_risk_engine.data.schema import TIME_COLUMN


def time_based_split(
    df: pd.DataFrame, config: dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by TransactionDT and split into train/valid/test fractionally."""
    split_config = config.get("split", {})
    train_frac = split_config.get("train_frac", 0.70)
    valid_frac = split_config.get("valid_frac", 0.15)
    # test_frac is the remainder

    if TIME_COLUMN not in df.columns:
        raise ValueError(f"Time column {TIME_COLUMN} not found in dataframe")

    # Sort chronologically
    df_sorted = df.sort_values(by=TIME_COLUMN).reset_index(drop=True)

    n = len(df_sorted)
    train_idx = int(n * train_frac)
    valid_idx = train_idx + int(n * valid_frac)

    train_df = df_sorted.iloc[:train_idx].copy()
    valid_df = df_sorted.iloc[train_idx:valid_idx].copy()
    test_df = df_sorted.iloc[valid_idx:].copy()

    return train_df, valid_df, test_df


def save_splits(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "split_train.parquet", index=False)
    valid.to_parquet(output_dir / "split_valid.parquet", index=False)
    test.to_parquet(output_dir / "split_test.parquet", index=False)
