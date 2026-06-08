import argparse
import json
from pathlib import Path

import pandas as pd

from transaction_risk_engine.config import load_config
from transaction_risk_engine.data.schema import FREQUENCY_COLUMNS, TARGET, TRANSACTION_ID
from transaction_risk_engine.data.split import time_based_split
from transaction_risk_engine.features.base import build_base_features
from transaction_risk_engine.features.frequency import (
    fit_frequency_encoder,
    save_frequency_maps,
    transform_frequency,
)
from transaction_risk_engine.features.historical import build_historical_features, get_latest_historical_state, save_historical_state
from transaction_risk_engine.data.schema import PROXY_ID_COLUMNS
from transaction_risk_engine.data.load import resolve_data_paths


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the numeric features, dropping raw strings, targets, and IDs."""
    drop_cols = [TARGET, TRANSACTION_ID, "TransactionDT", "relative_day", "relative_hour"]
    out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    out = out.select_dtypes(exclude=["object", "category", "string"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_data_paths(config)

    print(f"Loading interim data from {paths.joined_file}...")
    df = pd.read_parquet(paths.joined_file)

    print("Splitting data chronologically...")
    train, valid, test = time_based_split(df, config)

    print("Building Block A (basic features)...")
    train = build_base_features(train)
    valid = build_base_features(valid)
    test = build_base_features(test)

    print("Building Block B (frequency features)...")
    freq_maps = fit_frequency_encoder(train, FREQUENCY_COLUMNS)
    train = transform_frequency(train, freq_maps)
    valid = transform_frequency(valid, freq_maps)
    test = transform_frequency(test, freq_maps)

    print("Building Block C (historical features)...")
    train = build_historical_features(train, PROXY_ID_COLUMNS)
    valid = build_historical_features(valid, PROXY_ID_COLUMNS)
    test = build_historical_features(test, PROXY_ID_COLUMNS)

    print("Extracting Historical State for inference...")
    hist_state = get_latest_historical_state(train, PROXY_ID_COLUMNS)

    print("Extracting feature matrices...")
    X_train = get_feature_matrix(train)
    X_valid = get_feature_matrix(valid)
    X_test = get_feature_matrix(test)

    print(f"Saving features to {paths.processed_dir}...")
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(paths.processed_dir / "features_train.parquet", index=False)
    X_valid.to_parquet(paths.processed_dir / "features_valid.parquet", index=False)
    X_test.to_parquet(paths.processed_dir / "features_test.parquet", index=False)

    train[[TARGET]].to_parquet(paths.processed_dir / "target_train.parquet", index=False)
    valid[[TARGET]].to_parquet(paths.processed_dir / "target_valid.parquet", index=False)
    test[[TARGET]].to_parquet(paths.processed_dir / "target_test.parquet", index=False)

    metadata_path = Path("models/feature_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "features": list(X_train.columns),
        "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    save_frequency_maps(freq_maps, Path("models/frequency_maps.json"))
    save_historical_state(hist_state, Path("models/historical_state.json"))

    report_path = paths.reports_dir / "feature_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Feature Report\n\n")
        f.write(f"- Train size: {len(X_train)} (Fraud rate: {train[TARGET].mean():.4%})\n")
        f.write(f"- Valid size: {len(X_valid)} (Fraud rate: {valid[TARGET].mean():.4%})\n")
        f.write(f"- Test size: {len(X_test)} (Fraud rate: {test[TARGET].mean():.4%})\n")
        f.write(f"- Total features: {len(X_train.columns)}\n\n")

    print("Done!")


if __name__ == "__main__":
    main()
