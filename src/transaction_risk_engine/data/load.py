import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from transaction_risk_engine.config import load_config
from transaction_risk_engine.data.schema import (
    PROXY_ID_COMPONENTS,
    TARGET,
    TIME_COLUMN,
    TRANSACTION_ID,
)
from transaction_risk_engine.utils.paths import DataPaths


def resolve_data_paths(config: dict[str, Any]) -> DataPaths:
    data_config = config.get("data", {})
    raw_dir = Path(data_config.get("raw_dir", "data/raw"))
    interim_dir = Path(data_config.get("interim_dir", "data/interim"))
    processed_dir = Path(data_config.get("processed_dir", "data/processed"))

    transaction_file = raw_dir / data_config.get("transaction_file", "train_transaction.csv")
    identity_file = raw_dir / data_config.get("identity_file", "train_identity.csv")
    joined_file = interim_dir / data_config.get("joined_file", "joined_transactions.parquet")

    reports_dir = Path(config.get("reports", {}).get("output_dir", "reports"))

    return DataPaths(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        transaction_file=transaction_file,
        identity_file=identity_file,
        joined_file=joined_file,
        reports_dir=reports_dir,
    )


def load_transaction_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")
    return pd.read_csv(path)


def load_identity_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Identity file not found: {path}")
    return pd.read_csv(path)


def standardize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize identity column names by replacing hyphens with underscores."""
    frame.columns = [str(col).replace("-", "_") for col in frame.columns]
    return frame


def join_transaction_identity(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left join identity onto transaction by TransactionID."""
    return transactions.merge(identity, on=TRANSACTION_ID, how="left")


def add_proxy_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Create proxy IDs: card_uid, address_uid, email_uid, receiver_email_uid, device_uid."""
    for proxy_col, components in PROXY_ID_COMPONENTS.items():
        # Make sure all component columns exist in the frame, if not create them with missing values
        for c in components:
            if c not in frame.columns:
                frame[c] = pd.NA
        frame[proxy_col] = frame[components].fillna("UNKNOWN").astype(str).agg("_".join, axis=1)
    return frame


def add_relative_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add relative time columns derived from TransactionDT."""
    if TIME_COLUMN in frame.columns:
        frame["relative_day"] = frame[TIME_COLUMN] // (24 * 60 * 60)
        frame["relative_hour"] = (frame[TIME_COLUMN] // (60 * 60)) % 24
    return frame


def _format_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    # Reset index to make it a column for the table
    df_to_print = df.reset_index()
    headers = list(df_to_print.columns)
    
    # Create the markdown header and separator
    markdown_lines = []
    markdown_lines.append("| " + " | ".join([str(h) for h in headers]) + " |")
    markdown_lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    
    # Add the rows
    for _, row in df_to_print.iterrows():
        markdown_lines.append("| " + " | ".join([str(x) for x in row.values]) + " |")
    
    return "\n".join(markdown_lines)


def write_data_profile(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shape = frame.shape
    target_count = int(frame[TARGET].sum()) if TARGET in frame.columns else 0
    target_rate = frame[TARGET].mean() if TARGET in frame.columns else 0

    id_match_col = "id_01"
    id_match_count = int(frame[id_match_col].notna().sum()) if id_match_col in frame.columns else 0
    id_match_rate = id_match_count / shape[0] if shape[0] > 0 else 0

    missingness = frame.isna().mean().sort_values(ascending=False).head(20).to_frame(name="Missing Rate")

    cat_cols = frame.select_dtypes(include=["object", "category"]).columns
    top_cats = {}
    for col in cat_cols[:5]:
        top_cats[col] = frame[col].value_counts(dropna=False).head(5).to_dict()

    num_summary = frame.describe().T.head(20)

    dt_range = (frame[TIME_COLUMN].min(), frame[TIME_COLUMN].max()) if TIME_COLUMN in frame.columns else (None, None)

    with open(output_path, "w") as f:
        f.write("# Data Profile\n\n")
        f.write(f"**Shape:** {shape[0]} rows, {shape[1]} columns\n\n")
        f.write(f"**Target ({TARGET}):** Count = {target_count}, Rate = {target_rate:.4%}\n\n")
        f.write(f"**Identity Match Coverage:** Count = {id_match_count}, Rate = {id_match_rate:.4%}\n\n")
        f.write(f"**TransactionDT Range:** {dt_range[0]} to {dt_range[1]}\n\n")

        f.write("## Missingness (Top 20)\n\n")
        f.write(_format_markdown_table(missingness) + "\n\n")

        f.write("## Numeric Summary (Top 20)\n\n")
        f.write(_format_markdown_table(num_summary) + "\n\n")

        f.write("## Top Categorical Counts (Sample)\n\n")
        for col, counts in top_cats.items():
            f.write(f"### {col}\n")
            for val, count in counts.items():
                f.write(f"- {val}: {count}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_data_paths(config)

    print(f"Loading transaction data from {paths.transaction_file}...")
    transactions = load_transaction_data(paths.transaction_file)
    
    print(f"Loading identity data from {paths.identity_file}...")
    identity = load_identity_data(paths.identity_file)

    print("Standardizing column names...")
    identity = standardize_column_names(identity)

    print("Joining transaction and identity data...")
    joined = join_transaction_identity(transactions, identity)

    print("Adding proxy IDs...")
    joined = add_proxy_ids(joined)

    print("Adding relative time columns...")
    joined = add_relative_time_columns(joined)

    print(f"Writing joined data to {paths.joined_file}...")
    write_joined_data(joined, paths.joined_file)

    profile_path = paths.reports_dir / "data_profile.md"
    print(f"Writing data profile to {profile_path}...")
    write_data_profile(joined, profile_path)

    print("Done!")


def write_joined_data(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
