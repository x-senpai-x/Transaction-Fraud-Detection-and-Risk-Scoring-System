import json
from pathlib import Path

import pandas as pd

from transaction_risk_engine.data.schema import TARGET


def main():
    interim_file = Path("data/interim/joined_transactions.parquet")
    sample_file = Path("data/sample/sample_transactions.json")

    if not interim_file.exists():
        print(f"Interim file {interim_file} not found. Run Phase 1 first.")
        return

    df = pd.read_parquet(interim_file)

    # Need roughly 10 fraud and 10 non-fraud
    df_fraud = df[df[TARGET] == 1].sample(10, random_state=42)
    df_legit = df[df[TARGET] == 0].sample(10, random_state=42)

    df_sample = pd.concat([df_legit, df_fraud])
    
    # Fill NAs with None so it's valid JSON (pandas to_dict("records") sometimes leaves NaNs)
    # Actually, replacing NaNs with None is standard for API JSON payloads.
    df_sample = df_sample.where(pd.notnull(df_sample), None)

    records = df_sample.to_dict(orient="records")

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} samples to {sample_file}")


if __name__ == "__main__":
    main()
