import json
from pathlib import Path

import pandas as pd


def fit_frequency_encoder(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Fit frequency encoder on train split only to avoid leakage."""
    freq_maps = {}
    for col in columns:
        if col in train_df.columns:
            # We map the exact counts. For highly missing data, nan gets its own count.
            freq_maps[col] = train_df[col].value_counts(dropna=False).to_dict()
    return freq_maps


def transform_frequency(df: pd.DataFrame, freq_maps: dict[str, dict]) -> pd.DataFrame:
    """Transform data using fitted frequency maps. Unseen gets 0."""
    features = {}
    for col, mapping in freq_maps.items():
        if col in df.columns:
            features[f"{col}_freq"] = df[col].map(mapping).fillna(0).astype(int)
    
    new_features = pd.DataFrame(features, index=df.index)
    return pd.concat([df, new_features], axis=1)


def save_frequency_maps(freq_maps: dict[str, dict], path: Path) -> None:
    """Serialize the frequency maps for inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(freq_maps, f, indent=2)
