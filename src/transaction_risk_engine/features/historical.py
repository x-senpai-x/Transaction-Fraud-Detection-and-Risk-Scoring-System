import numpy as np
import pandas as pd

from transaction_risk_engine.data.schema import PROXY_ID_COLUMNS

def build_historical_features(df: pd.DataFrame, entities: list[str] = None) -> pd.DataFrame:
    """
    Build leakage-aware historical aggregations.
    Must be computed chronologically using only past rows.
    """
    if entities is None:
        entities = PROXY_ID_COLUMNS

    df_sorted = df.copy()
    if "TransactionDT" in df_sorted.columns:
        # Assumes already sorted chronologically, but just in case
        df_sorted = df_sorted.sort_values("TransactionDT")

    new_features = {}

    for entity in entities:
        if entity not in df_sorted.columns:
            continue

        # Groupby with expanding requires sorting to be strictly chronological.
        # We shift(1) to avoid leaking the current transaction into its own history.
        grouped = df_sorted.groupby(entity)

        # 1. Previous transaction count
        new_features[f"{entity}_hist_count"] = grouped.cumcount()

        # 2. Previous mean amount
        if "TransactionAmt" in df_sorted.columns:
            mean_amt = grouped["TransactionAmt"].apply(lambda x: x.shift(1).expanding().mean())
            # Pandas 2.2 groupby apply returns a multiindex sometimes or maintains original index if not group keys
            # Let's use a safer approach for pandas 3/2.2:

            # Using groupby + expanding + shift can be tricky with indices.
            # Safe way:
            shifted_amt = grouped["TransactionAmt"].shift(1)
            new_features[f"{entity}_hist_mean_amt"] = df_sorted.assign(sh_amt=shifted_amt).groupby(entity)["sh_amt"].expanding().mean().reset_index(level=0, drop=True)

            # 3. Previous max amount
            new_features[f"{entity}_hist_max_amt"] = df_sorted.assign(sh_amt=shifted_amt).groupby(entity)["sh_amt"].expanding().max().reset_index(level=0, drop=True)

            # 4. Amount ratio to previous mean
            # Will be added later when we construct dataframe

        # 5. Time since previous transaction
        if "TransactionDT" in df_sorted.columns:
            shifted_time = grouped["TransactionDT"].shift(1)
            new_features[f"{entity}_time_since_last"] = df_sorted["TransactionDT"] - shifted_time

    # Combine into dataframe
    features_df = pd.DataFrame(new_features, index=df_sorted.index)

    # Compute the ratio features now that we have the means
    if "TransactionAmt" in df_sorted.columns:
        for entity in entities:
            if entity in df_sorted.columns and f"{entity}_hist_mean_amt" in features_df.columns:
                mean_col = features_df[f"{entity}_hist_mean_amt"]
                # Handle division by zero or NaN
                features_df[f"{entity}_amt_to_hist_mean"] = np.where(
                    mean_col > 0,
                    df_sorted["TransactionAmt"] / mean_col,
                    np.nan
                )

    # Restore original index order just in case sort_values changed it
    features_df = features_df.reindex(df.index)

    return pd.concat([df, features_df], axis=1)


def get_latest_historical_state(df: pd.DataFrame, entities: list[str] = None) -> dict[str, dict]:
    """
    Compute and save the latest historical state for each entity to be used during inference.
    """
    if entities is None:
        entities = PROXY_ID_COLUMNS

    df_sorted = df.copy()
    if "TransactionDT" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("TransactionDT")

    state = {}
    for entity in entities:
        if entity not in df_sorted.columns:
            continue

        state[entity] = {}

        # We only need the very last seen state for each entity
        grouped = df_sorted.groupby(entity)

        # 1. Count (total occurrences)
        counts = grouped.size()

        # 2. Latest mean amount
        if "TransactionAmt" in df_sorted.columns:
            means = grouped["TransactionAmt"].mean()
            maxes = grouped["TransactionAmt"].max()
        else:
            means = pd.Series(index=counts.index, dtype=float)
            maxes = pd.Series(index=counts.index, dtype=float)

        # 3. Latest transaction time
        if "TransactionDT" in df_sorted.columns:
            last_times = grouped["TransactionDT"].max()
        else:
            last_times = pd.Series(index=counts.index, dtype=float)

        # Combine into dict
        for uid in counts.index:
            state[entity][str(uid)] = {
                "count": int(counts[uid]),
                "mean_amt": float(means[uid]) if pd.notna(means[uid]) else None,
                "max_amt": float(maxes[uid]) if pd.notna(maxes[uid]) else None,
                "last_dt": float(last_times[uid]) if pd.notna(last_times[uid]) else None
            }

    return state

def apply_historical_features(df: pd.DataFrame, state: dict[str, dict], entities: list[str] = None) -> pd.DataFrame:
    """
    Apply historical features for inference using the pre-computed state.
    Assumes df is a single row (or very few rows) and we use the static latest state.
    """
    if entities is None:
        entities = PROXY_ID_COLUMNS

    new_features = {}

    for entity in entities:
        if entity not in df.columns:
            continue

        # Initialize lists
        counts = []
        mean_amts = []
        max_amts = []
        times_since = []
        ratios = []

        for idx, row in df.iterrows():
            uid = str(row[entity])
            entity_state = state.get(entity, {}).get(uid, None)

            if entity_state is None:
                counts.append(0)
                mean_amts.append(np.nan)
                max_amts.append(np.nan)
                times_since.append(np.nan)
                ratios.append(np.nan)
            else:
                counts.append(entity_state["count"])
                mean_amts.append(entity_state["mean_amt"])
                max_amts.append(entity_state["max_amt"])

                if "TransactionDT" in df.columns and entity_state["last_dt"] is not None:
                    times_since.append(row["TransactionDT"] - entity_state["last_dt"])
                else:
                    times_since.append(np.nan)

                if "TransactionAmt" in df.columns and entity_state["mean_amt"] is not None and entity_state["mean_amt"] > 0:
                    ratios.append(row["TransactionAmt"] / entity_state["mean_amt"])
                else:
                    ratios.append(np.nan)

        new_features[f"{entity}_hist_count"] = counts
        new_features[f"{entity}_hist_mean_amt"] = mean_amts
        new_features[f"{entity}_hist_max_amt"] = max_amts
        new_features[f"{entity}_time_since_last"] = times_since
        new_features[f"{entity}_amt_to_hist_mean"] = ratios

    features_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, features_df], axis=1)

import json
from pathlib import Path

def save_historical_state(state: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
