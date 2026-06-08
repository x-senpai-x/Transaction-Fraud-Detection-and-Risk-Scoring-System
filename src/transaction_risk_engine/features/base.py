import numpy as np
import pandas as pd

from transaction_risk_engine.data.schema import LABEL_ENCODE_COLUMNS, MISSINGNESS_COLUMNS, M_COLUMNS


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Block A basic features without leakage risk."""
    features = {}

    # Amount features
    if "TransactionAmt" in df.columns:
        features["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        features["TransactionAmt_decimal"] = df["TransactionAmt"] - df["TransactionAmt"].apply(np.floor)

    # Time features (using relative time generated in Phase 1)
    if "relative_hour" in df.columns:
        features["hour_of_day"] = df["relative_hour"]
    if "relative_day" in df.columns:
        features["day_number"] = df["relative_day"]

    # Missingness indicators
    for col in MISSINGNESS_COLUMNS:
        if col in df.columns:
            features[f"{col}_missing"] = df[col].isna().astype(int)

    # Identity presence
    if "id_01" in df.columns:
        features["has_identity"] = df["id_01"].notna().astype(int)

    # M columns: boolean conversion
    for col in M_COLUMNS:
        if col in df.columns:
            features[f"{col}_missing"] = df[col].isna().astype(int)
            features[f"{col}_encoded"] = df[col].map({"T": 1, "F": 0, "nan": np.nan, "UNKNOWN": np.nan})

    # Label Encoding for specific categoricals
    for col in LABEL_ENCODE_COLUMNS:
        if col in df.columns:
            features[f"{col}_encoded"] = df[col].astype("category").cat.codes

    new_features_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, new_features_df], axis=1)
