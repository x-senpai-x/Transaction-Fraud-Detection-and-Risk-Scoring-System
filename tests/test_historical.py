import pandas as pd
import numpy as np
from transaction_risk_engine.features.historical import build_historical_features

def test_historical_features_chronological():
    df = pd.DataFrame({
        "TransactionDT": [1, 2, 3, 4],
        "TransactionAmt": [10, 20, 30, 40],
        "card_uid": ["A", "A", "B", "A"],
    })

    res = build_historical_features(df, ["card_uid"])

    # Check counts
    assert res["card_uid_hist_count"].tolist() == [0, 1, 0, 2]

    # Check historical mean (shifted)
    assert pd.isna(res.loc[0, "card_uid_hist_mean_amt"])
    assert res.loc[1, "card_uid_hist_mean_amt"] == 10.0
    assert pd.isna(res.loc[2, "card_uid_hist_mean_amt"])
    assert res.loc[3, "card_uid_hist_mean_amt"] == 15.0  # (10+20)/2

    # Check time since last
    assert pd.isna(res.loc[0, "card_uid_time_since_last"])
    assert res.loc[1, "card_uid_time_since_last"] == 1  # 2 - 1
    assert res.loc[3, "card_uid_time_since_last"] == 2  # 4 - 2


from transaction_risk_engine.features.historical import get_latest_historical_state, apply_historical_features

def test_inference_historical_features():
    df_train = pd.DataFrame({
        "TransactionDT": [1, 2, 3],
        "TransactionAmt": [10, 20, 30],
        "card_uid": ["A", "A", "B"],
    })

    state = get_latest_historical_state(df_train, ["card_uid"])

    assert state["card_uid"]["A"]["count"] == 2
    assert state["card_uid"]["A"]["mean_amt"] == 15.0
    assert state["card_uid"]["A"]["last_dt"] == 2.0

    df_inf = pd.DataFrame({
        "TransactionDT": [4],
        "TransactionAmt": [30],
        "card_uid": ["A"],
    })

    res = apply_historical_features(df_inf, state, ["card_uid"])

    assert res.loc[0, "card_uid_hist_count"] == 2
    assert res.loc[0, "card_uid_hist_mean_amt"] == 15.0
    assert res.loc[0, "card_uid_time_since_last"] == 2.0 # 4 - 2
    assert res.loc[0, "card_uid_amt_to_hist_mean"] == 2.0 # 30 / 15
