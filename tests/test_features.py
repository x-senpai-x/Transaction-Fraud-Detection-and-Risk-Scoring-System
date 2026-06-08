import numpy as np
import pandas as pd

from transaction_risk_engine.features.base import build_base_features
from transaction_risk_engine.features.build import get_feature_matrix
from transaction_risk_engine.features.frequency import fit_frequency_encoder, transform_frequency


def test_build_base_features():
    df = pd.DataFrame({
        "TransactionAmt": [10.5, 20.0],
        "relative_hour": [1, 2],
        "relative_day": [10, 11],
        "addr1": [100, np.nan],
        "id_01": [1.0, np.nan],
        "M1": ["T", "F"],
        "ProductCD": ["W", "C"],
        "V1": [0.1, 0.2]
    })
    
    out = build_base_features(df)
    
    # Amount
    np.testing.assert_allclose(out["TransactionAmt_log"], np.log1p([10.5, 20.0]))
    np.testing.assert_allclose(out["TransactionAmt_decimal"], [0.5, 0.0])
    
    # Missingness
    assert list(out["addr1_missing"]) == [0, 1]
    
    # Identity
    assert list(out["has_identity"]) == [1, 0]
    
    # M columns
    assert list(out["M1_encoded"]) == [1, 0]
    
    # Label encode (W is after C, so C=0, W=1)
    assert list(out["ProductCD_encoded"]) == [1, 0]


def test_frequency_encoding():
    train_df = pd.DataFrame({"card1": ["A", "A", "B", np.nan]})
    valid_df = pd.DataFrame({"card1": ["A", "B", "C", np.nan]})
    
    freq_maps = fit_frequency_encoder(train_df, ["card1"])
    assert freq_maps["card1"]["A"] == 2
    assert freq_maps["card1"]["B"] == 1
    assert freq_maps["card1"][np.nan] == 1
    
    out_valid = transform_frequency(valid_df, freq_maps)
    
    # Unseen C gets 0, A gets 2, B gets 1, nan gets 1
    assert list(out_valid["card1_freq"]) == [2, 1, 0, 1]


def test_get_feature_matrix():
    df = pd.DataFrame({
        "isFraud": [0, 1],
        "TransactionID": [1, 2],
        "TransactionDT": [100, 200],
        "relative_day": [1, 1],
        "relative_hour": [1, 2],
        "card1": ["A", "B"], # string, should be dropped
        "V1": [1.0, 2.0], # numeric, kept
        "card1_freq": [5, 2], # numeric feature, kept
    })
    
    feat_df = get_feature_matrix(df)
    assert "isFraud" not in feat_df.columns
    assert "TransactionID" not in feat_df.columns
    assert "TransactionDT" not in feat_df.columns
    assert "card1" not in feat_df.columns
    
    assert "V1" in feat_df.columns
    assert "card1_freq" in feat_df.columns

from transaction_risk_engine.features.historical import build_historical_features
from transaction_risk_engine.data.schema import PROXY_ID_COLUMNS
