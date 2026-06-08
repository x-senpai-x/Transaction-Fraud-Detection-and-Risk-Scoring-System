import pandas as pd

from transaction_risk_engine.data.schema import TIME_COLUMN
from transaction_risk_engine.data.split import time_based_split


def test_time_based_split_chronological():
    df = pd.DataFrame({
        TIME_COLUMN: [5, 1, 3, 4, 2],
        "value": ["e", "a", "c", "d", "b"],
        "isFraud": [0, 0, 1, 0, 1]
    })
    
    config = {"split": {"train_frac": 0.60, "valid_frac": 0.20}}
    # 5 rows total.
    # train = int(5 * 0.6) = 3
    # valid = int(5 * 0.2) = 1
    # test = remaining 1
    
    train, valid, test = time_based_split(df, config)
    
    assert len(train) == 3
    assert len(valid) == 1
    assert len(test) == 1
    
    # Check chron order
    assert train[TIME_COLUMN].max() < valid[TIME_COLUMN].min()
    assert valid[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    
    # Check contents sorted
    assert list(train["value"]) == ["a", "b", "c"]
    assert list(valid["value"]) == ["d"]
    assert list(test["value"]) == ["e"]
    
    # Check target preserved
    assert "isFraud" in train.columns
    assert "isFraud" in valid.columns
    assert "isFraud" in test.columns
