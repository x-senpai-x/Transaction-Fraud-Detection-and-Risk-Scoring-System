import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from transaction_risk_engine.models.baselines import train_logistic_regression
from transaction_risk_engine.models.lgbm import train_lightgbm


def test_train_logistic_regression():
    X_train = pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0],
        "feat2": [0.1, 0.2, 0.3, 0.4]
    })
    y_train = pd.Series([0, 1, 0, 1])
    
    model = train_logistic_regression(X_train, y_train)
    
    assert isinstance(model, Pipeline)
    
    # Check predictions
    probs = model.predict_proba(X_train)
    assert probs.shape == (4, 2)


def test_train_lightgbm():
    X_train = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feat2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    
    X_valid = pd.DataFrame({
        "feat1": [1.5, 2.5],
        "feat2": [0.15, 0.25]
    })
    y_valid = pd.Series([0, 1])
    
    config = {"project": {"seed": 42}}
    
    model = train_lightgbm(X_train, y_train, X_valid, y_valid, config)
    
    assert isinstance(model, lgb.LGBMClassifier)
    
    # Check predictions
    probs = model.predict_proba(X_valid)
    assert probs.shape == (2, 2)
