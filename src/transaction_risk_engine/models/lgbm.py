from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series | np.ndarray,
    config: dict[str, Any]
) -> lgb.LGBMClassifier:
    """Train a LightGBM model with early stopping on validation PR-AUC."""
    # We use basic LightGBM parameters suitable for a baseline.
    seed = config.get("project", {}).get("seed", 42)
    
    # Scale positive weights if highly imbalanced
    y_train_arr = np.asarray(y_train).ravel()
    n_pos = max(1, y_train_arr.sum())
    n_neg = len(y_train_arr) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        objective="binary",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train_arr,
        eval_set=[(X_valid, np.asarray(y_valid).ravel())],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
        ]
    )

    return model
