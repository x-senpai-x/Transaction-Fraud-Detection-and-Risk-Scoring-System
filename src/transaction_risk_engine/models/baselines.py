import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray
) -> Pipeline:
    """Train a Logistic Regression baseline model with median imputation and scaling."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_train, np.asarray(y_train).ravel())
    return pipeline
