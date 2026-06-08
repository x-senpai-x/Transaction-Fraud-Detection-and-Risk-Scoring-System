import json
from pathlib import Path

import joblib
import pandas as pd

from transaction_risk_engine.data.load import add_proxy_ids, add_relative_time_columns
from transaction_risk_engine.features.base import build_base_features
from transaction_risk_engine.features.frequency import transform_frequency
from transaction_risk_engine.features.historical import apply_historical_features
from transaction_risk_engine.data.schema import PROXY_ID_COLUMNS


class FraudRiskPredictor:
    """End-to-end inference pipeline for the Transaction Risk Engine."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "lgbm_model.joblib")

        with open(self.model_dir / "frequency_maps.json", "r") as f:
            self.freq_maps = json.load(f)

        with open(self.model_dir / "historical_state.json", "r") as f:
            self.hist_state = json.load(f)

        with open(self.model_dir / "feature_metadata.json", "r") as f:
            metadata = json.load(f)
            self.expected_features = metadata["features"]
            
        # Optional: Explainer for SHAP
        self.explainer = None
        try:
            from transaction_risk_engine.explain.shap_explainer import TreeExplainerWrapper
            self.explainer = TreeExplainerWrapper(self.model, self.expected_features)
        except ImportError:
            pass

    def predict(self, transaction_dict: dict) -> dict:
        """Score a single transaction dictionary."""
        # 1. Convert to DataFrame (1 row)
        df = pd.DataFrame([transaction_dict])

        # 2. Pipeline transforms
        df = add_proxy_ids(df)
        df = add_relative_time_columns(df)
        df = build_base_features(df)
        df = transform_frequency(df, self.freq_maps)
        df = apply_historical_features(df, self.hist_state, PROXY_ID_COLUMNS)

        # 3. Align features to exactly what the model expects
        # Add missing columns with None/NaN, and reorder
        missing_cols = [col for col in self.expected_features if col not in df.columns]
        if missing_cols:
            df_missing = pd.DataFrame({col: pd.NA for col in missing_cols}, index=df.index)
            df = pd.concat([df, df_missing], axis=1)

        X_inf = df[self.expected_features].copy()
        
        # Ensure correct dtypes (LightGBM handles objects terribly, so convert to float if numeric)
        for col in X_inf.columns:
            # We can just rely on pandas astype float where applicable, but LightGBM is okay with numeric
            X_inf[col] = pd.to_numeric(X_inf[col], errors="coerce")

        # 4. Predict
        prob = float(self.model.predict_proba(X_inf)[0, 1])
        
        # Scale raw probabilities (which max out around ~0.13 due to extreme class imbalance)
        # into a human-readable 0-100 risk score for the dashboard demo.
        # Base rate is ~0.035, max is ~0.13. 
        risk_score = int(min(100, max(0, (prob - 0.03) * 1000)))

        # Determine decision based on scaled score
        if risk_score > 75:
            decision = "BLOCK"
        elif risk_score > 40:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        response = {
            "fraud_probability": round(prob, 4),
            "risk_score": risk_score,
            "decision": decision,
            "top_reasons": []
        }

        # 5. Explain if available
        if self.explainer is not None:
            reasons = self.explainer.explain(X_inf.iloc[0])
            response["top_reasons"] = reasons

        return response
