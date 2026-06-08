from typing import Any

import pandas as pd
import shap


class TreeExplainerWrapper:
    """Wrapper around SHAP TreeExplainer for generating top features."""

    def __init__(self, model: Any, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def explain(self, row: pd.Series, top_k: int = 3) -> list[str]:
        """Return the top K feature names that contributed positively to the fraud risk."""
        # SHAP expects 2D array or dataframe for single row usually, but we can pass a dataframe
        df = pd.DataFrame([row], columns=self.feature_names)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(df)
        
        # Depending on LightGBM version/objective, shap_values might be a list of arrays (one per class) 
        # or a single array. For binary classification, sometimes it's list[1].
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Take class 1, first row
        else:
            sv = shap_values[0]  # single row

        # Create pairs of (feature_name, shap_value)
        contributions = list(zip(self.feature_names, sv))
        
        # Sort by SHAP value descending (highest positive impact on fraud)
        contributions.sort(key=lambda x: x[1], reverse=True)

        top_reasons = []
        for feat_name, val in contributions[:top_k]:
            if val > 0:
                # Format a nice readable reason
                top_reasons.append(f"{feat_name} showed elevated risk characteristics (impact: {val:.3f})")
                
        if not top_reasons:
            top_reasons.append("No strong individual risk signals detected.")
            
        return top_reasons
