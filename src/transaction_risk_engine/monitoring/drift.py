import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI)."""
    
    # Define decile thresholds from the expected (reference) distribution
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    q = np.percentile(expected, breakpoints)
    
    # Handle duplicate bins (e.g. lots of zeros)
    q = np.unique(q)
    # Ensure edges cover everything
    q[0] = -np.inf
    q[-1] = np.inf
    
    # Calculate counts
    expected_counts, _ = np.histogram(expected, bins=q)
    actual_counts, _ = np.histogram(actual, bins=q)
    
    # Calculate percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid div by zero
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    drift_report_path = reports_dir / "drift_report.json"

    print("Loading model and data...")
    model = joblib.load("models/lgbm_model.joblib")
    
    train_df = pd.read_parquet("data/processed/features_train.parquet")
    test_df = pd.read_parquet("data/processed/features_test.parquet")

    # Coerce to numeric just in case
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # Sample to keep execution quick
    sample_size = 20000
    if len(train_df) > sample_size:
        train_sample = train_df.sample(n=sample_size, random_state=42)
    else:
        train_sample = train_df
        
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df

    print("Generating predictions...")
    train_preds = model.predict_proba(train_sample)[:, 1]
    test_preds = model.predict_proba(test_sample)[:, 1]

    print("Calculating Prediction PSI...")
    prediction_psi = calculate_psi(train_preds, test_preds, buckets=10)
    
    # Determine drift status
    if prediction_psi < 0.1:
        status = "Stable"
    elif prediction_psi < 0.2:
        status = "Moderate Drift"
    else:
        status = "Severe Drift"

    print(f"PSI: {prediction_psi:.4f} ({status})")

    print("Calculating KS Test for top features...")
    # Read metadata to get ordered features. Just grab first 5 numerics for demo
    # We will grab common known numerical features for stability
    test_features = ["TransactionAmt", "card1", "V310", "C1", "C13"]
    ks_results = {}
    
    for feature in test_features:
        if feature in train_sample.columns and feature in test_sample.columns:
            train_vals = train_sample[feature].dropna()
            test_vals = test_sample[feature].dropna()
            
            # Subsample for KS to prevent heavy compute
            stat, p_val = ks_2samp(train_vals.sample(min(5000, len(train_vals)), random_state=42), 
                                   test_vals.sample(min(5000, len(test_vals)), random_state=42))
            
            ks_results[feature] = {
                "ks_statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": bool(p_val < 0.05)
            }

    report = {
        "prediction_drift": {
            "psi": prediction_psi,
            "status": status,
            "reference_size": len(train_sample),
            "current_size": len(test_sample)
        },
        "feature_drift_ks": ks_results
    }

    with open(drift_report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift report saved to {drift_report_path}")


if __name__ == "__main__":
    main()
