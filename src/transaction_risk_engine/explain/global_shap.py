import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    model_path = Path("models/lgbm_model.joblib")
    features_path = Path("data/processed/features_valid.parquet")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists() or not features_path.exists():
        print("Required artifacts missing. Ensure Phase 3 is completed.")
        return

    print("Loading model and validation features...")
    model = joblib.load(model_path)
    X_valid = pd.read_parquet(features_path)

    # Convert object/string columns to numeric (coerce errors to NaN) since LightGBM handles floats better
    for col in X_valid.columns:
        X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce")

    # Sample to keep computation fast and plot readable
    if len(X_valid) > args.sample_size:
        X_sample = X_valid.sample(n=args.sample_size, random_state=42)
    else:
        X_sample = X_valid

    print(f"Computing SHAP values for {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM objective='binary' returns a list of 2 arrays (negative class, positive class)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    print("Generating SHAP summary dot plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Global SHAP explanation plots saved to {reports_dir}")


if __name__ == "__main__":
    main()
