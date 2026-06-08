import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from transaction_risk_engine.config import load_config
from transaction_risk_engine.data.schema import TARGET
from transaction_risk_engine.models.baselines import train_logistic_regression
from transaction_risk_engine.models.evaluate import calculate_metrics, plot_curves
from transaction_risk_engine.models.lgbm import train_lightgbm
from transaction_risk_engine.data.load import resolve_data_paths


def format_report_table(metrics_logreg: dict[str, float], metrics_lgbm: dict[str, float]) -> str:
    """Format model comparison as a Markdown table."""
    keys = list(metrics_logreg.keys())
    
    table = "| Metric | Logistic Regression (Baseline) | LightGBM |\n"
    table += "|---|---|---|\n"
    
    for k in keys:
        name = k.replace("_", " ").title()
        val_logreg = f"{metrics_logreg[k]:.4f}"
        val_lgbm = f"{metrics_lgbm[k]:.4f}"
        table += f"| {name} | {val_logreg} | {val_lgbm} |\n"
        
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_data_paths(config)

    print("Loading engineered features and targets...")
    X_train = pd.read_parquet(paths.processed_dir / "features_train.parquet")
    X_valid = pd.read_parquet(paths.processed_dir / "features_valid.parquet")
    
    y_train = pd.read_parquet(paths.processed_dir / "target_train.parquet")[TARGET]
    y_valid = pd.read_parquet(paths.processed_dir / "target_valid.parquet")[TARGET]

    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

    # 1. Train and Evaluate Logistic Regression Baseline
    print("\n--- Training Logistic Regression Baseline ---")
    logreg_model = train_logistic_regression(X_train, y_train)
    y_prob_logreg = logreg_model.predict_proba(X_valid)[:, 1]
    
    metrics_logreg = calculate_metrics(y_valid, y_prob_logreg)
    print("Logistic Regression Metrics on Validation:")
    for k, v in metrics_logreg.items():
        print(f"  {k}: {v:.4f}")
        
    plot_curves(y_valid, y_prob_logreg, "logreg_baseline", paths.reports_dir)

    # 2. Train and Evaluate LightGBM
    print("\n--- Training LightGBM Model ---")
    lgbm_model = train_lightgbm(X_train, y_train, X_valid, y_valid, config)
    y_prob_lgbm = lgbm_model.predict_proba(X_valid)[:, 1]
    
    metrics_lgbm = calculate_metrics(y_valid, y_prob_lgbm)
    print("LightGBM Metrics on Validation:")
    for k, v in metrics_lgbm.items():
        print(f"  {k}: {v:.4f}")
        
    plot_curves(y_valid, y_prob_lgbm, "lgbm", paths.reports_dir)

    # 3. Save Artifacts
    print("\n--- Saving Artifacts ---")
    
    # Save the LightGBM model as the champion
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lgbm_model.joblib"
    joblib.dump(lgbm_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # Save metrics JSON
    all_metrics = {
        "logreg_baseline": metrics_logreg,
        "lgbm": metrics_lgbm
    }
    metrics_path = paths.reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Generate Markdown Report
    report_path = paths.reports_dir / "model_report.md"
    with open(report_path, "w") as f:
        f.write("# Baseline Model Evaluation Report\n\n")
        f.write("This report compares the simple Logistic Regression baseline against the primary LightGBM model on the holdout validation set.\n\n")
        f.write("## Metrics Comparison\n\n")
        f.write(format_report_table(metrics_logreg, metrics_lgbm) + "\n\n")
        f.write("## PR and ROC Curves\n\n")
        f.write("![LightGBM PR Curve](lgbm_pr_curve.png)\n\n")
        f.write("![LightGBM ROC Curve](lgbm_roc_curve.png)\n")
    print(f"Saved markdown report to {report_path}")

    print("\nTraining Pipeline Complete!")


if __name__ == "__main__":
    main()
