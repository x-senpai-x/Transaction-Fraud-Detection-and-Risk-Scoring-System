import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> dict[str, float]:
    """Calculate imbalanced classification metrics including Recall@Top-1%."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    metrics = {}

    # Standard AUC metrics
    metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    # Top-K% metrics (Recall and Precision @ Top 1%)
    n_total = len(y_true)
    n_top_1_percent = max(1, int(n_total * 0.01))

    # Sort descending by probability
    sort_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sort_indices]

    y_true_top_1 = y_true_sorted[:n_top_1_percent]

    total_frauds = y_true.sum()
    frauds_in_top_1 = y_true_top_1.sum()

    metrics["recall_at_top_1_pct"] = frauds_in_top_1 / total_frauds if total_frauds > 0 else 0.0
    metrics["precision_at_top_1_pct"] = frauds_in_top_1 / n_top_1_percent if n_top_1_percent > 0 else 0.0

    return metrics


def plot_curves(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    model_name: str,
    output_dir: str | Path
) -> None:
    """Generate and save PR and ROC curves."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{model_name}_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", color="darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{model_name}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
