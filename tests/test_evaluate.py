import numpy as np

from transaction_risk_engine.models.evaluate import calculate_metrics


def test_calculate_metrics():
    # Synthetic array of 100 predictions.
    # 5 ones, 95 zeros.
    y_true = np.zeros(100)
    y_true[:5] = 1
    
    # y_prob ranks the first 2 as highest, then 1 false positive as 3rd, then next 3 are lower.
    y_prob = np.zeros(100)
    y_prob[0] = 0.99
    y_prob[1] = 0.98
    y_prob[5] = 0.97 # False positive
    y_prob[2] = 0.90
    y_prob[3] = 0.80
    y_prob[4] = 0.70
    
    metrics = calculate_metrics(y_true, y_prob)
    
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "recall_at_top_1_pct" in metrics
    assert "precision_at_top_1_pct" in metrics
    
    # Top 1% of 100 = 1 record.
    # Top 1 probability is y_prob[0] = 0.99, which is y_true[0] = 1.
    # So precision at top 1% should be 1.0 (1 out of 1)
    # Recall at top 1% should be 1 / 5 = 0.2
    assert np.isclose(metrics["precision_at_top_1_pct"], 1.0)
    assert np.isclose(metrics["recall_at_top_1_pct"], 0.2)
