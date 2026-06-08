# Agent 03 — Model Training

## Mission

Train and evaluate a strong fraud classifier using imbalance-aware metrics.

## Inputs

- processed train/valid/test feature files,
- target column,
- feature metadata,
- model config.

## Outputs

- `models/baseline_logreg.joblib`
- `models/lgbm_model.joblib` or `models/xgb_model.joblib`
- `reports/model_report.md`
- `reports/metrics.json`
- `reports/pr_curve.png`
- `reports/roc_curve.png`

## Models

1. Logistic regression baseline.
2. LightGBM or XGBoost main model.

## Metrics

Primary:

- PR-AUC
- Recall@Top-1%
- Precision@Top-1%

Secondary:

- ROC-AUC
- confusion matrix at chosen threshold
- false-positive rate
- false-negative rate

## Thresholding

Convert probability into:

- `APPROVE`
- `REVIEW`
- `BLOCK`

Baseline thresholds can be simple, but final threshold should be justified by precision/recall or cost tradeoff.

## Acceptance criteria

- Training script runs from CLI.
- Metrics saved to JSON and Markdown.
- Model artifact and feature list saved.
- No invented metrics.

## Suggested command

```bash
python -m transaction_risk_engine.models.train --config config/config.yaml
```

## Agent prompt

```txt
Read AGENTS.md and docs/ML_CONCEPTS_TARGETED.md.
Implement model training and evaluation.
Train logistic regression and LightGBM/XGBoost.
Report PR-AUC, ROC-AUC, Recall@Top-1%, Precision@Top-1%.
Save model and metrics.
Do not use accuracy as the headline metric.
```
