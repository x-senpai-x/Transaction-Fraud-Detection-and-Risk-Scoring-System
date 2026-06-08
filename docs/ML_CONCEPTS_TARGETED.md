# ML and Technical Concepts Targeted by Transaction Risk Engine

Transaction Risk Engine is designed to signal that you can build an end-to-end ML system, not just train a model in a notebook.

## 1. Core machine learning concepts

### Binary classification

The target is whether a transaction is fraudulent or not. The model outputs a probability of fraud.

### Imbalanced classification

Fraud is rare, so a model can achieve high accuracy by predicting every transaction as non-fraud. This project focuses on PR-AUC, Recall@Top-K, precision at threshold, and cost-sensitive evaluation.

### Tabular ML

The primary data format is structured tabular data. This tests practical skill in cleaning, encoding, feature engineering, and tree-based models.

### Gradient boosting

LightGBM/XGBoost should be the main model because boosted trees are strong on tabular data. The project should compare a simple baseline against boosting.

### Baselines and ablations

You should show progression:

1. logistic regression / simple model,
2. LightGBM with raw/basic features,
3. LightGBM with frequency + historical features,
4. LightGBM with velocity/graph features if implemented.

This proves that each feature block adds value.

## 2. Feature engineering concepts

### Frequency encoding

High-cardinality categorical values such as card IDs, address IDs, and email domains are converted into frequency/count features.

### Historical aggregations

For each entity, compute prior behavior:

- historical count,
- mean amount,
- standard deviation,
- max amount,
- amount ratio to entity average,
- time since previous transaction.

### Velocity features

Measure recent bursts of activity:

- card transactions in last 1h/24h,
- device transactions in last 1h/24h,
- unique cards per device recently,
- rapid repeated attempts.

### Entity relationship features

Create proxy entities such as card, device, email domain, and address. Then capture relationships like:

- one device used by many cards,
- one card used by many devices,
- one email domain connected to unusual transaction clusters.

## 3. Data leakage and validation concepts

### Point-in-time correctness

Historical features must use only information available before the current transaction. This is one of the most important concepts in the project.

### Time-based splitting

Train on earlier transactions and validate/test on later transactions. Do not use random split for final results.

### Training-serving skew

The feature logic used during model training should match the logic used during API inference. The project should save preprocessing artifacts and feature names.

## 4. Evaluation concepts

### Precision-recall tradeoff

A fraud model should be evaluated by how many fraudulent transactions it catches at acceptable false-positive levels.

### PR-AUC

Primary metric for imbalanced classification.

### ROC-AUC

Useful secondary metric but less informative than PR-AUC under heavy imbalance.

### Recall@Top-K

Measures how many frauds appear among the highest-risk transactions. This is business-relevant because fraud teams often review only the highest-scored transactions.

### Cost-sensitive thresholding

Convert model probability into action:

- approve,
- manual review,
- block.

The threshold should consider false-positive cost and fraud-loss cost.

## 5. Explainability concepts

### SHAP values

Explain individual predictions by showing which features increased or decreased risk.

### Global feature importance

Aggregate SHAP values or model importances to show what drives risk generally.

### Human-readable reasoning

Translate model outputs into reasons such as:

- amount is much higher than historical card average,
- device is linked to many unique cards,
- high recent transaction velocity.

## 6. MLOps and production ML concepts

### Model serialization

Save model, preprocessors, feature names, and config so inference is reproducible.

### API-based inference

Expose `/predict` and `/batch_predict` endpoints using FastAPI.

### Monitoring

Track prediction distribution, feature distributions, and drift metrics.

### Drift detection

Use PSI/KS tests to detect when incoming data differs from training data.

### Champion-challenger evaluation

Compare current production model against a newly trained candidate model without automatically replacing it.

## 7. Software engineering concepts

### Modular architecture

Separate data loading, feature engineering, training, inference, dashboard, and monitoring.

### Testing

Unit-test feature functions, splits, inference, and API routes.

### Configuration management

Use YAML/env files for paths, model params, thresholds, and output locations.

### Reproducibility

Use fixed seeds, saved configs, and deterministic scripts.

### Documentation and demoability

The project should be understandable from README, runnable with commands, and demoable in under five minutes.

## 8. Finance/domain concepts

### Transaction risk scoring

The model scores the risk of a specific transaction, not a customer’s long-term creditworthiness.

### Fraud vs credit default

Credit default: will this person repay later?

Transaction Risk Engine: does this transaction look suspicious right now?

### False positives and user friction

Blocking legitimate payments hurts customer experience. The project should discuss the false-positive tradeoff.

### Manual review workflow

Not every suspicious transaction should be blocked. Many should be routed to review or step-up authentication.
