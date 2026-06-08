# Project Scope

## Goal

Build a resume-ready project that is broad enough for ML/DS/SDE roles and still relevant for finance/fintech interviews.

## Best project identity

**Transaction Risk Engine**

A production-style system that scores online transactions using behavioral features, entity relationships, explainability, and monitoring.

## Baseline scope

The baseline should be finished before adding advanced experiments.

### Included in baseline

- IEEE-CIS dataset ingestion.
- Time-based split.
- Basic EDA.
- Frequency encoding.
- Historical aggregations.
- LightGBM/XGBoost model.
- PR-AUC, ROC-AUC, Recall@Top-K.
- Risk threshold into approve/review/block.
- FastAPI inference.
- Streamlit dashboard.
- SHAP explanations.
- Clean README.

### Excluded from baseline

- Real Kafka/Flink.
- Full feature store.
- True online learning.
- GNN as main model.
- Kubernetes.
- Production-grade auth/security.
- Real payment processing.

## Advanced scope

Add if time allows:

- velocity features,
- graph-derived features,
- drift monitor,
- champion-challenger comparison,
- pseudo-streaming replay,
- Node2Vec/GraphSAGE experiment.

## Final demo story

The recruiter should see:

1. A transaction comes in.
2. The system builds risk features.
3. The model returns probability and decision.
4. The dashboard explains why the transaction was risky.
5. Monitoring shows whether the data distribution has shifted.
